"""
Experiment 2 — Causal Steering: Pre-Thinking vs Mid-Thinking
=============================================================
Paper: "Do Thinking Tokens Think? Mechanistic Evidence for Pre-Commitment
        and Causal Updating in OLMo-3-7B Extended Reasoning"

What this does
--------------
Uses the probe direction learned in Exp 1 as a steering vector and injects it
at five positions across the thinking trace:

    pos 0%   — last question token      (pre-thinking boundary)
    pos 25%  — 25% through thinking
    pos 50%  — halfway through thinking
    pos 75%  — late thinking
    pos 100% — last thinking token      (just before answer)

For each question and each wrong target answer class we:
  1. Compute the diff-of-means steering vector for that wrong class at layer 31.
  2. Inject the vector at each of the five positions (additive, scaled).
  3. Record whether the final answer flips to the steered class.

The flip rate tells us: does thinking causally update the residual stream,
or is the pre-committed answer sticky through the entire chain?

Expected finding
----------------
  Easy tasks:  flip rate flat across injection positions  (thinking decorative)
  Hard tasks:  flip rate falls for later positions       (thinking competes with steer)
  → Thinking is computationally significant for hard tasks only.

Outputs  ./outputs/exp2/
    steering_vectors.npz        diff-of-means vectors per wrong class
    steering_results.json       flip rates per position × class × difficulty
    figures/fig3_fliprate_curve.pdf
    figures/fig4_pre_vs_mid.pdf

Dependencies
    This script REQUIRES Exp 1 outputs:
        outputs/exp1/activations_prethinkin.npz
        outputs/exp1/answers.json
        outputs/exp1/labelled_dataset.json

    pip install nnsight>=0.6 transformers>=4.57 datasets matplotlib numpy torch
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID         = "allenai/Olmo-3-7B-Think"
STEER_LAYER      = 31                        # layer to inject steering vector
PROBE_LAYERS     = [8, 16, 24, 31]
THINK_MAX_TOKENS = 512
ANSWER_MAX_TOKENS = 8
N_STEER_PER_TIER = 200                       # questions per difficulty tier
STEER_SCALE      = 15.0                      # additive scaling for steering
POSITION_FRACS   = [0.0, 0.25, 0.50, 0.75, 1.0]   # where to inject
SEED             = 42

EXP1_DIR = Path("outputs/exp1")
OUT_DIR  = Path("outputs/exp2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHOICES   = ["A", "B", "C", "D"]
LABEL2IDX = {c: i for i, c in enumerate(CHOICES)}


# ── Helpers ────────────────────────────────────────────────────────────────

def extract_answer(decoded: str) -> int:
    after = decoded.split("</think>")[-1] if "</think>" in decoded else decoded
    match = re.findall(r"\b([ABCD])\b", after)
    return LABEL2IDX.get(match[0], 0) if match else 0


def find_think_end_token(token_ids: list[int], tokenizer) -> int:
    """
    Return the token index of '</think>' in the generated sequence.
    If not found, return len(token_ids) - 1.
    """
    think_end = tokenizer.encode("</think>", add_special_tokens=False)
    for i in range(len(token_ids) - len(think_end) + 1):
        if token_ids[i:i + len(think_end)] == think_end:
            return i + len(think_end) - 1
    return len(token_ids) - 1


# ── 1.  Compute steering vectors ──────────────────────────────────────────

def build_steering_vectors(
    acts: np.ndarray,
    records: list[dict],
) -> dict[int, np.ndarray]:
    """
    Diff-of-means steering vectors at STEER_LAYER.
    For each target class c:
        v_c = mean(acts[label==c, layer31]) - mean(acts[label!=c, layer31])
        v_c = v_c / ||v_c||

    Returns {class_idx: unit_vector (4096,)}
    """
    li     = PROBE_LAYERS.index(STEER_LAYER)
    labels = np.array([r["label_idx"] for r in records])
    vecs   = {}
    for c in range(4):
        pos = acts[labels == c, li, :]
        neg = acts[labels != c, li, :]
        diff = pos.mean(0) - neg.mean(0)
        diff /= (np.linalg.norm(diff) + 1e-8)
        vecs[c] = diff.astype(np.float32)
        print(f"  class {CHOICES[c]}: mean-diff norm={np.linalg.norm(diff):.4f}  "
              f"(from {len(pos)} positive, {len(neg)} negative examples)")
    return vecs


# ── 2.  Two-pass injection ─────────────────────────────────────────────────
#
# nnsight's multi-token generation API:
#   model.generate(...) opens a tracer.
#   tracer.iter[:] loops over generation steps.
#   Inside the loop, `step` = current step index.
#   We inject by setting model.model.layers[L].output[0] += steering_delta
#   at the step corresponding to our desired position fraction.
#
# Position fractions 0–1 are resolved via a two-pass approach:
#   Pass 1: generate *without* steering → record total thinking length T
#           (number of tokens before </think>)
#   Pass 2: inject at step = round(frac * T)
#
# For pos 0.0 (pre-thinking) we inject during the prefill of the prompt itself,
# using model.trace() instead of model.generate().


def steer_at_position(
    model: LanguageModel,
    prompt: str,
    steer_vec: np.ndarray,
    inject_frac: float,
    think_length: int,
) -> int:
    """
    Generate with steering vector added to layer STEER_LAYER at the
    token position corresponding to inject_frac * think_length.

    Returns predicted answer index (0-3).
    """
    steer_tensor = torch.tensor(steer_vec, dtype=torch.bfloat16).to("cuda")
    delta = STEER_SCALE * steer_tensor     # shape (4096,)
    inject_step = round(inject_frac * (think_length - 1))

    if inject_frac == 0.0:
        # ── Pre-thinking injection: modify during the prefill trace ────────
        # We run model.trace() (one forward pass) to inject at the last prompt
        # token, then run model.generate() to get the answer.
        # However nnsight does not easily chain trace→generate with modified KV
        # cache state.  Instead we use the generation loop and inject at step 0
        # (the first newly-generated token, which is the first <think> token).
        inject_step = 0

    # ── Injection via generation loop ─────────────────────────────────────
    # We iterate over generation steps.  At `inject_step` we add delta to the
    # hidden state of layer STEER_LAYER at the last sequence position.
    with model.generate(
        prompt,
        max_new_tokens=THINK_MAX_TOKENS + ANSWER_MAX_TOKENS,
        do_sample=False,
    ) as tracer:
        for step in tracer.iter[:]:
            if step == inject_step:
                # output[0]: (batch=1, seq_len, hidden)
                # At generation step k, seq_len=1 (autoregressive), so [:, -1, :]
                # adds the delta to the single new token's hidden state.
                hs = model.model.layers[STEER_LAYER].output[0]
                model.model.layers[STEER_LAYER].output[0] = hs + delta.unsqueeze(0).unsqueeze(0)

        out = model.generator.output.save()

    decoded = model.tokenizer.decode(out[0], skip_special_tokens=False)
    return extract_answer(decoded)


def measure_think_length(
    model: LanguageModel,
    prompt: str,
) -> tuple[int, str]:
    """
    Generate once without intervention.
    Return (think_length_in_tokens, full_decoded_text).
    think_length = number of newly generated tokens before and including </think>.
    """
    with model.generate(
        prompt,
        max_new_tokens=THINK_MAX_TOKENS + ANSWER_MAX_TOKENS,
        do_sample=False,
    ) as tracer:
        out = model.generator.output.save()

    token_ids = out[0].tolist()
    # The prompt tokens are at the start; new tokens start after len(prompt_ids)
    prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = token_ids[len(prompt_ids):]
    think_end_idx = find_think_end_token(new_ids, model.tokenizer)
    think_length = max(1, think_end_idx + 1)   # at least 1 to avoid /0
    full_text = model.tokenizer.decode(token_ids, skip_special_tokens=False)
    return think_length, full_text


# ── 3.  Main steering loop ────────────────────────────────────────────────

def run_steering_experiment(
    model: LanguageModel,
    dataset: list[dict],
    steering_vecs: dict[int, np.ndarray],
    records: list[dict],
    n_per_tier: int,
) -> list[dict]:
    """
    For each question (up to n_per_tier per difficulty tier):
      - Measure base think_length (pass 1)
      - For each wrong target class, inject at each position fraction
      - Record whether answer flips to the target class

    Returns list of result dicts.
    """
    # Select subset: n_per_tier per difficulty
    tier_counts = {"easy": 0, "medium": 0, "hard": 0}
    selected    = []
    for ex, rec in zip(dataset, records):
        d = rec["difficulty"]
        if tier_counts[d] < n_per_tier:
            selected.append((ex, rec))
            tier_counts[d] += 1
        if all(v >= n_per_tier for v in tier_counts.values()):
            break

    print(f"\n=== Steering experiment: {len(selected)} questions "
          f"(~{n_per_tier} per tier) ===")

    results = []
    for qi, (ex, rec) in enumerate(selected):
        prompt     = ex["prompt"]
        gt         = rec["label_idx"]
        difficulty = rec["difficulty"]

        # ── Pass 1: baseline generate → think length ───────────────────────
        think_len, baseline_text = measure_think_length(model, prompt)
        baseline_answer = extract_answer(baseline_text)

        # ── Pass 2: steer toward each wrong class at each position ─────────
        flips = {}  # {wrong_class: {frac: bool}}
        wrong_classes = [c for c in range(4) if c != gt]
        for wc in wrong_classes:
            flips[wc] = {}
            for frac in POSITION_FRACS:
                steered_answer = steer_at_position(
                    model, prompt,
                    steering_vecs[wc],
                    frac, think_len,
                )
                flips[wc][frac] = int(steered_answer == wc)

        results.append({
            "qid":             ex["qid"],
            "gt":              gt,
            "baseline_answer": baseline_answer,
            "difficulty":      difficulty,
            "think_length":    think_len,
            "flips":           {str(k): {str(f): v for f, v in fv.items()}
                                 for k, fv in flips.items()},
        })

        if qi % 20 == 0:
            print(f"  {qi+1}/{len(selected)}  "
                  f"think_len={think_len}  diff={difficulty}")

    return results


# ── 4.  Aggregate metrics ──────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Returns:
        agg[difficulty][frac] = mean flip rate (averaged over wrong classes)
    """
    tiers = ("easy", "medium", "hard", "all")
    agg   = {t: {f: [] for f in POSITION_FRACS} for t in tiers}

    for r in results:
        d = r["difficulty"]
        for wc_str, frac_dict in r["flips"].items():
            for frac_str, flipped in frac_dict.items():
                frac = float(frac_str)
                agg[d][frac].append(flipped)
                agg["all"][frac].append(flipped)

    summary = {}
    for t in tiers:
        summary[t] = {}
        for frac in POSITION_FRACS:
            vals = agg[t][frac]
            summary[t][frac] = {
                "mean_flip_rate": round(float(np.mean(vals)), 4) if vals else float("nan"),
                "n": len(vals),
            }
    return summary


# ── 5.  Figures ────────────────────────────────────────────────────────────

def plot_fliprate_curve(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    tier_styles = {
        "easy":   {"color": "#2196F3", "marker": "o"},
        "medium": {"color": "#FF9800", "marker": "s"},
        "hard":   {"color": "#F44336", "marker": "^"},
    }
    x_labels = ["0%\n(pre-think)", "25%", "50%", "75%", "100%\n(post-think)"]

    for tier, style in tier_styles.items():
        rates = [summary[tier][f]["mean_flip_rate"] for f in POSITION_FRACS]
        ax.plot(range(5), rates, label=tier.capitalize(),
                color=style["color"], marker=style["marker"],
                linewidth=2, markersize=7)

    ax.set_xticks(range(5))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Injection position (fraction of thinking tokens)")
    ax.set_ylabel("Mean flip rate")
    ax.set_title("Steering Flip Rate vs. Injection Position\n(OLMo-3-7B-Think, MMLU)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_pre_vs_mid(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tiers  = ["easy", "medium", "hard"]
    labels = ["Easy", "Medium", "Hard"]
    x = np.arange(len(tiers))
    w = 0.35

    pre  = [summary[t][0.0]["mean_flip_rate"] for t in tiers]
    mid  = [summary[t][0.5]["mean_flip_rate"] for t in tiers]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, pre, w, label="Pre-thinking (0%)", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, mid, w, label="Mid-thinking (50%)", color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean flip rate")
    ax.set_title("Pre-Thinking vs Mid-Thinking Steering\n(OLMo-3-7B-Think, MMLU)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ── 6.  Entry point ────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Load Exp 1 artefacts ───────────────────────────────────────────────
    print("Loading Exp 1 outputs …")
    acts = np.load(EXP1_DIR / "activations_prethinkin.npz")["activations"]
    with open(EXP1_DIR / "answers.json")         as f: records = json.load(f)
    with open(EXP1_DIR / "labelled_dataset.json") as f: dataset = json.load(f)
    print(f"  activations: {acts.shape}   records: {len(records)}")

    # ── Build / load steering vectors ─────────────────────────────────────
    sv_path = OUT_DIR / "steering_vectors.npz"
    if sv_path.exists():
        data = np.load(sv_path)
        steer_vecs = {int(k): data[k] for k in data.files}
        print(f"Loaded cached steering vectors for classes {list(steer_vecs.keys())}")
    else:
        print("\n=== Computing steering vectors (diff-of-means, layer 31) ===")
        steer_vecs = build_steering_vectors(acts, records)
        np.savez(sv_path, **{str(k): v for k, v in steer_vecs.items()})
        print(f"Saved → {sv_path}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_ID} …")
    model = LanguageModel(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        dispatch=True,
    )
    model.eval()

    # ── Run / load steering experiment ────────────────────────────────────
    res_path = OUT_DIR / "steering_results_raw.json"
    if res_path.exists():
        with open(res_path) as f:
            raw_results = json.load(f)
        print(f"Loaded cached steering results ({len(raw_results)} records)")
    else:
        raw_results = run_steering_experiment(
            model, dataset, steer_vecs, records, N_STEER_PER_TIER
        )
        with open(res_path, "w") as f:
            json.dump(raw_results, f, indent=2)
        print(f"Saved raw results → {res_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary = aggregate(raw_results)
    with open(OUT_DIR / "steering_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print
    print(f"\n{'':>10} " + "  ".join(f"{f:.0%}".rjust(7) for f in POSITION_FRACS))
    for tier in ("easy", "medium", "hard", "all"):
        row = "  ".join(
            f"{summary[tier][f]['mean_flip_rate']:7.3f}" for f in POSITION_FRACS
        )
        print(f"{tier:>10}  {row}")

    # ── Figures ───────────────────────────────────────────────────────────
    plot_fliprate_curve(summary, fig_dir / "fig3_fliprate_curve.pdf")
    plot_pre_vs_mid(summary, fig_dir / "fig4_pre_vs_mid.pdf")

    print("\n✓  Experiment 2 complete.")


if __name__ == "__main__":
    main()
