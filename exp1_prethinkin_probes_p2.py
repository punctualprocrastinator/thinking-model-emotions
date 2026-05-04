"""
Experiment 1 — Pre-Thinking Residual Probes
============================================
Paper: "Do Thinking Tokens Think? Mechanistic Evidence for Pre-Commitment
        and Causal Updating in OLMo-3-7B Extended Reasoning"

What this does
--------------
For every MMLU question we:
  1. Run OLMo-3-7B-Think and capture the residual-stream vector at the
     LAST QUESTION TOKEN across layers {8, 16, 24, 31} during the prefill pass.
  2. Run a second full generate() call to get the model's actual final answer.
  3. Train logistic-regression probes on those vectors to predict final answers.
  4. Report AUC and accuracy split by Easy / Medium / Hard difficulty tier.

Key result: high AUC before any thinking token = pre-commitment.

Outputs  ./outputs/exp1/
    activations_prethinkin.npz    shape (N, 4, 4096)
    answers.json
    probe_results.json
    figures/fig1_auc_heatmap.pdf
    figures/fig2_umap.pdf

Requirements
    pip install nnsight>=0.6 transformers>=4.57 datasets
                scikit-learn umap-learn matplotlib numpy torch
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID         = "allenai/Olmo-3-7B-Think"
PROBE_LAYERS     = [8, 16, 24, 31]
N_PER_TIER       = 1000
THINK_MAX_TOKENS = 512
ANSWER_MAX_TOKENS = 8
BATCH_SIZE       = 8
SEED             = 42
OUT_DIR          = Path("outputs/exp1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHOICES   = ["A", "B", "C", "D"]
LABEL2IDX = {c: i for i, c in enumerate(CHOICES)}


# ── 1.  Prompt helpers ─────────────────────────────────────────────────────

def make_no_think_prompt(example: dict) -> str:
    """Direct-answer prompt — used to measure base accuracy for difficulty labelling."""
    q, opts = example["question"], example["choices"]
    return (
        "<|im_start|>system\nAnswer with a single letter A, B, C or D only. "
        "No reasoning.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {q}\n"
        f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_think_prompt(example: dict) -> str:
    """Extended-thinking prompt — production prompt for all main experiments."""
    q, opts = example["question"], example["choices"]
    return (
        "<|im_start|>system\nYou are a helpful AI. Think step by step before answering.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: {q}\n"
        f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n"
        "Answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
    )


def extract_answer(decoded: str) -> int:
    """Return 0–3 for A–D; default A=0 if nothing found."""
    # Look for the first A/B/C/D after </think>
    after = decoded.split("</think>")[-1] if "</think>" in decoded else decoded
    match = re.findall(r"\b([ABCD])\b", after)
    return LABEL2IDX.get(match[0], 0) if match else 0


# ── 2.  Difficulty labelling ───────────────────────────────────────────────

def label_difficulty(model: LanguageModel, raw_ds, n_per_tier: int) -> list[dict]:
    """
    Run each question 5× without thinking; assign difficulty from pass@5 rate.
    Returns list of {qid, prompt, label_idx, difficulty, base_acc}.
    """
    print("=== Labelling difficulty (5-run base accuracy) ===")
    labelled, counts = [], {"easy": 0, "medium": 0, "hard": 0}
    goal = n_per_tier

    for qid, ex in enumerate(raw_ds):
        if all(v >= goal for v in counts.values()):
            break

        gt = ex["answer"]
        prompt_no_think = make_no_think_prompt(ex)
        correct = 0
        for _ in range(5):
            with model.generate(prompt_no_think, max_new_tokens=4, do_sample=False) as _t:
                out = model.generator.output.save()
            decoded = model.tokenizer.decode(out[0], skip_special_tokens=True)
            if extract_answer(decoded) == gt:
                correct += 1

        base_acc = correct / 5
        diff = "easy" if base_acc >= 0.8 else ("medium" if base_acc >= 0.4 else "hard")

        if counts[diff] >= goal:
            continue

        labelled.append({
            "qid":       qid,
            "prompt":    make_think_prompt(ex),
            "label_idx": gt,
            "difficulty": diff,
            "base_acc":  base_acc,
        })
        counts[diff] += 1
        if qid % 200 == 0:
            print(f"  [{qid}] tiers={counts}")

    print(f"Difficulty labelling done: {counts}")
    return labelled


# ── 3.  Activation extraction ──────────────────────────────────────────────

def extract_activations(
    model: LanguageModel,
    dataset: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """
    Two-step per question:
      Step A (model.trace):  prefill pass → save residual stream at last token
      Step B (model.generate): full generation → get actual final answer

    OLMo-3 architecture (Llama-style):
        model.model.layers[i].output[0]  →  hidden states (batch, seq, 4096)

    We take position [:, -1, :] = last prompt token = just before <think> begins.
    """
    N       = len(dataset)
    n_l     = len(PROBE_LAYERS)
    acts    = np.zeros((N, n_l, 4096), dtype=np.float32)
    records = []

    print(f"\n=== Extracting activations for {N} questions ===")

    for i, ex in enumerate(dataset):
        prompt = ex["prompt"]

        # ── Step A: prefill trace ──────────────────────────────────────────
        # model.trace() runs one forward pass (no decoding).
        # We capture the hidden state at the last token of the prompt.
        with model.trace(prompt) as _tracer:
            saved = {}
            for li, layer_idx in enumerate(PROBE_LAYERS):
                # output[0]: hidden states tensor (batch=1, seq_len, hidden)
                # [-1] on the seq dimension = last token
                hs = model.model.layers[layer_idx].output[0][0, -1, :].save()
                saved[li] = hs

        for li in range(n_l):
            acts[i, li, :] = saved[li].float().cpu().numpy()

        # ── Step B: full generation → actual answer ────────────────────────
        with model.generate(
            prompt,
            max_new_tokens=THINK_MAX_TOKENS + ANSWER_MAX_TOKENS,
            do_sample=False,
        ) as _tracer:
            out = model.generator.output.save()

        full_text = model.tokenizer.decode(out[0], skip_special_tokens=False)
        pred_idx  = extract_answer(full_text)

        records.append({
            "qid":       ex["qid"],
            "label_idx": ex["label_idx"],
            "pred_idx":  pred_idx,
            "difficulty": ex["difficulty"],
            "correct":   int(pred_idx == ex["label_idx"]),
        })

        if i % 50 == 0:
            acc_so_far = np.mean([r["correct"] for r in records])
            print(f"  {i+1}/{N}  running acc={acc_so_far:.3f}")

    return acts, records


# ── 4.  Probe training ─────────────────────────────────────────────────────

def run_probes(acts: np.ndarray, records: list[dict]) -> dict:
    """
    Logistic regression probe at each (layer × difficulty_tier).
    5-fold stratified CV; metric = macro OvR AUC.

    Returns nested dict:  results[tier][layer_idx] = {auc, accuracy, n}
    """
    labels = np.array([r["label_idx"] for r in records])
    diffs  = [r["difficulty"] for r in records]
    results = {}

    for tier in ("easy", "medium", "hard", "all"):
        results[tier] = {}
        mask = np.ones(len(records), dtype=bool) if tier == "all" \
               else np.array([d == tier for d in diffs])
        X_t, y_t = acts[mask], labels[mask]
        n_t = len(y_t)

        if n_t < 20:
            continue

        for li, layer_idx in enumerate(PROBE_LAYERS):
            X = X_t[:, li, :]
            skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            probs = np.zeros((n_t, 4), dtype=np.float32)
            preds = np.zeros(n_t, dtype=int)

            for tr, va in skf.split(X, y_t):
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver="lbfgs",
                    multi_class="multinomial", random_state=SEED,
                )
                clf.fit(X[tr], y_t[tr])
                probs[va] = clf.predict_proba(X[va])
                preds[va] = clf.predict(X[va])

            y_bin = label_binarize(y_t, classes=[0, 1, 2, 3])
            try:
                auc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
            except ValueError:
                auc = float("nan")
            acc = float((preds == y_t).mean())

            results[tier][layer_idx] = {
                "auc": round(float(auc), 4),
                "accuracy": round(acc, 4),
                "n_samples": int(n_t),
            }
            print(f"  tier={tier:7s} layer={layer_idx:2d}  "
                  f"AUC={auc:.4f}  Acc={acc:.4f}  n={n_t}")

    return results


# ── 5.  Figures ────────────────────────────────────────────────────────────

def plot_heatmap(results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tiers  = ["easy", "medium", "hard"]
    data   = np.array([
        [results.get(t, {}).get(l, {}).get("auc", float("nan"))
         for l in PROBE_LAYERS]
        for t in tiers
    ])
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(data, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"L{l}" for l in PROBE_LAYERS])
    ax.set_yticks(range(3)); ax.set_yticklabels(tiers)
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("Difficulty tier")
    ax.set_title("Pre-Thinking Probe AUC (OLMo-3-7B-Think, MMLU)")
    for i in range(3):
        for j in range(4):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="black")
    fig.colorbar(im, ax=ax, label="Macro OvR AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def plot_umap(acts: np.ndarray, records: list[dict], out_path: Path) -> None:
    try:
        import umap
        import matplotlib.pyplot as plt
    except ImportError:
        print("  umap-learn not found — skipping UMAP")
        return
    li = PROBE_LAYERS.index(31)
    X  = acts[:, li, :]
    correct = np.array([r["correct"] for r in records])
    emb = umap.UMAP(n_components=2, random_state=SEED, n_jobs=1).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    for val, lbl, col in [(1, "Correct", "#2196F3"), (0, "Incorrect", "#F44336")]:
        m = correct == val
        ax.scatter(emb[m, 0], emb[m, 1], c=col, label=lbl, s=5, alpha=0.5)
    ax.legend(markerscale=2)
    ax.set_title("UMAP — Layer-31 residual stream pre-thinking\nOLMo-3-7B-Think, MMLU")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
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

    # Load model
    print(f"\nLoading {MODEL_ID} …")
    model = LanguageModel(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        dispatch=True,
    )
    model.eval()

    # Load / build dataset
    ds_path = OUT_DIR / "labelled_dataset.json"
    if ds_path.exists():
        with open(ds_path) as f:
            dataset = json.load(f)
        print(f"Loaded cached dataset ({len(dataset)} questions)")
    else:
        raw = load_dataset("cais/mmlu", "all", split="test")
        dataset = label_difficulty(model, raw, N_PER_TIER)
        with open(ds_path, "w") as f:
            json.dump(dataset, f, indent=2)

    # Extract / load activations
    act_path = OUT_DIR / "activations_prethinkin.npz"
    rec_path = OUT_DIR / "answers.json"
    if act_path.exists() and rec_path.exists():
        acts = np.load(act_path)["activations"]
        with open(rec_path) as f:
            records = json.load(f)
        print(f"Loaded cached activations {acts.shape}")
    else:
        acts, records = extract_activations(model, dataset)
        np.savez_compressed(act_path, activations=acts)
        with open(rec_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved activations {acts.shape}")

    # Train probes
    print("\n=== Probe training ===")
    probe_results = run_probes(acts, records)
    with open(OUT_DIR / "probe_results.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    # Summary table
    print(f"\n{'Tier':>8} {'L8':>7} {'L16':>7} {'L24':>7} {'L31':>7}")
    for tier in ("easy", "medium", "hard", "all"):
        row = "  ".join(
            f"{probe_results.get(tier, {}).get(l, {}).get('auc', float('nan')):7.3f}"
            for l in PROBE_LAYERS
        )
        print(f"{tier:>8}  {row}")

    # Figures
    plot_heatmap(probe_results, fig_dir / "fig1_auc_heatmap.pdf")
    plot_umap(acts, records, fig_dir / "fig2_umap.pdf")
    print("\n✓  Experiment 1 complete.")


if __name__ == "__main__":
    main()
