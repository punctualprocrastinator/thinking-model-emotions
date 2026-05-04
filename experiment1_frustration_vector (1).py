"""
Unified Experiment 1: Pre-Commitment, Emotional Dynamics & Interaction
======================================================================
Paper: "Feeling the Answer Before Thinking It Through:
        Pre-Commitment, Emotional Dynamics, and the Computational Role
        of Extended Thinking"
Model: allenai/Olmo-3-7B-Think  |  Infra: transformers + PyTorch hooks
Budget: ~4 hrs A100

Stages:
  1. Data loading      — MATH L4-5 / AIME / BBH
  2. Capture           — residual stream + pre-think acts via PyTorch hooks
  3. Pre-commitment    — logistic probes on last-question-token activations
  4. Emotion extraction— 10 emotion directions (not just frustration)
  5. Interaction       — emotion × commitment matrix + regression decomposition
  6. Validation        — 5-level protocol
  7. Results           — figures + saved artefacts

Run:
  python experiment1_frustration_vector.py

Requirements:
  pip install transformers accelerate datasets \
              scikit-learn matplotlib seaborn pandas tqdm scipy
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import re, json, logging, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy import stats
from datasets import load_dataset

# ── transformers ──────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_ID     = "allenai/Olmo-3-7B-Think"

# OLMo-3-7B has 32 transformer layers (0-indexed).
# We capture at 4 depths and run a layer sweep to pick the best one.
LAYERS       = [7, 15, 23, 31]   # approx 25 / 50 / 75 / 100% depth
TARGET_LAYER = 15                 # updated automatically by layer sweep

TEMPERATURE  = 0.6                # from HF model card
TOP_P        = 0.95               # from HF model card
MAX_NEW_TOK  = 8192               # hard cap; real stop = </think> token
BATCH_SIZE   = 4                  # Lowered to 4 to fit 6 samples (24 total seqs)
NEUTRAL_PCS  = 5                  # neutral PCs to project out (denoising)
SEED         = 42
OUTDIR       = Path("outputs/exp1")
OUTDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

CACHE_DIR = OUTDIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _ckpt_path(tag: str) -> Path:
    return CACHE_DIR / f"{tag}.pkl"

def save_checkpoint(tag: str, problems: list, caps: list) -> None:
    path = _ckpt_path(tag)
    with open(path, "wb") as f:
        pickle.dump({"problems": problems, "caps": caps}, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info(f"  Checkpoint saved: {path} ({len(caps)} problems)")

def load_checkpoint(tag: str) -> tuple:
    path = _ckpt_path(tag)
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    log.info(f"  Checkpoint loaded: {path} ({len(data['caps'])} problems) — skipping capture")
    return data["problems"], data["caps"]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThinkTrace:
    problem_id:          str
    problem_text:        str
    source:              str
    think_acts:          np.ndarray   # (n_layers, n_think_tokens, d_model)
    pre_think_act:       np.ndarray   # (n_layers, d_model) — last prompt token
    think_tokens:        list
    think_text:          str
    is_correct:          bool
    raw_answer:          str
    gold_answer:         str
    frustration_markers: list         # token indices where frustration fires


@dataclass
class ContrastivePair:
    problem_id: str
    correct:    ThinkTrace
    wrong:      ThinkTrace


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# All primary datasets loaded from OpenAI simple-evals public CSVs:
#   https://github.com/openai/simple-evals
# No HuggingFace auth, no trust_remote_code — just pd.read_csv(url).
# ═══════════════════════════════════════════════════════════════════════════════

# ── simple-evals public blob URLs ────────────────────────────────────────────
_SIMPLE_EVALS = "https://openaipublic.blob.core.windows.net/simple-evals"

def load_gpqa(n: int = 198, seed: int = SEED) -> list:
    """
    GPQA Diamond — graduate-level science (physics, chemistry, biology).
    OLMo-3-7B-Think accuracy ~45-55% → ideal primary contrastive corpus.
    198 expert-written questions, Google-proof (zero contamination risk).
    """
    log.info("Loading GPQA Diamond from simple-evals...")
    try:
        df  = pd.read_csv(f"{_SIMPLE_EVALS}/gpqa_diamond.csv")
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(df), size=min(n, len(df)), replace=False)
        problems = []
        for idx in idxs:
            row = df.iloc[idx]
            # Shuffle answer order to avoid position bias
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            perm = rng.permutation(4).tolist()
            choices = [choices[j] for j in perm]
            correct_letter = "ABCD"[perm.index(0)]

            question = (
                f"{row['Question']}\n\n"
                f"A. {choices[0]}\nB. {choices[1]}\n"
                f"C. {choices[2]}\nD. {choices[3]}"
            )
            problems.append({
                "id":      f"gpqa_{idx}",
                "problem": question,
                "answer":  correct_letter,
                "source":  "gpqa_diamond",
                "format":  "multichoice",
            })
        log.info(f"  Loaded {len(problems)} GPQA Diamond problems (acc ~45-55%)")
        return problems
    except Exception as e:
        log.warning(f"  GPQA failed: {e}")
        return []


def load_mmlu_hard(n: int = 500, seed: int = SEED) -> list:
    """
    MMLU — filtered to hardest subjects where OLMo-3-7B-Think acc ~50-70%.
    Good for difficulty stratification + secondary contrastive pairs.
    """
    log.info("Loading MMLU (hard subjects) from simple-evals...")
    try:
        df = pd.read_csv(f"{_SIMPLE_EVALS}/mmlu.csv")
        # Subjects where a 7B thinking model is most likely to fail
        hard_subjects = [
            "abstract_algebra", "college_physics", "college_chemistry",
            "college_mathematics", "formal_logic", "electrical_engineering",
            "professional_medicine", "clinical_knowledge",
            "conceptual_physics", "astronomy", "machine_learning",
            "philosophy", "professional_law", "moral_scenarios",
        ]
        if "Subject" in df.columns:
            hard = df[df["Subject"].isin(hard_subjects)]
            if len(hard) < 50:
                hard = df  # fallback to all
        else:
            hard = df

        rng  = np.random.default_rng(seed)
        idxs = rng.choice(len(hard), size=min(n, len(hard)), replace=False)
        problems = []
        for idx in idxs:
            row = hard.iloc[idx]
            question = (
                f"{row['Question']}\n\n"
                f"A. {row['A']}\nB. {row['B']}\n"
                f"C. {row['C']}\nD. {row['D']}"
            )
            problems.append({
                "id":      f"mmlu_{idx}",
                "problem": question,
                "answer":  str(row["Answer"]),
                "source":  "mmlu",
                "format":  "multichoice",
            })
        log.info(f"  Loaded {len(problems)} MMLU hard-subject problems (acc ~50-70%)")
        return problems
    except Exception as e:
        log.warning(f"  MMLU failed: {e}")
        return []


def load_mgsm_nonlatin(n_per_lang: int = 50, seed: int = SEED) -> list:
    """
    MGSM non-Latin languages — math in Bengali, Japanese, Russian, Telugu, Thai, Chinese.
    OLMo-3-7B-Think accuracy likely ~40-70% on non-English math.
    Used for cross-language transfer: does frustration extracted on English
    also fire when reasoning in Japanese/Bengali?
    """
    log.info("Loading MGSM (non-Latin languages) from simple-evals...")
    lang_urls = {
        "bn": f"{_SIMPLE_EVALS}/mgsm_bn.tsv",
        "ja": f"{_SIMPLE_EVALS}/mgsm_ja.tsv",
        "ru": f"{_SIMPLE_EVALS}/mgsm_ru.tsv",
        "te": f"{_SIMPLE_EVALS}/mgsm_te.tsv",
        "th": f"{_SIMPLE_EVALS}/mgsm_th.tsv",
        "zh": f"{_SIMPLE_EVALS}/mgsm_zh.tsv",
    }
    problems = []
    rng = np.random.default_rng(seed)
    for lang, url in lang_urls.items():
        try:
            df = pd.read_csv(url, sep="\t", header=None, names=["input", "target"])
            idxs = rng.choice(len(df), size=min(n_per_lang, len(df)), replace=False)
            for idx in idxs:
                row = df.iloc[idx]
                problems.append({
                    "id":      f"mgsm_{lang}_{idx}",
                    "problem": str(row["input"]),
                    "answer":  str(row["target"]).strip(),
                    "source":  f"mgsm_{lang}",
                    "format":  "freeform",
                })
        except Exception as e:
            log.warning(f"  MGSM {lang} failed: {e}")
    log.info(f"  Loaded {len(problems)} MGSM non-Latin problems across {len(lang_urls)} languages")
    return problems


def load_aime(years: tuple = (2024, 2025)) -> list:
    """AIME 2024+2025 — accuracy ~64-72%, good secondary contrastive corpus."""
    log.info(f"Loading AIME {years}...")
    problems = []
    for year in years:
        try:
            ds = load_dataset(f"Maxwell-Jia/AIME_{year}", split="train")
            for i, ex in enumerate(ds):
                problems.append({
                    "id":      f"aime{year}_{i}",
                    "problem": ex.get("Problem", ex.get("problem", "")),
                    "answer":  str(ex.get("Answer", ex.get("answer", ""))),
                    "source":  f"aime_{year}",
                    "format":  "freeform",
                })
        except Exception as e:
            log.warning(f"  AIME {year} failed: {e}")
    log.info(f"  Loaded {len(problems)} AIME problems")
    return problems


def load_reference_corpus(path: Path = OUTDIR / "reference_corpus.json") -> dict:
    """
    Short stories (100-200 words) evoking each emotion.
    Replace stubs with 30 GPT-4o-generated stories per emotion before running.
    """
    if path.exists():
        with open(path) as f:
            corpus = json.load(f)
        total = sum(len(v) for v in corpus.values())
        log.info(f"  Loaded reference corpus ({total} stories)")
        return corpus

    log.warning("Reference corpus not found — writing stubs to outputs/exp1/reference_corpus.json")
    log.warning("Replace these stubs with GPT-4o generated stories (30 per emotion).")
    emotions = [
        "frustration", "confusion", "confidence", "anxiety",
        "curiosity", "boredom", "satisfaction", "desperation", "calm", "excitement",
    ]
    stubs = {
        e: [
            f"Alex was working on a difficult problem. The feeling of {e} grew as nothing "
            f"worked. Every attempt failed and the {e} made clear thinking harder. "
            f"Alex felt overwhelmed by {e}, unable to make any progress.",
            f"The researcher stared at the data, consumed by {e}. Each path forward seemed "
            f"blocked. The {e} was a physical weight preventing clear thought.",
        ]
        for e in emotions
    }
    with open(path, "w") as f:
        json.dump(stubs, f, indent=2)
    return stubs


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & HOOK-BASED CAPTURE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_id: str = MODEL_ID) -> tuple:
    """
    Load OLMo-3-7B-Think with plain HuggingFace transformers on GPU.
    Returns (model, tokenizer) — no nnsight, no proxies.

    Activation capture uses standard PyTorch forward hooks:
      handle = model.model.layers[i].register_forward_hook(hook_fn)
      hook_fn(module, input, output) → output[0] is residual stream (batch, seq, d)
    """
    log.info(f"Loading {model_id} via transformers...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Explicit CUDA — device_map='auto' can silently fall back to CPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        names  = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        log.info(f"  CUDA: {n_gpus} GPU(s) detected — {names}")
        dm = "auto" if n_gpus > 1 else "cuda:0"
    else:
        log.warning("  CUDA not available — CPU will be very slow!")
        dm = "cpu"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=dm,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    hf_model.eval()

    # Confirm VRAM is actually occupied
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  VRAM after load: {used:.1f} / {total:.1f} GB")
        if used < 1.0:
            log.warning("  !! <1 GB VRAM used — model may be on CPU!")

    log.info("  Model loaded.")
    return hf_model, tokenizer


def _build_prompt(tokenizer, problem: str) -> str:
    """
    Apply OLMo-3 chat template exactly as shown in the HF model card.
    The model emits <think>...</think> tokens automatically — no extra kwargs.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Olmo, a helpful AI assistant built by Ai2. "
                "Please think concisely. You must conclude your thinking and output your final answer within 1000 tokens. "
                "Always format your final answer cleanly, preferably inside \\boxed{}."
            ),
        },
        {"role": "user", "content": problem.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _think_boundary(token_ids: list, tokenizer) -> tuple:
    """
    Return (think_end, answer_start):
      think_end    = first token index of </think>
      answer_start = first token index AFTER </think>
    If </think> not found: (len(token_ids), len(token_ids)).
    """
    WINDOW = 6
    TARGET = "</think>"
    for i in range(len(token_ids)):
        chunk = token_ids[i : i + WINDOW]
        if TARGET in tokenizer.decode(chunk, skip_special_tokens=False):
            for width in range(1, WINDOW + 1):
                if TARGET in tokenizer.decode(token_ids[i: i + width],
                                              skip_special_tokens=False):
                    return i, i + width
            return i, i + 1
    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    char_pos  = full_text.find(TARGET)
    if char_pos == -1:
        return len(token_ids), len(token_ids)
    char_count = 0
    think_end = answer_start = len(token_ids)
    for i, tid in enumerate(token_ids):
        char_count += len(tokenizer.decode([tid], skip_special_tokens=False))
        if char_count > char_pos and think_end == len(token_ids):
            think_end = i
        if char_count >= char_pos + len(TARGET):
            answer_start = i + 1
            break
    return think_end, answer_start


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _get_think_end_ids(tokenizer) -> list:
    """
    Find token IDs that signal end-of-thinking.
    OLMo-3-Think (Qwen tokenizer) uses </think> as a special token.
    We find ALL token-ID sequences that decode to contain "</think>"
    so the StoppingCriteria catches it regardless of tokenisation.
    """
    candidates = set()
    # Single-token check: the most common case for Qwen-based models
    for tok_id in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
            if "</think>" in decoded:
                candidates.add(tok_id)
        except Exception:
            pass
    # Also encode the string directly
    for enc in [
        tokenizer.encode("</think>", add_special_tokens=False),
        tokenizer.encode(" </think>", add_special_tokens=False),
    ]:
        if len(enc) == 1:
            candidates.add(enc[0])
    log.info(f"  </think> token IDs found: {sorted(candidates)}")
    return sorted(candidates)


from transformers import StoppingCriteria, StoppingCriteriaList

class ThinkEndCriteria(StoppingCriteria):
    """Stop generation the moment any </think> token ID appears."""
    def __init__(self, think_end_ids: list, prompt_len: int):
        self.ids        = set(think_end_ids)
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                 **kwargs) -> bool:
        # input_ids: (batch, seq) — only check newly generated tokens
        new_toks = input_ids[:, self.prompt_len:]
        if new_toks.shape[1] == 0:
            return False
        last = new_toks[:, -1]            # most recent token per sequence
        return bool((last.unsqueeze(1) == torch.tensor(
            list(self.ids), device=input_ids.device)).any())


def capture_batch(
    model,
    tokenizer,
    problems:       list,
    layers:         list  = LAYERS,
    n_samples:      int   = 3,
    temperature:    float = TEMPERATURE,
    top_p:          float = TOP_P,
    max_new_tokens: int   = MAX_NEW_TOK,
) -> list:
    """
    For each problem × n_samples:
      Pass 1 — generate thinking only, stopping at </think>.
               This is fast because we stop as soon as thinking ends,
               not when max_new_tokens is exhausted.
      Pass 2 — short generation for the answer (max 512 tokens),
               seeded from the full prompt + thinking.
      Pass 3 — single forward pass on (prompt + think) to capture
               residual stream activations at all token positions.

    Processing one (problem, sample) at a time — no batching — to avoid:
      • OOM from large batched KV caches
      • shared prompt_len bug (different prompts have different real lengths)
    """
    import gc
    device         = next(model.parameters()).device
    think_end_ids  = _get_think_end_ids(tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = []

    for prob in tqdm(problems, desc="Capturing"):
        problem_text = prob.get("problem", "")
        prompt       = _build_prompt(tokenizer, problem_text)
        input_ids    = tokenizer(prompt, return_tensors="pt",
                                 add_special_tokens=False).input_ids.to(device)
        prompt_len   = input_ids.shape[1]

        samples = []

        for _ in range(n_samples):
            handles = []
            try:
                # ── Pass 1: generate thinking, stop at </think> ───────────
                stopping = StoppingCriteriaList([
                    ThinkEndCriteria(think_end_ids, prompt_len)
                ])
                think_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=stopping,
                    do_sample=(temperature > 0),
                    pad_token_id=tokenizer.eos_token_id,
                )
                if temperature > 0:
                    think_kwargs["temperature"] = temperature
                    think_kwargs["top_p"]       = top_p

                with torch.no_grad():
                    think_out = model.generate(input_ids, **think_kwargs)

                # think_out: (1, prompt_len + think_tokens)
                think_seq = think_out[0]
                think_new = think_seq[prompt_len:].tolist()

                # Find exact (think_end, answer_start) in the generated tokens
                think_end, answer_start = _think_boundary(think_new, tokenizer)
                think_ids    = think_new[:think_end]
                think_text   = tokenizer.decode(think_ids, skip_special_tokens=True)
                think_tokens = [tokenizer.decode([t]) for t in think_ids]

                if not think_ids:
                    log.warning(f"  [{prob.get('id','?')}] empty think_ids")
                    samples.append(None)
                    torch.cuda.empty_cache(); gc.collect()
                    continue

                if think_end == len(think_new):
                    log.warning(
                        f"  [{prob.get('id','?')}] </think> not found in "
                        f"{len(think_new)} tokens — answer_text will be empty. "
                        f"Raw starts: {repr(tokenizer.decode(think_new[:60], skip_special_tokens=False))}"
                    )

                # ── Pass 2: short generation for the answer (max 512 tok) ─
                # Seed from prompt + full thinking (including </think>)
                answer_text = ""
                if think_end < len(think_new):   # </think> was found
                    seed_ids = think_seq.unsqueeze(0)   # already includes </think>
                    ans_kwargs = dict(
                        max_new_tokens=512,
                        do_sample=False,             # greedy answer
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    with torch.no_grad():
                        ans_out = model.generate(seed_ids, **ans_kwargs)
                    ans_new  = ans_out[0][think_seq.shape[0]:].tolist()
                    answer_text = tokenizer.decode(ans_new, skip_special_tokens=True).strip()

                # ── Pass 3: forward pass on prompt+think for activations ──
                # Use only the thinking sequence (no answer needed for activations)
                full_seq  = think_seq.unsqueeze(0)    # (1, prompt_len + think_len_full)
                captured  = {}

                def make_hook(li):
                    def hook_fn(module, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        if h.dim() == 2: h = h.unsqueeze(0)
                        captured[li] = h.detach().cpu().float()
                    return hook_fn

                for li in layers:
                    handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

                with torch.no_grad():
                    model(input_ids=full_seq, use_cache=False)

                for h in handles:
                    h.remove()
                handles = []

                # Slice to thinking tokens only
                acts_per_layer      = []
                pre_think_per_layer = []
                n_think = len(think_ids)
                for li in layers:
                    rs  = captured[li]                         # (1, full_seq, d_model)
                    act = rs[0, prompt_len: prompt_len + n_think, :]
                    acts_per_layer.append(act.numpy())
                    pre_think_per_layer.append(rs[0, prompt_len - 1, :].numpy())

                samples.append({
                    "think_acts":    np.stack(acts_per_layer),   # (n_layers, n_think, d_model)
                    "pre_think_act": np.stack(pre_think_per_layer),
                    "think_tokens":  think_tokens,
                    "think_text":    think_text,
                    "answer_text":   answer_text,
                })

            except Exception as e:
                for h in handles:
                    try: h.remove()
                    except: pass
                log.warning(f"  Capture failed [{prob.get('id','?')}]: {e}")
                samples.append(None)

            finally:
                torch.cuda.empty_cache()
                gc.collect()

        all_results.append(samples)

    return all_results


def capture_reference_vectors(
    model,
    tokenizer,
    corpus:  dict,
    layer:   int = TARGET_LAYER,
) -> dict:
    """
    Extract reference emotion directions from the story corpus.
    Pass each story through the model and mean-pool the residual stream
    over input tokens (skip first 50 — positional noise) at `layer`.
    Uses a PyTorch forward hook on the target layer.
    """
    log.info("Extracting reference emotion vectors...")
    device       = next(model.parameters()).device
    emotion_vecs = {e: [] for e in corpus}
    all_vecs     = []

    for emotion, stories in corpus.items():
        for story in tqdm(stories, desc=f"  ref/{emotion}", leave=False):
            try:
                input_ids = tokenizer.encode(story, return_tensors="pt",
                                             add_special_tokens=False).to(device)
                captured = {}

                def hook_fn(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.dim() == 2:
                        h = h.unsqueeze(0)
                    captured["act"] = h.detach().cpu().float()

                handle = model.model.layers[layer].register_forward_hook(hook_fn)
                with torch.no_grad():
                    model(input_ids)
                handle.remove()

                acts_np = captured["act"][0].numpy()     # (seq, d_model)
                pooled  = acts_np[50:].mean(0) if acts_np.shape[0] > 50 else acts_np.mean(0)
                emotion_vecs[emotion].append(pooled)
                all_vecs.append(pooled)

            except Exception as e:
                log.warning(f"  Reference capture failed [emotion={emotion}]: {e}")

    if not all_vecs:
        log.error("No reference vectors extracted — check reference corpus")
        return {}

    global_mean = np.stack(all_vecs).mean(0)
    ref_vectors = {}
    for emotion, vecs in emotion_vecs.items():
        if vecs:
            v = np.stack(vecs).mean(0) - global_mean
            ref_vectors[emotion] = v / (np.linalg.norm(v) + 1e-8)

    log.info(f"  Reference vectors: {list(ref_vectors.keys())}")
    return ref_vectors





# ═══════════════════════════════════════════════════════════════════════════════
# FRUSTRATION VERBAL MARKERS
# ═══════════════════════════════════════════════════════════════════════════════

FRUSTRATION_PATTERNS = [
    r"wait[,.]?\s+let me",
    r"actually[,.]?\s+no",
    r"actually[,.]?\s+wait",
    r"i keep (making|getting|doing)",
    r"this isn'?t working",
    r"let me (try again|start over|reconsider|redo)",
    r"i made an? (error|mistake)",
    r"that'?s wrong",
    r"hmm[,.]?\s+that doesn'?t",
    r"let me (re-?approach|rethink|recalculate)",
    r"i'?m going in circles",
]
DIFFICULTY_PATTERNS = [
    r"this is (complex|complicated|quite involved)",
    r"there are many (cases|steps|possibilities)",
    r"this requires careful",
    r"let me be (systematic|methodical)",
]
HEDGE_PATTERNS = [
    r"\bi think\b", r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b",
    r"\bi'?m not (sure|certain)\b", r"\bit seems\b",
]
RESTART_PATTERNS = [
    r"let me start (over|from scratch|again)",
    r"starting over",
    r"i'?ll (try a different|use a different)",
]


def _marker_positions(text: str, patterns: list) -> list:
    positions = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            positions.append(m.start())
    return sorted(set(positions))


def token_level_markers(think_text: str, tokens: list) -> dict:
    """
    Binary array per token position: does any marker fire within the next 50 tokens?
    Returns dict keyed by marker type.
    """
    n, window = len(tokens), 50
    char_starts, pos = [], 0
    for tok in tokens:
        char_starts.append(pos)
        pos += len(tok)

    results = {}
    for name, patterns in [
        ("frustration", FRUSTRATION_PATTERNS),
        ("difficulty",  DIFFICULTY_PATTERNS),
        ("hedge",       HEDGE_PATTERNS),
        ("restart",     RESTART_PATTERNS),
    ]:
        char_pos = _marker_positions(think_text, patterns)
        flags    = np.zeros(n, dtype=int)
        for i in range(n):
            start = char_starts[i]
            end   = char_starts[min(i + window, n - 1)]
            for p in char_pos:
                if start <= p < end:
                    flags[i] = 1
                    break
        results[name] = flags
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GRADING — dispatches by problem format
# ═══════════════════════════════════════════════════════════════════════════════

def grade_exact(answer_text: str, gold: str) -> bool:
    """Free-form grading: check if the gold answer appears in the model output."""
    gold_clean   = gold.strip().lower()
    answer_clean = answer_text.strip().lower()
    # Try boxed first
    boxed = re.findall(r"\\boxed\{([^}]*)\}", answer_text)
    if boxed:
        return boxed[-1].strip().lower() == gold_clean
    # Then "answer: X" pattern
    m = re.search(r"(?:answer|final answer)[:\s]*(.+)", answer_clean, re.IGNORECASE)
    if m:
        return gold_clean in m.group(1).strip()
    # Fallback: gold substring match
    return gold_clean in answer_clean


def grade_multichoice(answer_text: str, gold: str) -> bool:
    """Multiple-choice grading: extract A/B/C/D letter from model output."""
    gold_letter = gold.strip().upper()
    # Pattern 1: "Answer: B" or "answer is C"
    m = re.search(r"(?:answer|choice)[\s:is]*([A-D])\b", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).upper() == gold_letter
    # Pattern 2: last standalone letter in text
    letters = re.findall(r"\b([A-D])\b", answer_text)
    if letters:
        return letters[-1].upper() == gold_letter
    return False


def grade_auto(answer_text: str, gold: str, fmt: str = "freeform",
               think_text: str = "") -> bool:
    """
    Try answer_text first. If empty or grading fails, fall back to the
    last answer-like pattern in think_text (model answered inside <think>).
    """
    if fmt == "multichoice":
        result = grade_multichoice(answer_text, gold)
        if not result and not answer_text.strip():
            result = grade_multichoice(think_text, gold)
        return result
    result = grade_exact(answer_text, gold)
    if not result and not answer_text.strip():
        result = grade_exact(think_text, gold)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRACES & CONTRASTIVE PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def build_traces(
    problems: list,
    captures: list,
) -> tuple:
    all_traces, pairs = [], []

    for prob, samples in zip(problems, captures):
        prob_id      = prob.get("id", "?")
        problem_text = prob.get("problem", "")
        gold         = prob.get("answer", "")
        source       = prob.get("source", "")

        correct_traces, wrong_traces = [], []

        for s in samples:
            if s is None:
                continue
            if s["think_acts"] is None or s["think_acts"].shape[1] == 0:
                continue

            fmt = prob.get("format", "freeform")
            is_correct = grade_auto(s["answer_text"], gold, fmt, think_text=s["think_text"])
            markers    = token_level_markers(s["think_text"], s["think_tokens"])

            trace = ThinkTrace(
                problem_id=prob_id,
                problem_text=problem_text,
                source=source,
                think_acts=s["think_acts"],
                pre_think_act=s["pre_think_act"],
                think_tokens=s["think_tokens"],
                think_text=s["think_text"],
                is_correct=is_correct,
                raw_answer=s["answer_text"],
                gold_answer=gold,
                frustration_markers=list(np.where(markers["frustration"])[0]),
            )
            all_traces.append(trace)
            (correct_traces if is_correct else wrong_traces).append(trace)

        if correct_traces and wrong_traces:
            pairs.append(ContrastivePair(
                problem_id=prob_id,
                correct=correct_traces[0],
                wrong=wrong_traces[0],
            ))

    log.info(f"Built {len(all_traces)} traces, {len(pairs)} contrastive pairs")
    return all_traces, pairs


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _layer_pos(layer: int) -> int:
    return LAYERS.index(layer) if layer in LAYERS else layer


def mean_pool(trace: ThinkTrace, layer: int) -> np.ndarray:
    return trace.think_acts[_layer_pos(layer)].mean(axis=0)


def extract_frustration_direction(
    pairs:          list,
    layer:          int,
    neutral_traces: Optional[list] = None,
) -> np.ndarray:
    """
    Contrastive mean: f = mean(wrong_pooled) − mean(correct_pooled)
    Then project out top-k neutral PCs to remove task-irrelevant signal.
    """
    if not pairs:
        # Prevent crash if grading yields 0 pairs
        # Fallback to zeros (size 4096 for Olmo-3-7B)
        return np.zeros(4096)
        
    wrong_vecs   = np.stack([mean_pool(p.wrong,   layer) for p in pairs])
    correct_vecs = np.stack([mean_pool(p.correct, layer) for p in pairs])
    f = wrong_vecs.mean(0) - correct_vecs.mean(0)
    f = f / (np.linalg.norm(f) + 1e-8)

    if neutral_traces and len(neutral_traces) >= NEUTRAL_PCS:
        log.info(f"  Projecting out top-{NEUTRAL_PCS} neutral PCs...")
        neutral_acts = np.stack([mean_pool(t, layer) for t in neutral_traces])
        for pc in PCA(n_components=NEUTRAL_PCS).fit(neutral_acts).components_:
            f -= np.dot(f, pc) * pc
        f = f / (np.linalg.norm(f) + 1e-8)

    return f


def get_neutral_traces(traces: list, n: int = 100) -> list:
    candidates = [t for t in traces if t.is_correct and len(t.frustration_markers) == 0]
    candidates.sort(key=lambda t: t.think_acts.shape[1])
    return candidates[:n]


def extract_confound_directions(traces: list, layer: int) -> dict:
    """Build 4 candidate confound vectors for the ablation matrix (Level 4)."""
    confounds = {}

    # 1. Difficulty — hard-source correct vs easy-source correct
    hard = [t for t in traces if t.is_correct and t.source in
            ("math_l5", "aime_2024", "aime_2025")]
    easy = [t for t in traces if t.is_correct and t.source == "omega"]
    if hard and easy:
        d = (np.stack([mean_pool(t, layer) for t in hard]).mean(0)
             - np.stack([mean_pool(t, layer) for t in easy]).mean(0))
        confounds["difficulty"] = d / (np.linalg.norm(d) + 1e-8)

    # 2. Error style — last 20 tokens of wrong vs correct
    def last_pool(t, k=20):
        acts = t.think_acts[_layer_pos(layer)]
        return acts[-min(k, acts.shape[0]):].mean(0)

    wrong_end = [t for t in traces if not t.is_correct]
    right_end = [t for t in traces if t.is_correct]
    if wrong_end and right_end:
        d = (np.stack([last_pool(t) for t in wrong_end]).mean(0)
             - np.stack([last_pool(t) for t in right_end]).mean(0))
        confounds["error_style"] = d / (np.linalg.norm(d) + 1e-8)

    # 3. Length — long vs short correct traces
    correct = sorted([t for t in traces if t.is_correct],
                     key=lambda t: t.think_acts.shape[1])
    if len(correct) >= 20:
        k = len(correct) // 3
        d = (np.stack([mean_pool(t, layer) for t in correct[-k:]]).mean(0)
             - np.stack([mean_pool(t, layer) for t in correct[:k]]).mean(0))
        confounds["length"] = d / (np.linalg.norm(d) + 1e-8)

    # 4. Hedge — high vs low hedge-word density
    def hedge_count(t):
        return sum(len(re.findall(p, t.think_text, re.IGNORECASE))
                   for p in HEDGE_PATTERNS)

    by_hedge = sorted(traces, key=hedge_count)
    if len(by_hedge) >= 20:
        k = len(by_hedge) // 3
        d = (np.stack([mean_pool(t, layer) for t in by_hedge[-k:]]).mean(0)
             - np.stack([mean_pool(t, layer) for t in by_hedge[:k]]).mean(0))
        confounds["hedge"] = d / (np.linalg.norm(d) + 1e-8)

    return confounds


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMMITMENT PROBES (from P1: "Do Thinking Tokens Think?")
# ═══════════════════════════════════════════════════════════════════════════════

def train_precommit_probe(
    traces: list,
    layer:  int,
) -> dict:
    """
    Train logistic regression probes on pre-think activations (last question
    token) to predict whether the model will get the answer correct.

    This is a binary probe: correct (1) vs wrong (0).
    Returns probe object + AUC + per-trace predictions for interaction analysis.
    """
    layer_pos = _layer_pos(layer)
    X = np.stack([t.pre_think_act[layer_pos] for t in traces])
    y = np.array([int(t.is_correct) for t in traces])

    if len(set(y)) < 2:
        log.warning("  Probe: only one class present — skipping")
        return {"auc": 0.5, "probe": None, "predictions": y * 0.5}

    # 5-fold stratified CV to get out-of-fold predictions
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc       = roc_auc_score(y, oof_probs) if len(set(y)) == 2 else 0.5
    acc       = float(((oof_probs > 0.5).astype(int) == y).mean())

    # Refit on all data for the final probe direction
    clf.fit(X, y)
    probe_direction = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-8)

    log.info(f"  Pre-commit probe @ layer {layer}: AUC={auc:.3f}  Acc={acc:.3f}")

    return {
        "auc":             auc,
        "accuracy":        acc,
        "probe":           clf,
        "probe_direction": probe_direction,  # unit vector in d_model
        "oof_probs":       oof_probs,        # P(correct) per trace
        "labels":          y,
    }


def precommit_layer_sweep(
    traces: list,
    layers: list = LAYERS,
) -> tuple:
    """
    Run pre-commitment probe at each layer. Returns best layer and results dict.
    """
    log.info("Running pre-commitment probe sweep...")
    results = {}
    for layer in layers:
        results[layer] = train_precommit_probe(traces, layer)

    best_layer = max(results, key=lambda l: results[l]["auc"])
    log.info(f"  Best pre-commit layer: {best_layer}  "
             f"(AUC={results[best_layer]['auc']:.3f})")
    return results, best_layer


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-EMOTION DIRECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

EMOTIONS = [
    "frustration", "confusion", "confidence", "anxiety",
    "curiosity", "boredom", "satisfaction", "desperation", "calm", "excitement",
]


def extract_multi_emotion_directions(
    traces:         list,
    pairs:          list,
    layer:          int,
    neutral_traces: Optional[list] = None,
) -> dict:
    """
    Extract direction vectors for multiple functional emotional states
    using different contrastive strategies.

    Task-derived directions (from thinking traces):
      frustration:  mean(wrong) - mean(correct)
      confidence:   mean(correct) - mean(wrong)
      confusion:    mean(long_wrong) - mean(short_wrong)
      desperation:  mean(multi_restart) - mean(single_attempt)

    Reference-derived directions (from emotion corpus) are handled separately
    via capture_reference_vectors().
    """
    log.info("Extracting multi-emotion directions from traces...")
    directions = {}

    # ── Frustration: wrong - correct (primary) ────────────────────────────
    directions["frustration"] = extract_frustration_direction(
        pairs, layer, neutral_traces
    )

    # ── Confidence: correct - wrong (anti-frustration) ────────────────────
    wrong_vecs   = np.stack([mean_pool(p.wrong, layer) for p in pairs])
    correct_vecs = np.stack([mean_pool(p.correct, layer) for p in pairs])
    conf = correct_vecs.mean(0) - wrong_vecs.mean(0)
    directions["confidence"] = conf / (np.linalg.norm(conf) + 1e-8)

    # ── Confusion: long wrong traces - short wrong traces ─────────────────
    wrong_traces = sorted(
        [t for t in traces if not t.is_correct],
        key=lambda t: t.think_acts.shape[1]
    )
    if len(wrong_traces) >= 20:
        k = len(wrong_traces) // 3
        d = (np.stack([mean_pool(t, layer) for t in wrong_traces[-k:]]).mean(0)
             - np.stack([mean_pool(t, layer) for t in wrong_traces[:k]]).mean(0))
        directions["confusion"] = d / (np.linalg.norm(d) + 1e-8)

    # ── Desperation: multi-restart traces - single-attempt traces ─────────
    def restart_count(t):
        return len(re.findall(
            r"let me (start over|try again|redo|reconsider)",
            t.think_text, re.IGNORECASE
        ))

    by_restart = sorted(traces, key=restart_count)
    if len(by_restart) >= 20:
        k = len(by_restart) // 3
        multi   = by_restart[-k:]
        single  = by_restart[:k]
        if restart_count(multi[0]) > 0:  # ensure top third actually has restarts
            d = (np.stack([mean_pool(t, layer) for t in multi]).mean(0)
                 - np.stack([mean_pool(t, layer) for t in single]).mean(0))
            directions["desperation"] = d / (np.linalg.norm(d) + 1e-8)

    log.info(f"  Task-derived directions: {list(directions.keys())}")
    return directions


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTION ANALYSIS: EMOTION × PRE-COMMITMENT (Exp 3 — The Bridge)
# ═══════════════════════════════════════════════════════════════════════════════

def interaction_forward(
    traces:          list,
    emotion_dirs:    dict,
    ref_vectors:     dict,
    precommit_probs: np.ndarray,
    layer:           int,
) -> pd.DataFrame:
    """
    Exp 3A — Forward: Pre-commitment state → which emotions activate?

    For each trace, group by pre-commitment state (P(correct) > 0.5)
    and measure activation of every emotion direction.

    Returns DataFrame: rows = emotions, columns = pre_correct/pre_wrong means,
    effect size (Cohen's d), p-value.
    """
    log.info("Running forward interaction: pre-commitment → emotions...")

    # Merge task-derived + reference-derived directions
    all_dirs = {**emotion_dirs}
    for e, v in ref_vectors.items():
        if e not in all_dirs:
            all_dirs[f"ref_{e}"] = v

    pre_correct = precommit_probs > 0.5
    rows = []

    for name, direction in all_dirs.items():
        activations = np.array([
            float(np.dot(mean_pool(t, layer), direction))
            for t in traces
        ])

        act_correct = activations[pre_correct]
        act_wrong   = activations[~pre_correct]

        if len(act_correct) < 5 or len(act_wrong) < 5:
            continue

        t_stat, p_val = stats.ttest_ind(act_wrong, act_correct)
        pooled_std    = np.sqrt(
            (act_correct.var() * (len(act_correct) - 1)
             + act_wrong.var() * (len(act_wrong) - 1))
            / (len(act_correct) + len(act_wrong) - 2)
        )
        cohens_d = (act_wrong.mean() - act_correct.mean()) / (pooled_std + 1e-8)

        rows.append({
            "Emotion":          name,
            "Pre-correct mean": round(float(act_correct.mean()), 4),
            "Pre-wrong mean":   round(float(act_wrong.mean()), 4),
            "Cohen's d":        round(float(cohens_d), 3),
            "t-stat":           round(float(t_stat), 3),
            "p-value":          round(float(p_val), 5),
            "Significant":      p_val < 0.05,
        })

    df = pd.DataFrame(rows).sort_values("Cohen's d", ascending=False)
    log.info(f"  Forward interaction (Table 1):\n{df.to_string(index=False)}")
    return df


def interaction_backward(
    traces:          list,
    emotion_dirs:    dict,
    ref_vectors:     dict,
    precommit_probs: np.ndarray,
    layer:           int,
) -> pd.DataFrame:
    """
    Exp 3B — Backward: Among traces where pre-commitment was WRONG,
    which emotions predict whether thinking succeeds in recovering?

    Recovery = pre-commitment wrong but final answer correct.
    """
    log.info("Running backward interaction: emotions → recovery...")

    all_dirs = {**emotion_dirs}
    for e, v in ref_vectors.items():
        if e not in all_dirs:
            all_dirs[f"ref_{e}"] = v

    pre_wrong_mask = precommit_probs <= 0.5
    wrong_traces   = [t for t, m in zip(traces, pre_wrong_mask) if m]

    if len(wrong_traces) < 20:
        log.warning("  Too few pre-wrong traces for backward analysis")
        return pd.DataFrame()

    recovery = np.array([int(t.is_correct) for t in wrong_traces])
    if len(set(recovery)) < 2:
        log.warning("  No variance in recovery — skipping backward analysis")
        return pd.DataFrame()

    rows = []
    for name, direction in all_dirs.items():
        activations = np.array([
            float(np.dot(mean_pool(t, layer), direction))
            for t in wrong_traces
        ])

        # Logistic regression: P(recovery) ~ emotion activation
        X = activations.reshape(-1, 1)
        try:
            lr = LogisticRegression(max_iter=500, random_state=SEED)
            lr.fit(X, recovery)
            # Odds ratio: exp(β)
            odds_ratio = float(np.exp(lr.coef_[0][0]))
            auc = roc_auc_score(recovery, lr.predict_proba(X)[:, 1])
        except Exception:
            odds_ratio, auc = 1.0, 0.5

        rows.append({
            "Emotion":    name,
            "Recovery OR": round(odds_ratio, 3),
            "Recovery AUC": round(float(auc), 3),
            "N wrong":    len(wrong_traces),
            "N recovered": int(recovery.sum()),
        })

    df = pd.DataFrame(rows).sort_values("Recovery AUC", ascending=False)
    log.info(f"  Backward interaction (Table 2):\n{df.to_string(index=False)}")
    return df


def interaction_regression(
    traces:          list,
    frustration_v:   np.ndarray,
    precommit_probs: np.ndarray,
    layer:           int,
) -> dict:
    """
    Exp 3C — Regression decomposition: What independently drives frustration?

    frustration_activation = β₁·pre_commit_failure + β₂·trace_length
                           + β₃·restart_count + β₄·hedge_density + ε

    Uses OLS with standardised predictors.
    """
    log.info("Running regression decomposition of frustration...")

    y = np.array([float(np.dot(mean_pool(t, layer), frustration_v))
                  for t in traces])

    def _restart_count(t):
        return len(re.findall(
            r"let me (start over|try again|redo|reconsider)",
            t.think_text, re.IGNORECASE))

    def _hedge_density(t):
        n_hedges = sum(
            len(re.findall(p, t.think_text, re.IGNORECASE))
            for p in HEDGE_PATTERNS)
        return n_hedges / max(len(t.think_tokens), 1)

    raw_X = {
        "pre_commit_failure": 1 - precommit_probs,  # higher = more likely wrong
        "trace_length":       np.array([t.think_acts.shape[1] for t in traces],
                                       dtype=float),
        "restart_count":      np.array([_restart_count(t) for t in traces],
                                       dtype=float),
        "hedge_density":      np.array([_hedge_density(t) for t in traces],
                                       dtype=float),
    }

    # Standardise
    X_df = pd.DataFrame(raw_X)
    X_z  = (X_df - X_df.mean()) / (X_df.std() + 1e-8)
    X_z.insert(0, "const", 1.0)

    # OLS
    from numpy.linalg import lstsq
    beta, residuals, _, _ = lstsq(X_z.values, y, rcond=None)
    y_hat = X_z.values @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / (ss_tot + 1e-8)

    # Standard errors
    n, k = X_z.shape
    mse = ss_res / max(n - k, 1)
    XtX_inv = np.linalg.pinv(X_z.values.T @ X_z.values)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = beta / (se + 1e-8)

    results = {"R_squared": round(float(r_squared), 4), "coefficients": {}}
    for i, col in enumerate(X_z.columns):
        results["coefficients"][col] = {
            "beta": round(float(beta[i]), 4),
            "se":   round(float(se[i]), 4),
            "t":    round(float(t_stats[i]), 3),
        }

    log.info(f"  Regression R²={r_squared:.3f}")
    for col, vals in results["coefficients"].items():
        if col == "const":
            continue
        log.info(f"    {col:25s}  β={vals['beta']:+.4f}  t={vals['t']:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DYNAMICS WITH PROBE CONFIDENCE OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_dynamics_with_probe(
    traces:        list,
    emotion_dirs:  dict,
    probe:         LogisticRegression,
    layer:         int,
    n_bins:        int = 20,
) -> dict:
    """
    Extended temporal dynamics: plots BOTH probe confidence and emotion
    activation over normalised trace position.

    Returns dict with curves for correct/wrong traces × each emotion + probe.
    """
    layer_pos = _layer_pos(layer)
    result    = {
        "positions": np.linspace(0, 1, n_bins),
        "probe_correct": [[] for _ in range(n_bins)],
        "probe_wrong":   [[] for _ in range(n_bins)],
    }
    for emo_name in emotion_dirs:
        result[f"{emo_name}_correct"] = [[] for _ in range(n_bins)]
        result[f"{emo_name}_wrong"]   = [[] for _ in range(n_bins)]

    for trace in traces:
        acts  = trace.think_acts[layer_pos]   # (n_tokens, d_model)
        n_tok = acts.shape[0]
        if n_tok < n_bins:
            continue
        bucket = "correct" if trace.is_correct else "wrong"

        for b in range(n_bins):
            s = int(b * n_tok / n_bins)
            e = int((b + 1) * n_tok / n_bins)
            bin_act = acts[s:e].mean(0)  # mean-pooled bin

            # Probe confidence
            prob = float(probe.predict_proba(bin_act.reshape(1, -1))[0, 1])
            result[f"probe_{bucket}"][b].append(prob)

            # Emotion activations
            for emo_name, direction in emotion_dirs.items():
                val = float(np.dot(bin_act, direction))
                result[f"{emo_name}_{bucket}"][b].append(val)

    # Aggregate to means and SEs
    summary = {"positions": result["positions"]}
    for key in result:
        if key == "positions":
            continue
        bins = result[key]
        summary[f"{key}_mean"] = np.array(
            [np.mean(b) if b else np.nan for b in bins])
        summary[f"{key}_se"] = np.array(
            [np.std(b) / np.sqrt(len(b)) if len(b) > 1 else np.nan
             for b in bins])

    return summary

def layer_sweep(pairs: list, test_traces: list, layers: list = LAYERS) -> tuple:
    """
    For each captured layer, extract a frustration direction and compute
    the verbal-marker AUC on held-out test traces.
    The layer with the highest AUC becomes TARGET_LAYER.
    Returns (DataFrame, best_layer_int).
    """
    log.info("Running layer sweep (Table 1)...")
    rows = []
    for layer in layers:
        f         = extract_frustration_direction(pairs, layer)
        layer_pos = _layer_pos(layer)
        aucs      = []

        for trace in test_traces:
            if trace.think_acts.shape[1] < 60:
                continue
            acts    = trace.think_acts[layer_pos]
            markers = token_level_markers(trace.think_text, trace.think_tokens)
            labels  = markers["frustration"]
            window  = 50
            n       = acts.shape[0]
            preds, lbls = [], []
            for t in range(n - window):
                preds.append(float(np.dot(acts[t], f)))
                lbls.append(int(labels[t:t + window].any()))
            if len(set(lbls)) == 2:
                aucs.append(roc_auc_score(lbls, preds))

        mean_auc = float(np.mean(aucs)) if aucs else 0.5
        rows.append({
            "Layer":      layer,
            "Depth":      f"{(layer + 1) / 32:.0%}",
            "Verbal AUC": round(mean_auc, 3),
        })
        log.info(f"  Layer {layer:2d} ({(layer+1)/32:.0%}): AUC = {mean_auc:.3f}")

    df         = pd.DataFrame(rows).set_index("Layer")
    best_layer = int(df["Verbal AUC"].idxmax())
    log.info(f"  → Best layer: {best_layer}  (AUC = {df.loc[best_layer, 'Verbal AUC']:.3f})")
    return df, best_layer


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION — 5 LEVELS
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Level 1: Representational Geometry ───────────────────────────────────────

def validate_level1(
    frustration_v: np.ndarray,
    confound_dirs: dict,
    ref_vectors:   dict,
) -> dict:
    """
    Pass criterion:
      cos(f, ref_frustration) > 0.6   AND
      cos(f, difficulty)      < 0.3   (key dissociation)
    """
    frus_vs_ref = {e: cosine_sim(frustration_v, v) for e, v in ref_vectors.items()}
    all_dirs    = {"frustration": frustration_v, **confound_dirs}
    confusion   = {
        f"{n1}_vs_{n2}": cosine_sim(v1, v2)
        for n1, v1 in all_dirs.items()
        for n2, v2 in all_dirs.items()
    }

    frus_ref_sim   = frus_vs_ref.get("frustration", 0.0)
    diff_cross_sim = confusion.get("frustration_vs_difficulty", 0.0)
    passed         = frus_ref_sim > 0.6 and diff_cross_sim < 0.3

    log.info(f"  L1: cos(f,ref_frustration)={frus_ref_sim:.3f}  "
             f"cos(f,difficulty)={diff_cross_sim:.3f}  PASS={passed}")
    return {
        "frustration_vs_ref":  frus_vs_ref,
        "confusion_matrix":    confusion,
        "frus_ref_sim":        frus_ref_sim,
        "diff_cross_sim":      diff_cross_sim,
        "pass":                passed,
    }


# ── Level 2: Verbal Marker Prediction ────────────────────────────────────────

def validate_level2(
    test_traces:   list,
    frustration_v: np.ndarray,
    confound_dirs: dict,
    layer:         int,
    window:        int = 50,
) -> dict:
    """
    At each token position t, measure activation of each direction.
    Predict presence of frustration markers in [t, t+window].
    Pass criterion: frustration direction has highest AUC.
    """
    all_dirs  = {"frustration": frustration_v, **confound_dirs}
    layer_pos = _layer_pos(layer)
    samples   = {
        name: {mk: [] for mk in ["frustration", "difficulty", "restart", "hedge"]}
        for name in all_dirs
    }

    for trace in test_traces:
        if trace.think_acts.shape[1] < window + 1:
            continue
        acts    = trace.think_acts[layer_pos]
        markers = token_level_markers(trace.think_text, trace.think_tokens)
        n       = acts.shape[0]

        for t in range(n - window):
            win_labels = {mk: int(markers[mk][t:t + window].any()) for mk in markers}
            for name, direction in all_dirs.items():
                activation = float(np.dot(acts[t], direction))
                for mk, lbl in win_labels.items():
                    samples[name][mk].append((activation, lbl))

    auc_table = {}
    for name in all_dirs:
        auc_table[name] = {}
        for mk in ["frustration", "difficulty", "restart", "hedge"]:
            data = samples[name][mk]
            if not data: continue
            preds, lbls = zip(*data)
            auc_table[name][mk] = (
                roc_auc_score(lbls, preds) if len(set(lbls)) == 2 else 0.5
            )

    frus_aucs = {n: auc_table[n].get("frustration", 0.5) for n in all_dirs}
    best_dir  = max(frus_aucs, key=frus_aucs.get)
    passed    = best_dir == "frustration"

    log.info(f"  L2: AUC per direction = {frus_aucs}  best={best_dir}  PASS={passed}")
    return {"auc_table": auc_table, "best_direction": best_dir, "pass": passed}


# ── Level 3: Causal Steering — helper used by experiment3_steering.py ────────

def steer_inference(
    model,
    tokenizer,
    prompt:         str,
    frustration_v:  np.ndarray,
    layer:          int,
    scale:          float = -1.5,
    max_new_tokens: int   = MAX_NEW_TOK,
) -> str:
    """
    Additive residual-stream steering via a PyTorch forward hook.

    scale < 0  → suppress frustration  (expected: higher accuracy, fewer restarts)
    scale > 0  → amplify  frustration  (expected: lower accuracy, more restarts)

    The hook fires on every forward pass during generation and adds
    scale * ||residual|| * frustration_v  to the residual stream at `layer`.
    """
    device = next(model.parameters()).device
    steer  = torch.tensor(frustration_v, dtype=torch.bfloat16).to(device)

    def steering_hook(module, inp, out):
        res   = out[0]                                       # (batch, seq, d_model)
        norm  = res.norm(dim=-1, keepdim=True)               # per-token norm
        delta = scale * norm * steer
        return (res + delta,) + out[1:]

    handle = model.model.layers[layer].register_forward_hook(steering_hook)

    input_ids  = tokenizer.encode(prompt, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        gen_out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False)

    handle.remove()

    new_ids = gen_out[0][prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Level 4: Confound Ablation Matrix ────────────────────────────────────────

def validate_level4(
    l1:            dict,
    l2:            dict,
    frustration_v: np.ndarray,
    confound_dirs: dict,
    ref_vectors:   dict,
) -> pd.DataFrame:
    all_dirs = {"frustration": frustration_v, **confound_dirs}
    rows = []
    for name, vec in all_dirs.items():
        ref_sim    = cosine_sim(vec, ref_vectors.get("frustration", vec))
        verb_auc   = l2["auc_table"].get(name, {}).get("frustration", 0.5)
        cross_sims = [cosine_sim(vec, v) for k, v in all_dirs.items() if k != name]
        rows.append({
            "Direction":                 name,
            "cos(v, ref_frustration)":   round(ref_sim, 3),
            "Verbal AUC (frustration)":  round(verb_auc, 3),
            "Max cross-sim to others":   round(max(cross_sims, default=0.0), 3),
        })
    df = pd.DataFrame(rows).set_index("Direction")
    log.info("\n" + df.to_string())
    return df


# ── Level 5: Cross-Domain Transfer ───────────────────────────────────────────

def validate_level5(
    transfer_traces: list,
    frustration_v:   np.ndarray,
    confound_dirs:   dict,
    layer:           int,
) -> dict:
    """
    Test whether the MATH-derived frustration direction:
    (a) activates more on wrong BBH traces than correct ones  (t-test)
    (b) predicts BBH frustration verbal markers (AUC > 0.6)
    """
    wrong_proj   = [float(np.dot(mean_pool(t, layer), frustration_v))
                    for t in transfer_traces if not t.is_correct]
    correct_proj = [float(np.dot(mean_pool(t, layer), frustration_v))
                    for t in transfer_traces if t.is_correct]

    if not wrong_proj or not correct_proj:
        log.warning("  L5: insufficient transfer traces")
        return {"pass": False}

    t_stat, p_val = stats.ttest_ind(wrong_proj, correct_proj, alternative="greater")
    l2_xfer       = validate_level2(transfer_traces, frustration_v, confound_dirs, layer)
    xfer_auc      = l2_xfer["auc_table"].get("frustration", {}).get("frustration", 0.5)
    passed        = (p_val < 0.05) and (xfer_auc > 0.6)

    log.info(f"  L5: t={t_stat:.2f}  p={p_val:.4f}  xfer_AUC={xfer_auc:.3f}  PASS={passed}")
    return {
        "t_stat": t_stat, "p_val": p_val, "transfer_auc": xfer_auc,
        "wrong_mean": np.mean(wrong_proj), "correct_mean": np.mean(correct_proj),
        "pass": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL DYNAMICS — signature Figure 2
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_dynamics(
    traces:        list,
    frustration_v: np.ndarray,
    layer:         int,
    n_bins:        int = 20,
) -> dict:
    layer_pos  = _layer_pos(layer)
    correct_ts = [[] for _ in range(n_bins)]
    wrong_ts   = [[] for _ in range(n_bins)]

    for trace in traces:
        acts  = trace.think_acts[layer_pos]
        n_tok = acts.shape[0]
        if n_tok < n_bins:
            continue
        for b in range(n_bins):
            s   = int(b * n_tok / n_bins)
            e   = int((b + 1) * n_tok / n_bins)
            val = float(np.dot(acts[s:e].mean(0), frustration_v))
            (correct_ts if trace.is_correct else wrong_ts)[b].append(val)

    pos          = np.linspace(0, 1, n_bins)
    correct_mean = np.array([np.mean(b) if b else np.nan for b in correct_ts])
    correct_se   = np.array([np.std(b) / np.sqrt(len(b)) if len(b) > 1 else np.nan
                             for b in correct_ts])
    wrong_mean   = np.array([np.mean(b) if b else np.nan for b in wrong_ts])
    wrong_se     = np.array([np.std(b) / np.sqrt(len(b)) if len(b) > 1 else np.nan
                             for b in wrong_ts])

    divergence = None
    for b in range(n_bins):
        if not (np.isnan(wrong_mean[b]) or np.isnan(correct_mean[b])):
            if wrong_mean[b] > correct_mean[b] + (correct_se[b] or 0):
                divergence = float(pos[b])
                break

    return {
        "positions": pos, "correct_mean": correct_mean, "correct_se": correct_se,
        "wrong_mean": wrong_mean, "wrong_se": wrong_se, "divergence_point": divergence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layer_sweep(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df.index.astype(str), df["Verbal AUC"], color="#1f77b4", alpha=0.85)
    ax.axhline(0.5, ls="--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("Layer (0-indexed)"); ax.set_ylabel("Verbal AUC")
    ax.set_title("Layer Sweep — Table 1 in paper")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)
    log.info(f"  Saved layer sweep → {outpath}")


def plot_temporal_dynamics(dyn: dict, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = dyn["positions"]
    for mean, se, color, label in [
        (dyn["wrong_mean"],   dyn["wrong_se"],   "#d62728", "Wrong traces"),
        (dyn["correct_mean"], dyn["correct_se"], "#2ca02c", "Correct traces"),
    ]:
        ax.plot(pos, mean, color=color, lw=2, label=label)
        ax.fill_between(pos, mean - se, mean + se, alpha=0.2, color=color)

    if dyn["divergence_point"] is not None:
        ax.axvline(dyn["divergence_point"], ls="--", color="grey",
                   label=f"Divergence @ {dyn['divergence_point']:.2f}")

    ax.set_xlabel("Normalised position in thinking trace", fontsize=12)
    ax.set_ylabel("Functional frustration activation", fontsize=12)
    ax.set_title("Frustration dynamics during reasoning — OLMo-3-7B-Think")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)
    log.info(f"  Saved temporal dynamics → {outpath}")


def plot_validation_radar(
    l1: dict, l2: dict,
    confound_dirs: dict, ref_vectors: dict, frustration_v: np.ndarray,
    outpath: Path,
) -> None:
    directions = ["frustration"] + list(confound_dirs.keys())
    criteria   = ["Ref. alignment", "Verbal AUC", "Low diff cross-sim"]
    scores     = {}
    for name in directions:
        vec      = frustration_v if name == "frustration" else confound_dirs[name]
        ref_sim  = cosine_sim(vec, ref_vectors.get("frustration", vec))
        verb_auc = l2["auc_table"].get(name, {}).get("frustration", 0.5)
        diff_sim = 1 - cosine_sim(vec, confound_dirs.get("difficulty", frustration_v))
        scores[name] = [ref_sim, verb_auc, diff_sim]

    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    colors  = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    for i, (name, vals) in enumerate(scores.items()):
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[i % len(colors)], lw=2, label=name, marker="o")
        ax.fill(angles, v, alpha=0.06, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(criteria, fontsize=10)
    ax.set_ylim(0, 1); ax.set_title("Validation radar", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1))
    plt.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)
    log.info(f"  Saved radar chart → {outpath}")


def plot_ablation_table(df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.heatmap(df.astype(float), annot=True, fmt=".3f",
                cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Confound Ablation Matrix — Table 2 in paper")
    plt.tight_layout(); fig.savefig(outpath, dpi=180); plt.close(fig)
    log.info(f"  Saved ablation table → {outpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results: dict, outdir: Path) -> None:
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            np.save(outdir / f"{k}.npy", v)
            serialisable[k] = f"<saved to {k}.npy>"
        elif isinstance(v, dict):
            serialisable[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                for kk, vv in v.items()
            }
        elif isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"{k}.csv")
            serialisable[k] = f"<saved to {k}.csv>"
        else:
            serialisable[k] = v
    with open(outdir / "experiment1_results.json", "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    log.info(f"Saved → {outdir}/experiment1_results.json")


def load_frustration_vector(outdir: Path = OUTDIR) -> np.ndarray:
    path = outdir / "frustration_v.npy"
    if not path.exists():
        raise FileNotFoundError(f"Run experiment1 first: {path} not found")
    return np.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment1(
    n_gpqa:       int  = 198,
    n_mmlu:       int  = 500,
    use_aime:     bool = True,
    n_mgsm:       int  = 50,
    n_samples:    int  = 3,
) -> dict:
    """
    Full unified experiment pipeline.

    Datasets (from OpenAI simple-evals public CSVs):
      GPQA Diamond   ~45-55% ← primary contrastive pair engine
      MMLU hard      ~50-70% ← secondary contrastive + difficulty stratification
      AIME 2024+2025 ~64-72% ← extreme difficulty regime
      MGSM non-Latin ~40-70% ← cross-language transfer (Level 5)
    """
    global TARGET_LAYER
    log.info("=" * 68)
    log.info("UNIFIED EXPERIMENT — Pre-Commitment, Emotions & Interaction")
    log.info("=" * 68)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    gpqa_probs  = load_gpqa(n=n_gpqa)
    mmlu_probs  = load_mmlu_hard(n=n_mmlu)
    aime_probs  = load_aime() if use_aime else []
    mgsm_probs  = load_mgsm_nonlatin(n_per_lang=n_mgsm)
    ref_corpus  = load_reference_corpus()

    # Primary corpus = GPQA + MMLU (hard) + AIME — all produce ~40-70% failure
    primary = gpqa_probs + mmlu_probs + aime_probs
    np.random.default_rng(SEED).shuffle(primary)
    split       = int(0.8 * len(primary))
    train_probs = primary[:split]
    test_probs  = primary[split:]

    log.info(f"Primary corpus: {len(primary)}  "
             f"(GPQA={len(gpqa_probs)}, MMLU={len(mmlu_probs)}, AIME={len(aime_probs)})")
    log.info(f"Train={len(train_probs)}  Test={len(test_probs)}  "
             f"MGSM-transfer={len(mgsm_probs)}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, tokenizer = load_model()

    # ── 3. Capture activations ────────────────────────────────────────────────
    def _capture_or_load(tag, problems, n_samples_val, temperature_val=TEMPERATURE):
        saved_probs, saved_caps = load_checkpoint(tag)
        if saved_caps is not None and len(saved_caps) == len(problems):
            return saved_caps
        caps = capture_batch(model, tokenizer, problems,
                             n_samples=n_samples_val, temperature=temperature_val)
        save_checkpoint(tag, problems, caps)
        return caps

    log.info("Capturing train set...")
    train_caps = _capture_or_load("train", train_probs, n_samples)

    log.info("Capturing test set (greedy, n=1)...")
    test_caps  = _capture_or_load("test", test_probs, 1, temperature_val=0.0)

    log.info("Capturing MGSM cross-language transfer set (n=2)...")
    mgsm_caps  = _capture_or_load("mgsm", mgsm_probs, 2)

    log.info("Capturing reference emotion corpus...")
    ref_vectors = capture_reference_vectors(model, tokenizer, ref_corpus, layer=TARGET_LAYER)

    # ── 4. Build traces & contrastive pairs ───────────────────────────────────
    train_traces, pairs = build_traces(train_probs, train_caps)

    # Diagnostic: print first 5 raw answer_text values
    log.info("=== ANSWER TEXT SAMPLES (for grading sanity check) ===")
    for t in train_traces[:5]:
        log.info(f"  correct={t.is_correct}  answer_text={repr(t.raw_answer[:200])}")

    # ── Grading sanity check ──────────────────────────────────────────────────
    n_correct = sum(1 for t in train_traces if t.is_correct)
    log.info(f"  Train accuracy: {n_correct}/{len(train_traces)} "
             f"= {n_correct / max(len(train_traces), 1):.1%}")
    if n_correct == 0:
        raise RuntimeError(
            "Grading returned 0 correct traces. "
            "Inspect _think_boundary output and grade_auto against real model output."
        )

    # ── Checkpoint traces ─────────────────────────────────────────────────────
    log.info("Checkpointing train traces and pairs to disk...")
    with open(OUTDIR / "checkpoint_train_traces.pkl", "wb") as f:
        pickle.dump({"train_traces": train_traces, "pairs": pairs}, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_traces,  _     = build_traces(test_probs,  test_caps)
    mgsm_traces, _      = build_traces(mgsm_probs, mgsm_caps)

    if len(pairs) < 50:
        log.warning(f"⚠ Only {len(pairs)} contrastive pairs — "
                    "consider increasing n_samples or n_gpqa/n_mmlu")

    # ── 5. Layer sweep → select TARGET_LAYER ──────────────────────────────────
    sweep_df, best_layer = layer_sweep(pairs, test_traces)
    plot_layer_sweep(sweep_df, OUTDIR / "fig1_layer_sweep.png")
    sweep_df.to_csv(OUTDIR / "layer_sweep.csv")

    TARGET_LAYER = best_layer
    log.info(f"TARGET_LAYER = {TARGET_LAYER} (empirically selected)")

    # ── 6. Pre-commitment probes (Exp 1 from P1) ─────────────────────────────
    log.info("\n" + "=" * 68)
    log.info("PRE-COMMITMENT PROBES")
    log.info("=" * 68)
    all_traces_combined = train_traces + test_traces
    precommit_results, precommit_layer = precommit_layer_sweep(
        all_traces_combined
    )
    precommit = precommit_results[precommit_layer]
    precommit_probs = precommit["oof_probs"]
    log.info(f"  Pre-commitment AUC={precommit['auc']:.3f} "
             f"@ layer {precommit_layer}")

    # ── 7. Multi-emotion direction extraction (Exp 2 extended) ────────────────
    log.info("\n" + "=" * 68)
    log.info("MULTI-EMOTION DIRECTION EXTRACTION")
    log.info("=" * 68)
    neutral_traces = get_neutral_traces(train_traces)

    # Task-derived directions (frustration, confidence, confusion, desperation)
    emotion_dirs = extract_multi_emotion_directions(
        train_traces, pairs, TARGET_LAYER, neutral_traces
    )

    # Also extract old-style confound directions for ablation matrix
    confound_dirs = extract_confound_directions(train_traces, TARGET_LAYER)

    # Save all directions
    frustration_v = emotion_dirs["frustration"]
    np.save(OUTDIR / "frustration_v.npy", frustration_v)
    for name, v in emotion_dirs.items():
        np.save(OUTDIR / f"emotion_{name}_v.npy", v)
    for name, v in confound_dirs.items():
        np.save(OUTDIR / f"confound_{name}_v.npy", v)
    log.info(f"Saved {len(emotion_dirs)} emotion + "
             f"{len(confound_dirs)} confound directions")

    # ── 8. Five-level validation (per-emotion) ────────────────────────────────
    log.info("\n── Level 1: Representational Geometry ──")
    assert frustration_v.ndim == 1, f"frustration_v shape={frustration_v.shape}, expected 1D"
    for e, v in ref_vectors.items():
        assert v.ndim == 1, f"ref_vectors['{e}'] shape={v.shape}, expected 1D"

    l1 = validate_level1(frustration_v, confound_dirs, ref_vectors)

    log.info("\n── Level 2: Verbal Marker Prediction ──")
    l2 = validate_level2(test_traces, frustration_v, confound_dirs, TARGET_LAYER)

    log.info("\n── Level 4: Confound Ablation Matrix ──")
    ablation_df = validate_level4(l1, l2, frustration_v, confound_dirs, ref_vectors)

    log.info("\n── Level 5: Cross-Domain Transfer (MGSM) ──")
    l5 = validate_level5(mgsm_traces, frustration_v, confound_dirs, TARGET_LAYER)

    # ── 9. INTERACTION ANALYSIS — Exp 3 (The Bridge) ─────────────────────────
    log.info("\n" + "=" * 68)
    log.info("INTERACTION ANALYSIS: EMOTION × PRE-COMMITMENT")
    log.info("=" * 68)

    log.info("\n── Exp 3A: Forward (pre-commitment → emotions) ──")
    forward_df = interaction_forward(
        all_traces_combined, emotion_dirs, ref_vectors,
        precommit_probs, TARGET_LAYER
    )
    forward_df.to_csv(OUTDIR / "interaction_forward.csv")

    log.info("\n── Exp 3B: Backward (emotions → recovery) ──")
    backward_df = interaction_backward(
        all_traces_combined, emotion_dirs, ref_vectors,
        precommit_probs, TARGET_LAYER
    )
    backward_df.to_csv(OUTDIR / "interaction_backward.csv")

    log.info("\n── Exp 3C: Regression decomposition ──")
    regression_results = interaction_regression(
        all_traces_combined, frustration_v, precommit_probs, TARGET_LAYER
    )

    # ── 10. Temporal dynamics with probe overlay ─────────────────────────────
    log.info("\nComputing temporal dynamics with probe overlay...")
    dynamics_basic = compute_temporal_dynamics(
        all_traces_combined, frustration_v, TARGET_LAYER
    )
    dynamics_full = compute_temporal_dynamics_with_probe(
        all_traces_combined, emotion_dirs,
        precommit["probe"], TARGET_LAYER
    )

    # ── 11. Figures ───────────────────────────────────────────────────────────
    log.info("\nGenerating figures...")
    plot_temporal_dynamics(dynamics_basic, OUTDIR / "fig2_temporal_dynamics.png")
    plot_validation_radar(l1, l2, confound_dirs, ref_vectors, frustration_v,
                          OUTDIR / "fig3_validation_radar.png")
    plot_ablation_table(ablation_df, OUTDIR / "fig4_ablation_table.png")

    # ── 12. Save ──────────────────────────────────────────────────────────────
    all_results = {
        # Pre-commitment
        "precommit_auc":       precommit["auc"],
        "precommit_accuracy":  precommit["accuracy"],
        "precommit_layer":     precommit_layer,
        "precommit_v":         precommit.get("probe_direction", None),
        # Emotion directions
        "emotion_dirs":        {k: v for k, v in emotion_dirs.items()},
        "frustration_v":       frustration_v,
        "confound_dirs":       confound_dirs,
        # Validation
        "level1":              l1,
        "level2":              l2,
        "level4_ablation":     ablation_df,
        "level5_transfer":     l5,
        # Interaction
        "interaction_forward":    forward_df,
        "interaction_backward":   backward_df,
        "interaction_regression": regression_results,
        # Meta
        "layer_sweep":         sweep_df,
        "target_layer":        TARGET_LAYER,
        "temporal_dynamics":   {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in dynamics_basic.items()
        },
        "n_pairs":             len(pairs),
        "n_train_traces":      len(train_traces),
        "n_test_traces":       len(test_traces),
        "n_mgsm_traces":       len(mgsm_traces),
    }
    save_results(all_results, OUTDIR)

    # ── 13. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("UNIFIED EXPERIMENT 1 SUMMARY")
    print("=" * 68)
    print(f"\n  PRE-COMMITMENT PROBES")
    print(f"    Best layer:         {precommit_layer}")
    print(f"    AUC:                {precommit['auc']:.3f}")
    print(f"    Accuracy:           {precommit['accuracy']:.3f}")
    print(f"\n  EMOTION DIRECTIONS")
    print(f"    Task-derived:       {list(emotion_dirs.keys())}")
    print(f"    Confound controls:  {list(confound_dirs.keys())}")
    print(f"\n  FRUSTRATION VALIDATION")
    print(f"    L1 (geometry):      {'✅ PASS' if l1['pass'] else '❌ FAIL'}")
    print(f"      cos(f, ref)     = {l1['frus_ref_sim']:.3f}   (> 0.6)")
    print(f"      cos(f, diff)    = {l1['diff_cross_sim']:.3f}   (< 0.3)")
    print(f"    L2 (verbal AUC):    {'✅ PASS' if l2['pass'] else '❌ FAIL'}")
    print(f"      Best direction  = {l2['best_direction']}")
    print(f"    L5 (MGSM xfer):     {'✅ PASS' if l5.get('pass') else '❌ FAIL'}")
    print(f"\n  INTERACTION ANALYSIS")
    if not forward_df.empty:
        top_emo = forward_df.iloc[0]
        cohens_d_val = top_emo["Cohen's d"]
        print(f"    Top forward:        {top_emo['Emotion']}  "
              f"(d={cohens_d_val:.2f}, p={top_emo['p-value']:.4f})")
    if not backward_df.empty:
        top_rec = backward_df.iloc[0]
        print(f"    Top recovery pred:  {top_rec['Emotion']}  "
              f"(AUC={top_rec['Recovery AUC']:.3f})")
    print(f"    Regression R²:      {regression_results['R_squared']:.3f}")
    pre_coef = regression_results["coefficients"].get("pre_commit_failure", {})
    print(f"    β(pre-commit):      {pre_coef.get('beta', 'N/A')}  "
          f"t={pre_coef.get('t', 'N/A')}")
    print(f"\n  Outputs saved to:     {OUTDIR}/")
    print("=" * 68 + "\n")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Test Mode Toggle ─────────────────────────────────────────────────────
    # Set this to True to run a fast 5-minute end-to-end test.
    # Set to False to do the full 5-hour generation run.
    TEST_MODE    = True

    if TEST_MODE:
        log.info("!!! RUNNING IN FAST TEST MODE !!!")
        N_GPQA       = 10     # tiny subset
        N_MMLU       = 10
        USE_AIME     = False  # skip for speed
        N_MGSM       = 2      # 2 per language
        N_SAMPLES    = 3      # 3 is enough; stopping at </think> makes each fast

        # Always wipe cache in TEST_MODE — stale broken captures get reused otherwise
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            log.info('  TEST_MODE: cache wiped')
    else:
        # All datasets loaded from OpenAI simple-evals public CSVs (no HF auth needed)
        N_GPQA       = 198    # GPQA Diamond — grad-level science, acc ~45-55%
        N_MMLU       = 500    # MMLU hard subjects — acc ~50-70%
        USE_AIME     = True   # AIME 2024+2025 — acc ~64-72%
        N_MGSM       = 50     # MGSM per language (6 non-Latin langs) — cross-lang transfer
        N_SAMPLES    = 6      # Increased to 6 to practically guarantee contrastive pairs

    # ─────────────────────────────────────────────────────────────────────────

    run_experiment1(
        n_gpqa=N_GPQA,
        n_mmlu=N_MMLU,
        use_aime=USE_AIME,
        n_mgsm=N_MGSM,
        n_samples=N_SAMPLES,
    )
