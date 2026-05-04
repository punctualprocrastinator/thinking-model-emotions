"""
Experiment 1, NNsight capture version.

This keeps the original experiment's loaders, graders, trace construction,
probes, direction extraction, validation, plotting, and result saving, but
replaces the PyTorch hook capture engine with NNsight.

Install:
    pip install nnsight transformers accelerate datasets scikit-learn matplotlib seaborn pandas tqdm scipy

Pilot run:
    python experiment1_frustration_vector_nnsight.py --pilot

Full-ish run:
    python experiment1_frustration_vector_nnsight.py --n-gpqa 198 --n-mmlu 500 --use-aime --n-mgsm 50 --n-samples 2 --max-new-tokens 2048
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

try:
    from nnsight import LanguageModel
except ImportError as exc:
    raise SystemExit(
        "nnsight is not installed. Run: pip install nnsight"
    ) from exc

import experiment1_frustration_vector as base


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


MODEL_ID = "allenai/Olmo-3-7B-Think"

# Fast defaults. Scale these up only after a pilot run works.
LAYERS = [15]
TARGET_LAYER = 15
MAX_NEW_TOK = 512
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42
OUTDIR = Path("outputs/exp1_nnsight")
OUTDIR.mkdir(parents=True, exist_ok=True)


# Edit these variables to configure the run.
PILOT = True
N_GPQA = 40
N_MMLU = 80
USE_AIME = False
N_MGSM = 0
N_SAMPLES = 2
RUN_LAYERS = [15]
RUN_TARGET_LAYER = 15
RUN_MAX_NEW_TOKENS = 512
RUN_OUTDIR = OUTDIR


def configure_base(layers: list[int], outdir: Path) -> None:
    """Keep imported downstream functions aligned with this script's config."""
    base.LAYERS = layers
    base.TARGET_LAYER = layers[0]
    base.OUTDIR = outdir


def load_model_nnsight(
    model_id: str = MODEL_ID,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    dispatch: bool = True,
) -> LanguageModel:
    """
    Load a HuggingFace causal LM through NNsight.

    NNsight docs use:
        LanguageModel(model_id, device_map="auto", dispatch=True)

    Extra kwargs are forwarded to the HF loader in recent NNsight versions.
    """
    log.info("Loading %s with NNsight...", model_id)
    kwargs = dict(
        device_map=device_map,
        dispatch=dispatch,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    model = LanguageModel(model_id, **kwargs)
    model.tokenizer.padding_side = "left"
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    log.info("Loaded NNsight LanguageModel.")
    return model


def _layer_output(nn_model: LanguageModel, layer_idx: int):
    """
    OLMo HF models expose transformer blocks at model.model.layers.
    Layer outputs are tuples; index 0 is residual stream hidden states.
    """
    return nn_model.model.layers[layer_idx].output[0]


def _build_prompt(tokenizer, problem: str) -> str:
    return base._build_prompt(tokenizer, problem)


def _decode_token_list(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "value"):
        x = x.value
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", dtype=torch.float16).numpy()
    return np.asarray(x)


def generate_ids_nnsight(
    nn_model: LanguageModel,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Generate with NNsight and return full token ids: prompt + completion.
    """
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=nn_model.tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        with nn_model.generate(prompt, **gen_kwargs):
            output = nn_model.generator.output.save()

    if hasattr(output, "value"):
        output = output.value
    return output.detach().cpu()


def trace_full_sequence_nnsight(
    nn_model: LanguageModel,
    full_ids: torch.Tensor,
    layers: list[int],
) -> dict[int, np.ndarray]:
    """
    Run one NNsight trace on the completed sequence and save selected layer
    residual streams.

    This mirrors the old two-pass hook design: generate first, then do one
    teacher-forced pass over the full sequence so every generated token has a
    normal residual-stream activation.
    """
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)

    # NNsight LanguageModel accepts tokenized inputs. Keeping this as a dict
    # avoids re-tokenizing decoded text and preserves exact generated ids.
    inputs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
    }

    saved = {}
    with torch.inference_mode():
        with nn_model.trace(inputs):
            for li in layers:
                saved[li] = _layer_output(nn_model, li).save()

    return {li: _to_numpy(val) for li, val in saved.items()}


def capture_batch_nnsight(
    nn_model: LanguageModel,
    problems: list[dict],
    *,
    layers: list[int] = LAYERS,
    n_samples: int = 2,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOK,
) -> list[list[dict | None]]:
    """
    Capture completed thinking traces with NNsight.

    Output format matches base.capture_batch(), so base.build_traces() and the
    rest of the original analysis can be reused unchanged.
    """
    tokenizer = nn_model.tokenizer
    all_results: list[list[dict | None]] = []

    for prob in tqdm(problems, desc="Capturing with NNsight"):
        prompt = _build_prompt(tokenizer, prob.get("problem", ""))
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        samples: list[dict | None] = []

        for _ in range(n_samples):
            try:
                full_ids = generate_ids_nnsight(
                    nn_model,
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                ids = full_ids[0].tolist()
                new_ids = ids[prompt_len:]

                think_end = base._think_boundary(new_ids, tokenizer)
                think_ids = new_ids[:think_end]
                answer_ids = new_ids[think_end:]

                if not think_ids:
                    samples.append(None)
                    continue

                captured = trace_full_sequence_nnsight(nn_model, full_ids, layers)

                acts_per_layer = []
                pre_think_per_layer = []
                for li in layers:
                    rs = captured[li]
                    acts_per_layer.append(rs[0, prompt_len: prompt_len + think_end, :])
                    pre_think_per_layer.append(rs[0, prompt_len - 1, :])

                samples.append({
                    "think_acts": np.stack(acts_per_layer),
                    "pre_think_act": np.stack(pre_think_per_layer),
                    "think_tokens": _decode_token_list(tokenizer, think_ids),
                    "think_text": tokenizer.decode(think_ids, skip_special_tokens=True),
                    "answer_text": tokenizer.decode(answer_ids, skip_special_tokens=True),
                })
            except Exception as exc:
                log.warning("Capture failed [%s]: %s", prob.get("id", "?"), exc)
                samples.append(None)

        all_results.append(samples)

    return all_results


def capture_reference_vectors_nnsight(
    nn_model: LanguageModel,
    corpus: dict[str, list[str]],
    *,
    layer: int = TARGET_LAYER,
) -> dict[str, np.ndarray]:
    """Reference emotion vectors with NNsight single-pass tracing."""
    tokenizer = nn_model.tokenizer
    refs: dict[str, np.ndarray] = {}

    for emotion, stories in corpus.items():
        vecs = []
        for story in tqdm(stories, desc=f"Reference {emotion}", leave=False):
            ids = tokenizer.encode(story, return_tensors="pt", add_special_tokens=False)
            inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            with torch.inference_mode():
                with nn_model.trace(inputs):
                    hidden = _layer_output(nn_model, layer).save()
            h = _to_numpy(hidden)[0]
            start = min(50, max(0, h.shape[0] - 1))
            vecs.append(h[start:].mean(axis=0))

        v = np.stack(vecs).mean(axis=0)
        refs[emotion] = v / (np.linalg.norm(v) + 1e-8)

    return refs


def run_experiment1_nnsight(
    *,
    n_gpqa: int,
    n_mmlu: int,
    use_aime: bool,
    n_mgsm: int,
    n_samples: int,
    layers: list[int],
    target_layer: int,
    max_new_tokens: int,
    outdir: Path = OUTDIR,
) -> dict:
    """NNsight-backed version of the original unified experiment."""
    configure_base(layers, outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gpqa_probs = base.load_gpqa(n=n_gpqa)
    mmlu_probs = base.load_mmlu_hard(n=n_mmlu)
    aime_probs = base.load_aime() if use_aime else []
    mgsm_probs = base.load_mgsm_nonlatin(n_per_lang=n_mgsm) if n_mgsm > 0 else []
    ref_corpus = base.load_reference_corpus(outdir / "reference_corpus.json")

    primary = gpqa_probs + mmlu_probs + aime_probs
    np.random.default_rng(SEED).shuffle(primary)
    split = int(0.8 * len(primary))
    train_probs = primary[:split]
    test_probs = primary[split:]

    nn_model = load_model_nnsight()

    train_caps = capture_batch_nnsight(
        nn_model, train_probs, layers=layers, n_samples=n_samples,
        max_new_tokens=max_new_tokens,
    )
    test_caps = capture_batch_nnsight(
        nn_model, test_probs, layers=layers, n_samples=1, temperature=0.0,
        max_new_tokens=max_new_tokens,
    )
    mgsm_caps = capture_batch_nnsight(
        nn_model, mgsm_probs, layers=layers, n_samples=1,
        max_new_tokens=max_new_tokens,
    ) if mgsm_probs else []
    ref_vectors = capture_reference_vectors_nnsight(nn_model, ref_corpus, layer=target_layer)

    train_traces, pairs = base.build_traces(train_probs, train_caps)
    test_traces, _ = base.build_traces(test_probs, test_caps)
    mgsm_traces, _ = base.build_traces(mgsm_probs, mgsm_caps) if mgsm_probs else ([], [])

    if len(layers) > 1:
        sweep_df, best_layer = base.layer_sweep(pairs, test_traces, layers=layers)
        base.plot_layer_sweep(sweep_df, outdir / "fig1_layer_sweep.png")
        sweep_df.to_csv(outdir / "layer_sweep.csv")
        target_layer = best_layer
    else:
        sweep_df = None
        target_layer = layers[0]

    base.TARGET_LAYER = target_layer
    all_traces = train_traces + test_traces

    precommit_results, precommit_layer = base.precommit_layer_sweep(all_traces, layers=layers)
    precommit = precommit_results[precommit_layer]
    neutral_traces = base.get_neutral_traces(train_traces)
    emotion_dirs = base.extract_multi_emotion_directions(
        train_traces, pairs, target_layer, neutral_traces
    )
    confound_dirs = base.extract_confound_directions(train_traces, target_layer)

    frustration_v = emotion_dirs["frustration"]
    np.save(outdir / "frustration_v.npy", frustration_v)
    for name, vec in emotion_dirs.items():
        np.save(outdir / f"emotion_{name}_v.npy", vec)
    for name, vec in confound_dirs.items():
        np.save(outdir / f"confound_{name}_v.npy", vec)

    l1 = base.validate_level1(frustration_v, confound_dirs, ref_vectors)
    l2 = base.validate_level2(test_traces, frustration_v, confound_dirs, target_layer)
    ablation_df = base.validate_level4(l1, l2, frustration_v, confound_dirs, ref_vectors)
    l5 = base.validate_level5(mgsm_traces, frustration_v, confound_dirs, target_layer) if mgsm_traces else {}

    forward_df = base.interaction_forward(
        all_traces, emotion_dirs, ref_vectors, precommit["oof_probs"], target_layer
    )
    backward_df = base.interaction_backward(
        all_traces, emotion_dirs, ref_vectors, precommit["oof_probs"], target_layer
    )
    regression_results = base.interaction_regression(
        all_traces, frustration_v, precommit["oof_probs"], target_layer
    )
    dynamics_basic = base.compute_temporal_dynamics(all_traces, frustration_v, target_layer)
    dynamics_full = base.compute_temporal_dynamics_with_probe(
        all_traces, emotion_dirs, precommit["probe"], target_layer
    )

    base.plot_temporal_dynamics(dynamics_basic, outdir / "fig2_temporal_dynamics.png")
    base.plot_validation_radar(
        l1, l2, confound_dirs, ref_vectors, frustration_v,
        outdir / "fig3_validation_radar.png",
    )
    base.plot_ablation_table(ablation_df, outdir / "fig4_ablation_table.png")

    results = {
        "backend": "nnsight",
        "model_id": MODEL_ID,
        "layers": layers,
        "target_layer": target_layer,
        "max_new_tokens": max_new_tokens,
        "precommit_auc": precommit["auc"],
        "precommit_accuracy": precommit["accuracy"],
        "precommit_layer": precommit_layer,
        "emotion_dirs": emotion_dirs,
        "frustration_v": frustration_v,
        "confound_dirs": confound_dirs,
        "level1": l1,
        "level2": l2,
        "level4_ablation": ablation_df,
        "level5_transfer": l5,
        "interaction_forward": forward_df,
        "interaction_backward": backward_df,
        "interaction_regression": regression_results,
        "layer_sweep": sweep_df,
        "temporal_dynamics": {
            key: val.tolist() if isinstance(val, np.ndarray) else val
            for key, val in dynamics_basic.items()
        },
        "temporal_dynamics_with_probe_keys": list(dynamics_full.keys()),
        "n_pairs": len(pairs),
        "n_train_traces": len(train_traces),
        "n_test_traces": len(test_traces),
        "n_mgsm_traces": len(mgsm_traces),
    }
    base.save_results(results, outdir)

    print(json.dumps({
        "backend": "nnsight",
        "outdir": str(outdir),
        "n_pairs": len(pairs),
        "n_train_traces": len(train_traces),
        "n_test_traces": len(test_traces),
        "precommit_auc": round(float(precommit["auc"]), 3),
        "target_layer": target_layer,
        "level1_pass": bool(l1.get("pass")),
        "level2_pass": bool(l2.get("pass")),
    }, indent=2))

    return results


if __name__ == "__main__":
    if PILOT:
        n_gpqa = min(N_GPQA, 40)
        n_mmlu = min(N_MMLU, 80)
        use_aime = False
        n_mgsm = 0
        layers = [RUN_TARGET_LAYER]
        max_new_tokens = min(RUN_MAX_NEW_TOKENS, 512)
    else:
        n_gpqa = N_GPQA
        n_mmlu = N_MMLU
        use_aime = USE_AIME
        n_mgsm = N_MGSM
        layers = RUN_LAYERS
        max_new_tokens = RUN_MAX_NEW_TOKENS

    run_experiment1_nnsight(
        n_gpqa=n_gpqa,
        n_mmlu=n_mmlu,
        use_aime=use_aime,
        n_mgsm=n_mgsm,
        n_samples=N_SAMPLES,
        layers=layers,
        target_layer=RUN_TARGET_LAYER,
        max_new_tokens=max_new_tokens,
        outdir=RUN_OUTDIR,
    )
