"""
Experiment 5: Dual Steering (Emotion x Commitment)
==================================================
RQ7: Can steering emotions and pre-commitment independently and jointly affect accuracy?

Design: 2 x 3 Factorial
- Commitment: [No Steer, Suppress Confidence (-2.0) (Proxy for 'Steer to Wrong')]
- Emotion:    [No Steer, Suppress Frustration (-1.5), Amplify Frustration (+2.0)]

This script tests the independence of emotion and commitment by applying simultaneous
additive residual stream steering vectors at different phases of the thinking trace.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "allenai/Olmo-3-7B-Think"
VECTOR_DIR  = Path("outputs/exp1")
OUTDIR      = Path("outputs/exp5")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_PROBS     = 48    # Use a multiple of 24 for clean GH200 batching
MAX_NEW_TOK = 1024  
TARGET_LAYER = 15   
BATCH_SIZE   = 24   # GH200 optimized

def load_resources():
    """
    Load steering vectors from experiment1's saved .npy files.
    experiment1 saves numpy arrays as .npy files (not inside the JSON).
    """
    global TARGET_LAYER

    res_path = VECTOR_DIR / "experiment1_results.json"
    if not res_path.exists():
        raise FileNotFoundError("experiment1_results.json not found! Run Exp 1 first.")

    with open(res_path, "r") as f:
        data = json.load(f)

    # Read target layer from experiment1's empirical selection
    TARGET_LAYER = data.get("target_layer", TARGET_LAYER)
    print(f"  Using TARGET_LAYER = {TARGET_LAYER}")

    # Load frustration vector from .npy (experiment1 saves it as frustration_v.npy)
    frus_path = VECTOR_DIR / "frustration_v.npy"
    if not frus_path.exists():
        # Fallback: try emotion_frustration_v.npy
        frus_path = VECTOR_DIR / "emotion_frustration_v.npy"
    if not frus_path.exists():
        raise FileNotFoundError(
            f"Frustration vector not found! Looked for:\n"
            f"  {VECTOR_DIR / 'frustration_v.npy'}\n"
            f"  {VECTOR_DIR / 'emotion_frustration_v.npy'}"
        )
    frustration_v = np.load(frus_path)
    print(f"  Loaded frustration vector: {frus_path} (shape={frustration_v.shape})")

    # Load pre-commitment vector (or use -confidence as proxy)
    precommit_path = VECTOR_DIR / "precommit_v.npy"
    if precommit_path.exists():
        precommit_v = np.load(precommit_path)
        print(f"  Loaded precommit vector: {precommit_path}")
    else:
        conf_path = VECTOR_DIR / "emotion_confidence_v.npy"
        if conf_path.exists():
            precommit_v = -np.load(conf_path)
            print(f"  precommit_v.npy not found — using -confidence proxy from {conf_path}")
        else:
            print("  Warning: No precommit or confidence vector found — using zero vector")
            precommit_v = np.zeros_like(frustration_v)

    return {
        "frustration": frustration_v,
        "precommit":   precommit_v,
    }

def dual_steer_hook_factory(v_emo, scale_emo, v_commit, scale_commit):
    """Creates a hook that applies two simultaneous steering vectors."""
    if scale_emo == 0.0 and scale_commit == 0.0:
        return lambda m, i, o: o
        
    t_emo = torch.tensor(v_emo, dtype=torch.bfloat16) if v_emo is not None else None
    t_com = torch.tensor(v_commit, dtype=torch.bfloat16) if v_commit is not None else None
    _logged = [False]  # mutable flag for one-time debug print
    
    def hook(module, inp, out):
        # out[0] works for both tuple and ModelOutput — it's the hidden states
        res = out[0]
        device = res.device
        norm = res.norm(dim=-1, keepdim=True)
        
        delta = torch.zeros_like(res)
        if scale_emo != 0.0 and t_emo is not None:
            delta = delta + scale_emo * norm * t_emo.to(device)
        if scale_commit != 0.0 and t_com is not None:
            delta = delta + scale_commit * norm * t_com.to(device)
        
        # In-place modification — works for both tuple and dataclass outputs
        # because res is a reference to the same tensor object
        res.data.add_(delta)
        
        if not _logged[0]:
            mag = delta.norm().item()
            res_mag = res.norm().item()
            print(f"    [STEER] delta_mag={mag:.2f}  res_mag={res_mag:.2f}  "
                  f"scale_emo={scale_emo}  scale_commit={scale_commit}")
            _logged[0] = True
        
        # Return None — in-place modification means no output replacement needed
    
    return hook

def run_steering_sweep():
    print("Loading vectors...")
    vectors = load_resources()
    frus_v = vectors["frustration"]
    precommit_v = vectors["precommit"]
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    from experiment1_frustration_vector import load_gpqa, grade_auto, _build_prompt
    problems = load_gpqa(n=N_PROBS)
    
    # Define 2x3 Factorial conditions: (Commitment_Scale, Emotion_Scale, Label)
    # Using negative confidence as a proxy for "steering toward wrong answer"
    conditions = [
        ( 0.0,  0.0, "Baseline"),
        ( 0.0, -1.5, "Emo: Suppress Frustration"),
        ( 0.0,  2.0, "Emo: Amplify Frustration"),
        (-2.0,  0.0, "Commit: Push Wrong"),
        (-2.0, -1.5, "Commit: Push Wrong + Suppress Frus"),
        (-2.0,  2.0, "Commit: Push Wrong + Amplify Frus"),
    ]
    
    results = {c[2]: [] for c in conditions}
    
    print("\nRunning 2x3 Dual Steering Sweep...")
    for commit_s, emo_s, label in conditions:
        print(f"\nCondition: {label}")
        hook = dual_steer_hook_factory(frus_v, emo_s, precommit_v, commit_s)
        handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
        
        pbar = tqdm(total=len(problems))
        for i in range(0, len(problems), BATCH_SIZE):
            chunk = problems[i:i + BATCH_SIZE]
            prompts = [_build_prompt(tokenizer, p.get("problem", "")) for p in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK, do_sample=False, use_cache=True)
                
            for j, p in enumerate(chunk):
                ans_text = tokenizer.decode(out[j][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                is_correct = grade_auto(ans_text, p["answer"], p["format"])
                results[label].append(int(is_correct))
                
            pbar.update(len(chunk))
            torch.cuda.empty_cache()
            
        pbar.close()
        handle.remove()
        
    print("\n" + "="*50)
    print("EXPERIMENT 5: DUAL STEERING RESULTS")
    print("="*50)
    
    summary = {}
    for label, accs in results.items():
        acc = np.mean(accs)
        summary[label] = acc
        print(f"{label:40} | Accuracy: {acc:.1%}")
        
    with open(OUTDIR / "factorial_results.json", "w") as f:
        json.dump({"summary": summary, "raw_binary": results}, f, indent=2)
    print(f"\nResults saved to {OUTDIR}/factorial_results.json")

if __name__ == "__main__":
    run_steering_sweep()
