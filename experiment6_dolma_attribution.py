"""
Experiment 6: Training Data Attribution (OLMo-3 Exclusive)
=========================================================
Goal: Identify where in the Dolma 3 pretraining data the emotion directions originate.

Method:
1. Load validated emotion vectors (.npy) from Exp 1.
2. Sample documents from the Dolma 3 dataset (used for OLMo-3-Think).
3. Score documents by their activation of each emotion direction.
4. Categorize high-activation documents to distinguish the "Method Actor" 
   vs. "Functional" hypotheses.

Requirements:
    pip install datasets pandas tqdm torch transformers
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID    = "allenai/Olmo-3-7B-Think"
VECTOR_DIR  = Path("outputs/exp1")
OUTDIR      = Path("outputs/exp6")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Dolma 3 sample (using a subset of the official pretraining data)
DOLMA_DATASET = "allenai/dolma" 
SAMPLE_SIZE   = 1000  # Number of documents to score
MAX_SEQ_LEN   = 512   # Tokens per chunk
TARGET_LAYER  = 15    # Must match the best layer from Exp 1 sweep

def load_vectors():
    """Load the full suite of emotion vectors saved by Experiment 1."""
    res_path = VECTOR_DIR / "experiment1_results.json"
    
    if not res_path.exists():
        raise FileNotFoundError(f"Missing {res_path}. Run Experiment 1 first.")
        
    with open(res_path, "r") as f:
        data = json.load(f)
        
    emotion_dirs = data.get("emotion_dirs", {})
    if "frustration_v" not in emotion_dirs:
        raise ValueError("Missing required vectors in results.")
        
    # Return all 10 emotion vectors for attribution scoring
    return {k: np.array(v) for k, v in emotion_dirs.items()}

def get_activations(model, tokenizer, texts, layer):
    """Batch process texts and extract activations at the target layer."""
    device = next(model.parameters()).device
    activations = []
    
    for text in tqdm(texts, desc="Scoring Dolma chunks"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                           max_length=MAX_SEQ_LEN).to(device)
        
        captured = {}
        def hook_fn(module, inp, out):
            # Handle tuple output (hidden_states, ...)
            h = out[0] if isinstance(out, tuple) else out
            captured["act"] = h.detach().cpu().float()

        handle = model.model.layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # Mean-pool activations over the sequence (excluding padding/bos)
        # Note: We take the mean across the sequence to get a document-level score
        act = captured["act"][0].numpy().mean(axis=0)
        activations.append(act)
        
    return np.stack(activations)

def run_attribution():
    # 1. Load model & vectors
    vectors = load_vectors()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    # 2. Load Dolma sample
    # Note: Using 'v1_7' or similar specific version if needed
    print(f"Loading Dolma sample ({SAMPLE_SIZE} docs)...")
    ds = load_dataset(DOLMA_DATASET, "v1_7", split="train", streaming=True)
    
    docs = []
    doc_metadata = []
    for i, ex in enumerate(ds):
        if i >= SAMPLE_SIZE: break
        docs.append(ex["text"])
        doc_metadata.append({
            "id": ex.get("id", f"doc_{i}"),
            "source": ex.get("source", "unknown"),
            "text_snippet": ex["text"][:200].replace("\n", " ")
        })

    # 3. Score documents
    acts = get_activations(model, tokenizer, docs, TARGET_LAYER)
    
    results = []
    for name, vec in vectors.items():
        if vec is None: continue
        
        # Calculate dot products (activations)
        scores = np.dot(acts, vec)
        
        # Rank
        top_indices = np.argsort(scores)[::-1][:50]
        
        emotion_results = []
        for idx in top_indices:
            res = doc_metadata[idx].copy()
            res["score"] = float(scores[idx])
            emotion_results.append(res)
            
        results.append({
            "emotion": name,
            "top_documents": emotion_results
        })

    # 4. Save results
    with open(OUTDIR / "attribution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Attribution complete. Results saved to {OUTDIR}/attribution_results.json")
    
    # 5. Simple summary report
    print("\n-- Attribution Summary --")
    for res in results:
        print(f"\nTop sources for '{res['emotion']}':")
        sources = pd.Series([d["source"] for d in res["top_documents"]]).value_counts()
        print(sources.head(5))

if __name__ == "__main__":
    try:
        run_attribution()
    except Exception as e:
        print(f"Error in attribution: {e}")
