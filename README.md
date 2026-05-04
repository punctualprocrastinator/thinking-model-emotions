# Thinking Model Emotions: Pre-Commitment and Functional Frustration in Extended Thinking

**Mechanistic interpretability of emotional dynamics in reasoning models**

[![Paper](https://img.shields.io/badge/Technical_Report-PDF-blue)](technical_report_precommitment_frustration.md)

---

## Key Findings

Using contrastive activation analysis on **OLMo-3-7B-Think** across 596 reasoning traces (GPQA Diamond, MMLU Hard, MGSM):

| Finding | Metric | Value |
|---|---|---|
| **Pre-commitment** | AUC (5-fold CV, n=480) | **0.780** |
| **Frustration direction** | Verbal marker AUC | **0.598** |
| **Recovery prediction** | AUC among pre-wrong traces | **0.666** |
| **Cross-lingual transfer** | t-test (6 languages) | **p=0.0017** |
| **Emotion × commitment** | 4 significant interactions | **p<0.002** |

> **The model knows whether it will succeed before it starts thinking** (AUC = 0.780).  
> Internal frustration dynamics predict self-correction but never surface in the CoT text.

## What This Means for AI Safety

If thinking models pre-commit to answers before reasoning, then Chain-of-Thought monitoring — the primary oversight tool for frontier models — may be monitoring post-hoc rationalisation rather than genuine deliberation. The internal emotional dynamics we detect (frustration, arousal, epistemic engagement) operate below the text layer and are invisible to output-level monitoring.

This work is Phase 0 of a research program on **emotion deflection vectors** — representations that activate when models suppress internal states from their output. See [the ERA proposal](era.md) for the full research agenda.

---

## Repository Structure

```
├── experiment1_frustration_vector.py     # Main pipeline: trace generation,
│                                         # activation capture, direction extraction,
│                                         # 5-level validation, interaction analysis
├── experiment5_dual_steering.py          # Causal steering (2×3 factorial)
├── technical_report_precommitment_frustration.md  # Full technical report
├── era.md                                # Research proposal with preliminary results
├── outputs/
│   ├── experiment1_results.json          # All metrics, tables, vectors
│   ├── layer_sweep.csv                   # Table 1: layer sweep
│   ├── interaction_forward.csv           # Table 3: emotion × commitment
│   ├── interaction_backward.csv          # Table 4: recovery prediction
│   └── level4_ablation.csv              # Confound ablation matrix
└── README.md
```

## Quick Start

```bash
# Requires: GH200 or A100 (80GB+), PyTorch, transformers, flash-attn
pip install transformers accelerate flash-attn scipy scikit-learn matplotlib pandas

# Run full pipeline (~6 hours on GH200)
python experiment1_frustration_vector.py
```

## Hardware

All experiments run on a single **NVIDIA GH200** (101.5 GB VRAM). The pipeline uses pure PyTorch hooks for activation capture — no nnsight, no TransformerLens — ensuring exact reproducibility.

## Citation

```
@techreport{thinking-model-emotions-2026,
  title={Pre-Commitment and Functional Frustration in Extended Thinking: 
         Mechanistic Evidence from OLMo-3-7B-Think},
  year={2026},
  url={https://github.com/punctualprocrastinator/thinking-model-emotions}
}
```

## License

MIT
