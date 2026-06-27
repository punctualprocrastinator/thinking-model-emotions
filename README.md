# Thinking Model Emotions: Pre-Commitment and Functional Frustration in Extended Thinking

**Mechanistic interpretability of emotional dynamics in reasoning models, and how the same dynamics can be induced as an attack surface or read as a safety monitor**

[![Paper](https://img.shields.io/badge/Technical_Report-PDF-blue)](technical_report_precommitment_frustration.md)

---

## Key Findings

Using contrastive activation analysis on **OLMo-3-7B-Think** across 596 reasoning traces (GPQA Diamond, MMLU Hard, MGSM). Of those 596, only 46 form correct/wrong **contrastive pairs** (same problem, one sample right, one sample wrong); that 46-pair subset is what the frustration direction is extracted from:

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

If thinking models pre-commit to answers before reasoning, then Chain-of-Thought monitoring, the primary oversight tool for frontier models, may be monitoring post-hoc rationalisation rather than genuine deliberation. The internal emotional dynamics we detect (frustration, arousal, epistemic engagement) operate below the text layer and are invisible to output-level monitoring. A growing 2025-26 literature shows this internal/external gap is also an active **attack surface**, not just a naturally occurring curiosity.

This work is Phase 0 of **RedArc Labs' emotion-aware monitoring program**: turning these findings into a working proof-of-concept runtime monitor, `EmotionMonitor`, that reads pre-commitment and frustration signals at inference time and logs, steers, or halts generation before a wrong or harmful output is produced. See [PROJECT.md](PROJECT.md) for the system architecture and execution plan, [research_direction.md](research_direction.md) for the full research narrative, and [ATTACK_SURFACE_AND_DEFENSE.md](ATTACK_SURFACE_AND_DEFENSE.md) for the threat model and detection-layer design.

---

## Attack Surface: How the Same Signals Could Be Induced

The findings above describe a *naturally occurring* phenomenon: frustration rises and falls on its own as the model reasons. Separately, we are characterizing whether these same signals can be deliberately *induced*, since three distinct mechanisms already exist in the literature for doing exactly that to comparable internal directions:

- **Steering.** Push the emotion direction directly at inference time. Anthropic's 2026 dose-response result already shows this causally in a closed model: steering toward "desperate" raises blackmail compliance from 22% to 72%; steering toward "calm" drops it to 0%. We have not run this ourselves yet; it is a funded next step (`PROJECT.md` Run 5).
- **Directional ablation.** Suppress the emotion direction at every token during generation, so the model's Chain-of-Thought text stays fluent while its actual internal state has been flattened. Demonstrated on a "caution" direction by Yamaguchi, Etheridge, and Arditi (ICML 2025 Workshop).
- **Backdoor-induced decoupling.** Bake the decoupling into training, so a trigger pairs clean-looking reasoning with a steered answer, no runtime intervention needed at inference. Demonstrated by MirageBackdoor (2026).

A fourth question, **monitor/probe evasion** (can the detector itself, not the model, be fooled by an adversarial prompt), is named as an open mechanism we have not yet tested. See `ATTACK_SURFACE_AND_DEFENSE.md` §1 for the full mechanism breakdown and detection-layer design, and `PROJECT.md` §4.3a-§4.6 for how this connects to the `EmotionMonitor` system and an honest accounting of what's measured vs. hypothesized, including the jailbreak-relevant version of this story, which is currently an analogy to Anthropic's data rather than our own result.

---

## Current Status

- **Phase 0 complete.** The findings above, produced on a $0, compute-constrained run (single GH200, one weekend, one model). This is a **preliminary approximation** of the full methodology, not yet the exact protocol at scale. See `PROJECT.md` §2.1.
- **In progress now, zero additional cost.** Running the synthetic attack-surface validation in `ATTACK_SURFACE_AND_DEFENSE.md` §4: testing whether the same monitoring primitives that catch naturally occurring frustration-driven decoupling also catch synthetic stand-ins for two documented adversarial mechanisms (directional ablation, backdoor-induced decoupling). CPU-only, no live attack, no GPU dependency.
- **Pending funding, highest-leverage next steps.** Multi-model replication on a second, architecturally distinct model family, and causal validation via activation patching plus calm-vector dose-response steering. See `PROJECT.md` §4.6 for the full priority ranking of what actually moves novelty and scope.
- **Open research question, scope decision pending.** Whether to test two newly identified, untested attacks on our own constructs: "recovery suppression" (can the frustration signal that predicts self-correction be selectively suppressed to keep a model deterministically stuck in a wrong answer?) and "Mechanism C" (can the monitor itself, not just the model, be fooled by an adversarial prompt?). Both require live model access and are flagged, not yet decided. See `PROJECT.md` §4.5-4.6.

---

## Repository Structure

```
├── experiment1_frustration_vector.py     # Main pipeline: trace generation,
│                                         # activation capture, direction extraction,
│                                         # 5-level validation, interaction analysis
├── experiment5_dual_steering.py          # Causal steering (2×3 factorial)
├── technical_report_precommitment_frustration.md  # Full technical report
├── research_direction.md                 # Research narrative: lit review, threat model, EmotionMonitor design
├── PROJECT.md                             # System architecture + $100 RunPod execution plan
├── ATTACK_SURFACE_AND_DEFENSE.md          # Attack mechanisms + detection-layer spec
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

All experiments run on a single **NVIDIA GH200** (101.5 GB VRAM). The pipeline uses pure PyTorch hooks for activation capture, no nnsight, no TransformerLens, ensuring exact reproducibility.

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
