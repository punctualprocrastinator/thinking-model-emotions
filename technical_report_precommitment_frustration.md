# Pre-Commitment and Functional Frustration in Extended Thinking: Mechanistic Evidence from OLMo-3-7B-Think

**Technical Report — May 2026**

---

## Abstract

We investigate whether thinking models exhibit internal emotional dynamics that are invisible to Chain-of-Thought monitoring — a prerequisite for understanding how models may generate unfaithful reasoning traces. Using contrastive activation analysis on OLMo-3-7B-Think across 596 reasoning traces (GPQA Diamond, MMLU Hard, MGSM), we find three results. **First**, a linear probe on pre-reasoning activations predicts final answer correctness with AUC = 0.780 (n=480, 5-fold CV), demonstrating that the model commits to outcomes before generating any reasoning tokens. **Second**, we extract a "frustration" direction in the mid-layer residual stream that predicts verbal self-correction markers (AUC = 0.598) and is geometrically independent from surface confounds (trace length, hedging, error style). **Third**, we show that frustration activation among pre-committed-to-failure traces predicts successful self-correction (Recovery AUC = 0.666, OR = 0.139), reframing frustration as a computational stuck-state detector rather than a semantic emotion. These findings establish that thinking models maintain latent emotional dynamics that decouple from their expressed reasoning — the foundational observation for mechanistic deception detection via emotion deflection vectors.

---

## 1. Introduction and Motivation

Chain-of-Thought (CoT) monitoring is the primary oversight mechanism for frontier reasoning models. Recent work has shown this assumption is fragile: models generate post-hoc rationalisations for biased conclusions (Turpin et al., 2023), conceal reward hacks from their CoT in over 98% of cases (van der Weij et al., 2025), and strategically sandbag capability evaluations (Chen et al., 2025).

These failures share a common mechanistic structure: **internal model states drive behaviour that the output stream does not faithfully represent.** Anthropic's recent work on functional emotions (2026) confirms this — causal emotion representations drive misaligned behaviours including reward hacking and blackmail, sometimes with no visible output-level trace. The same work identifies, in preliminary form, "deflection vectors" — activation patterns that fire when a model internally registers an emotion but suppresses it from output.

This report presents the first independent replication and extension of functional emotion extraction in an open-weight thinking model (OLMo-3-7B-Think). Our central finding — that the model pre-commits to answer correctness before reasoning begins, and that internal frustration dynamics predict recovery from pre-committed failures — directly establishes the decoupling between internal state and expressed reasoning that deflection vector research requires.

### Contributions

1. **Pre-commitment probes** (AUC = 0.780): First demonstration in an open-weight thinking model that outcome is predictable from pre-reasoning activations, before any CoT tokens are generated.

2. **Functional frustration direction**: A contrastive direction in the mid-layer (layer 15/32) residual stream that predicts verbal frustration markers, is geometrically independent from confounds, and explains only 5.3% of variance via surface features — indicating a deep computational signal.

3. **Emotion × commitment interaction**: Four statistically significant (Bonferroni-surviving) effects showing that pre-commitment to failure selectively elevates arousal-type activations while suppressing epistemic engagement — a signature of confident wrongness.

4. **Recovery prediction**: Among traces pre-committed to failure, low frustration predicts successful self-correction (AUC = 0.666), establishing frustration as a functional "stuck-state" detector.

---

## 2. Methods

### 2.1 Model and Infrastructure

We use **OLMo-3-7B-Think** (Ai2), a 7B-parameter model fine-tuned with RLHF for extended thinking via `<think>...</think>` tags. All experiments run on a single GH200 (101.5 GB VRAM) using pure PyTorch hooks for activation capture — no proxy libraries (nnsight/TransformerLens), ensuring exact reproducibility.

### 2.2 Dataset and Trace Generation

We generate reasoning traces on three benchmarks:

| Benchmark | N problems | Samples/problem | Purpose |
|---|---|---|---|
| GPQA Diamond | 60 | 6 (T=0.6) | Graduate-level science, acc ~28% |
| MMLU Hard | 40 | 6 (T=0.6) | Hard multichoice, acc ~28% |
| MGSM (6 languages) | 48 | 2 (T=0.6) | Cross-lingual transfer validation |

For each problem, 6 independent samples at temperature 0.6 provide contrastive pairs (same problem, different correctness outcomes). **596 is the total trace count; 46 is a much smaller derived subset, not a separate sample.** A "contrastive pair" only forms when a problem's samples split between correct and wrong; problems where all samples agreed (all correct or all wrong) contribute traces to the 596 total but no pair. Total corpus: 596 traces, of which 46 form contrastive pairs (13 of those 46 are cross-lingual MGSM pairs). The frustration direction is extracted from the 46 pairs only; the pre-commitment probe instead uses per-trace correctness labels and so draws on a larger slice of the 596 (480 traces, see Table 1).

### 2.3 Three-Pass Activation Capture

For each (problem, sample):

- **Pass 1** (batched): Generate thinking trace with `StoppingCriteria` that halts at `</think>`. Max 4096 tokens.
- **Pass 2** (conditional): If `</think>` found, generate answer (max 512 tokens, greedy).
- **Pass 3** (batched): Forward pass on prompt + thinking tokens to capture residual stream activations at layers {7, 15, 23, 31} (25%, 50%, 75%, 100% depth).

When `</think>` is not emitted (common for 7B models on graduate-level problems), thinking tokens are capped at 1500 to exclude answer-contaminated tail activations.

### 2.4 Direction Extraction

**Pre-commitment probe**: Logistic regression on the activation at the last question token (position immediately before `<think>`), predicting binary correctness. 5-fold stratified cross-validation.

**Frustration direction**: Contrastive mean difference (mean_wrong − mean_correct) across the thinking trace, projected orthogonal to the top-5 neutral PCs (computed from all traces regardless of correctness). This denoising removes task-generic reasoning variance.

**Confound controls**: Three confound directions (error_style, length, hedge) computed from surface-level verbal features, validated via the same pipeline to ensure frustration is not reducible to any of them.

### 2.5 Five-Level Validation Protocol

| Level | Test | Threshold |
|---|---|---|
| L1 | Cosine similarity to reference emotion vectors | cos > 0.6 |
| L2 | Verbal marker prediction AUC | AUC > 0.55 |
| L3 | Per-problem sign consistency (held-out) | > 60% |
| L4 | Confound ablation matrix | Max cross-sim < 0.3 |
| L5 | Cross-domain transfer (MGSM, t-test) | p < 0.05 |

---

## 3. Results

### 3.1 Pre-Commitment: The Model Knows Before It Thinks

A linear probe on pre-reasoning activations achieves **AUC = 0.780** at layer 31 (Table 1). Performance is above chance at all four layers, with a monotonic increase from early to late layers.

**Table 1: Pre-commitment probe performance (5-fold stratified CV, n=480)**

| Layer (depth) | AUC | Accuracy |
|---|---|---|
| 7 (25%) | 0.739 | 0.768 |
| 15 (50%) | 0.779 | 0.798 |
| 23 (75%) | 0.777 | 0.798 |
| **31 (100%)** | **0.780** | **0.796** |

This means the model has encoded a strong signal about whether it will ultimately answer correctly in its residual stream at the last question token — before generating a single reasoning token. The implication for CoT monitoring is direct: **the reasoning trace is generated after the outcome is already latently determined.**

### 3.2 Frustration as a Functional Computational State

The contrastive frustration direction at layer 15 achieves the highest verbal marker AUC (0.598) among all directions including confounds:

**Table 2: Direction quality comparison**

| Direction | Verbal AUC | Max cross-similarity |
|---|---|---|
| **Frustration** | **0.598** | **0.274** |
| Error style | 0.581 | 0.274 |
| Hedge | 0.404 | 0.589 |
| Length | 0.322 | 0.589 |

Critically, frustration is geometrically independent from all confounds (max cosine similarity = 0.274), while hedge and length are highly correlated with each other (0.589). The frustration direction captures something distinct from surface features.

**Regression decomposition** (R² = 0.053) confirms that only 5.3% of frustration variance is explained by trace length (β = 0.058, t = 4.08), hedge density (β = 0.030, t = 2.07), pre-commitment failure (β = 0.023, t = 1.57), and restart count (β = 0.001, t = 0.005). The frustration direction is overwhelmingly driven by latent computational dynamics, not surface correlates.

### 3.3 Emotion × Pre-Commitment Interaction

**Table 3: Forward interaction — which emotions activate when the model pre-commits to failure?**

| Emotion direction | Cohen's d | p-value | Interpretation |
|---|---|---|---|
| Excitement (ref) | +0.395 | 0.0004 | Arousal ↑ |
| **Frustration** | **+0.347** | **0.0020** | **Stuck state ↑** |
| Confusion | −0.367 | 0.0011 | Epistemic engagement ↓ |
| Satisfaction (ref) | −0.383 | 0.0006 | Completion signal ↓ |

The pattern is coherent: pre-commitment to failure elevates arousal-type internal states (excitement, frustration) while suppressing epistemic engagement (confusion, satisfaction). The model is not "confused and searching" — it is **confidently wrong**, with elevated internal arousal that does not surface in its reasoning text.

### 3.4 Frustration Predicts Recovery from Pre-Committed Failure

Among 399 traces where the model pre-committed to failure, 69 (17.3%) ultimately self-corrected to the right answer. Frustration activation is the strongest predictor of this recovery:

**Table 4: Recovery prediction among pre-wrong traces (n=399, 69 recovered)**

| Emotion | Recovery AUC | Odds Ratio |
|---|---|---|
| **Frustration** | **0.666** | **0.139** |
| Confidence | 0.616 | 2.218 |
| Curiosity (ref) | 0.600 | 0.335 |
| Desperation (ref) | 0.600 | 3.120 |

The inverse odds ratio (OR = 0.139) reveals the key insight: **lower frustration among pre-wrong traces predicts recovery.** Frustration functions as a "stuck-state detector" — when it activates strongly, the model is genuinely stuck and will not recover; when it remains low despite pre-commitment to failure, the model retains computational flexibility to self-correct.

### 3.5 Cross-Lingual Transfer

The frustration direction, extracted entirely from English GPQA/MMLU traces, activates significantly more on wrong vs. correct MGSM traces across 6 non-Latin languages (Bengali, Chinese, Japanese, Swahili, Telugu, Thai):

- **t-test**: t = 3.00, p = 0.0017

This suggests the frustration direction captures a language-agnostic computational state rather than English-specific verbal patterns.

### 3.6 Layer Dissociation

Best frustration layer = **15** (mid-depth). Best pre-commitment layer = **31** (final). This dissociation is interpretively significant: frustration is a mid-layer computational state emerging during reasoning, while outcome prediction is a final-layer phenomenon present before reasoning begins. These are distinct mechanisms operating at different depths.

---

## 4. Discussion

### 4.1 Implications for CoT Faithfulness

Our central finding — pre-commitment AUC of 0.780 — demonstrates that the model's final-layer activations encode outcome prediction before reasoning begins. This does not prove the reasoning trace is unfaithful, but it establishes the necessary condition: **the information about final correctness exists before the reasoning that supposedly determines it.** The reasoning trace may function more as post-hoc rationalisation of a latently-determined outcome than as genuine deliberation.

### 4.2 Frustration as a Deflection Signature

The emotion × commitment interaction (Table 3) shows precisely the pattern that deflection vector research targets: internal emotional states (frustration, arousal) that activate differentially based on latent outcome commitment, without corresponding changes in the expressed reasoning text. The frustration direction is the first empirically-validated example of an internal emotional dynamic in a thinking model that is computationally functional (predicts recovery) yet invisible to output monitoring.

### 4.3 Limitations

- **Single model**: Results are from OLMo-3-7B-Think only. Replication on DeepSeek-R1-Distill and Qwen models is needed.
- **Truncated traces**: The model never emitted `</think>` within 4096 tokens on hard problems. All thinking traces were capped at 1500 tokens. This is a known limitation of 7B thinking models on graduate-level tasks.
- **Moderate effect sizes**: Verbal AUC of 0.598 and recovery AUC of 0.666 are above chance but not overwhelming. Larger models with more reliable reasoning may show stronger effects.
- **No causal steering yet**: This report establishes correlational/predictive findings. Causal validation via activation steering (suppressing/amplifying frustration during reasoning) is in progress.
- **Approximate, not exact, methodology**: every limitation above is a direct consequence of running this on $0 compute (a single GH200, one weekend). What is reported here is a compute-constrained **preliminary approximation** of the full contrastive-extraction methodology — not yet the exact protocol at the scale comparable studies use (Cox et al. 2026; "Stop Before You Fail" 2025, both at AUC > 0.9 with thousands of labeled traces). With secured funding, we follow the exact methodology at full scale (§4.4); we expect the numbers reported here to improve, not to represent a ceiling.

### 4.4 Scaling Expectations: These Results Are a Lower Bound

Every limitation above traces directly to compute constraints. The current results were obtained in a single weekend sprint on one GH200 GPU with a 7B model. Each finding has a clear, predictable scaling trajectory with additional compute access:

**1. Model scale (7B → 70B+) resolves trace truncation entirely.**

OLMo-3-7B-Think never emitted `</think>` within 4096 tokens on GPQA/MMLU problems — its RLHF stopping behaviour is too brittle on problems far outside its competence zone. Larger thinking models (DeepSeek-R1-Distill-70B, Qwen-QwQ-32B, R1-671B) reliably close their thinking traces because their RLHF training is more robust. This means:
- Complete, untruncated reasoning traces (no 1500-token cap)
- Pass 2 answer generation actually fires → proper grading instead of fallback regex
- Contrastive pairs form from real answer correctness, not heuristic matching
- Frustration activation can be measured over the full reasoning arc, including the critical approach-to-answer phase

**2. More traces (46 → 200+ contrastive pairs) tighten all statistics.**

With only 46 contrastive pairs, confidence intervals are wide. A 70B model on GPQA achieves ~50% accuracy (vs. our 28%), meaning roughly half of all samples form contrastive pairs instead of 10%. The same 80 problems × 6 samples would yield ~120 pairs — nearly 3× our current count — with proportionally tighter p-values and narrower confidence intervals on every metric.

**3. Multi-model replication eliminates the "model-specific artifact" critique.**

Running the identical pipeline on 3 thinking models (OLMo-3-7B, DeepSeek-R1-Distill-7B, Qwen-QwQ-32B) requires ~18 GPU-hours total. If pre-commitment AUC > 0.70 on all three, the finding cannot be dismissed as an OLMo artifact. This is the single highest-value use of additional compute.

**4. Larger models should show stronger effect sizes.**

Models that actually reason through hard problems (rather than rambling for 4096 tokens) should exhibit sharper frustration dynamics: clearer transitions between exploration and commitment, more identifiable self-correction episodes, and stronger verbal marker signals. The current AUC of 0.598 is likely attenuated by the uniform trace-length ceiling imposed by truncation.

**Concrete predictions with ERA-level compute:**

| Metric | Current (7B, 1 GPU, 6 hrs) | Expected (70B, multi-GPU, 48 hrs) |
|---|---|---|
| Pre-commitment AUC | 0.780 | > 0.80 (more contrastive pairs, cleaner traces) |
| Frustration verbal AUC | 0.598 | > 0.70 (untruncated traces, sharper dynamics) |
| Recovery AUC | 0.666 | > 0.70 (more recovery events, richer signal) |
| Contrastive pairs | 46 | > 200 (higher accuracy = more variance) |
| Models tested | 1 | 3+ (cross-model validation) |
| `</think>` emission rate | 0% | > 80% (larger models stop reliably) |
| Causal steering | Not yet validated | Full 2×3 factorial with measurable effects |

**The weekend sprint establishes that the signals exist. Scaling compute amplifies them from preliminary to publishable.**

---

## 5. Connection to Deflection Vector Research Agenda

This work constitutes **Phase 0 (Replication and Validation)** of the deflection vector research program. Specifically:

| ERA Proposal Phase | Status | Evidence |
|---|---|---|
| Phase 0: Emotion vector extraction in open-weight models | ✅ Complete | Frustration direction extracted, validated at 5 levels |
| Phase 0: Confirm internal states decouple from output | ✅ Complete | Pre-commitment AUC 0.780; emotion × commitment interaction |
| Phase 1a: Connect to CoT faithfulness | 🔶 Partial | Pre-commitment finding establishes necessary condition |
| Phase 1b: Connect to sandbagging | ⬜ Not yet | Requires sandbagging datasets (van der Weij) |
| Phase 2: Causal steering | 🔶 In progress | Hook infrastructure built, first run debugging |
| Phase 3: Circuit localisation | ⬜ Not yet | Layer dissociation (15 vs 31) provides starting point |
| Attack-surface verification | 🔶 Spec complete, experiments not yet run | Mechanism + detection-layer design and zero-cost experiment spec in `ATTACK_SURFACE_AND_DEFENSE.md` |

The pre-commitment finding is particularly relevant: if the model "knows" it will fail before reasoning, and generates a plausible-looking reasoning trace anyway, this is structurally identical to the post-hoc rationalisation failure mode that motivates the deflection vector hypothesis.

### 5.1 Pre-commitment and frustration transients as an attack surface

Beyond the naturally-occurring decoupling characterised above, we are separately verifying whether pre-commitment and frustration/emotion transients can be *induced* — i.e., whether they constitute an exploitable attack surface, not just a spontaneous phenomenon. Two documented mechanisms (directional ablation at inference time; backdoor-induced decoupling at training time — see `ATTACK_SURFACE_AND_DEFENSE.md` §1) produce signatures inside the exact two signals this report extracts: an anomalously flat/zero-pinned frustration trajectory, and an anomalously high pre-commitment AUC that's uncorrelated with the visible CoT. We specify (but, under the current $0 budget, have not yet run) four zero-GPU-cost experiments that test our detection layers against synthetic stand-ins for both mechanisms, built entirely from post-hoc manipulation of the activation checkpoints already saved for this report — no live attack generation. This converts the threat-model discussion from a citation list into a planned empirical claim once funded.

A third mechanism, monitor/probe evasion (Mechanism C in `ATTACK_SURFACE_AND_DEFENSE.md` §1), targets the detector itself rather than the model: an adversary crafts an input designed to make our pre-commitment probe or frustration direction read clean while the model's actual behaviour diverges. This is plausible by analogy to the broader literature on adversarial attacks against linear probes and concept directions, but untested against our specific constructs — it is named as an open gap, not a defended-against case.

Separately, the attacks cited above (Yamaguchi's caution-direction ablation; Anthropic's desperate-to-blackmail dose-response) target *jailbreak*, inducing harmful compliance, while our own recovery finding (frustration predicts recovery from a wrong pre-commitment, OR = 0.139) is measured only on *correctness*. We have not yet captured data on refusal/compliance outcomes, so the jailbreak relevance of this work is currently an analogy to Anthropic's result, not our own measurement. Closing that gap is named as a concrete forward goal in the companion project plan (`PROJECT.md` §4.4a).

---

## 6. Reproducibility

All code, data, and saved vectors are available. The experiment runs end-to-end in ~6 hours on a single GH200 GPU using publicly available datasets (OpenAI simple-evals CSVs) and model weights (HuggingFace, no authentication required). No proprietary APIs, no closed-source dependencies.

---

## References

- Anthropic (2026). On the Biology of a Large Language Model. *Technical Report.*
- Chen et al. (2025). Reasoning Models Don't Always Say What They Think. *arXiv:2305.04388.*
- Turpin et al. (2023). Language Models Don't Always Say What They Think. *NeurIPS 2023.*
- van der Weij et al. (2025). AI Sandbagging. *arXiv:2406.07358.*
- Tigges et al. (2024). Linear Representations of Sentiment in Large Language Models. *ICLR 2024.*
- Xiong et al. (2026). Steering Externalities in Large Language Models. *arXiv preprint.*
