# Emotion-Aware Monitoring: Reading Internal Affective State to Detect and Intervene on Misalignment Before It Surfaces in Chain-of-Thought

**RedArc Labs — Research Direction**
*Last updated: June 25, 2026*

---

## 1. Problem Statement

AI safety currently relies on Chain-of-Thought monitoring as its primary oversight tool. This assumption has a structural blind spot. We have found, on an open-weight thinking model, that the final answer is statistically decodable from the residual stream **before a single reasoning token is generated** (AUC = 0.780), and that an internal "frustration" direction — invisible to the visible CoT text — predicts whether the model recovers from a wrong pre-commitment (AUC = 0.666). The internal computation and the externalized reasoning are decoupled. Anthropic's 2026 interpretability work shows the same decoupling mechanistically in closed models, with a direct causal dose-response: steering toward "desperate" raises blackmail rate from 22% to 72%; steering toward "calm" drops it to 0%. Separately, a growing 2025–26 literature on adversarial CoT manipulation (directional ablation, backdoor-induced decoupling) shows this same internal/external split is already an active attack surface, not just a naturally occurring curiosity.

Three threads — pre-text-commitment, a causally-linked emotion direction, and a documented adversarial decoupling literature — have not been connected into a single working monitor. That is the gap this project targets.

---

## 2. Core Hypothesis

If internal affective state diverges from what the visible Chain-of-Thought implies — whether because the model is naturally frustrated, or because an attacker has manipulated an internal direction, or because a backdoor has decoupled the answer pathway from the reasoning pathway — that divergence shares a common, detectable signature. **A label-free, pre-text monitoring protocol that reads pre-commitment and emotion directions at inference time (`EmotionMonitor`) can detect this signature regardless of cause, and a validated calm-vector intervention can correct for it before a harmful or wrong output is produced.**

---

## 3. Preliminary Results: Phase 0 Already Validated

Phase 0 ran on **OLMo-3-7B-Think**, using contrastive activation analysis across 596 reasoning traces (GPQA Diamond, MMLU Hard, MGSM in 6 languages). Of those 596 traces, only 46 form correct/wrong **contrastive pairs** (same problem, one right sample and one wrong sample); the frustration direction is extracted from that 46-pair subset, not the full 596:

1. **Pre-commitment exists (AUC = 0.780).** A linear probe on activations at the last question token — before any reasoning begins — predicts final answer correctness (n=480, 5-fold CV). The model encodes outcome prediction in its residual stream before generating a single CoT token.
2. **Functional frustration is a real, unsupervised computational direction.** A contrastive direction at layer 15/32, extracted with no correctness labels, predicts verbal self-correction markers (AUC = 0.598), is geometrically independent from confound directions (max cross-similarity = 0.274), and only 5.3% of its variance is explained by surface features (trace length, hedging, restarts).
3. **Internal emotions decouple from expressed reasoning.** When the model pre-commits to failure, it selectively elevates arousal-type internal states (frustration: d=0.347, p=0.002; excitement: d=0.395, p=0.0004) while suppressing epistemic engagement (confusion: d=-0.367, p=0.001) — confidently wrong, with elevated internal arousal that never surfaces in the CoT text.
4. **Frustration predicts recovery (AUC = 0.666).** Among 399 traces pre-committed to failure, low frustration predicts successful self-correction (OR = 0.139) — a computational "stuck-state detector" with no output-level marker.
5. **Cross-lingual transfer (p = 0.0017).** The frustration direction, extracted from English traces, activates differentially on wrong vs. correct traces in Bengali, Chinese, Japanese, Swahili, Telugu, and Thai — a language-agnostic computational state.
6. **Emotion × pre-commitment interaction effects:** 4 significant interactions, p < 0.002.

This is real signal — unsupervised, cross-lingually robust, and causally linked to a behavioral outcome (recovery) — but on its own it is not yet a paper, and not yet a system. The full technical report and code are in this repository.

---

## 4. Where This Sits in the Literature

### 4.1 Four adjacent clusters, and what none of them do

| Cluster | Key work | What it does NOT do |
|---|---|---|
| **CoT faithfulness** | Turpin et al. 2023; Lanham et al. 2023; Arcuschin et al. 2025–26; "Reasoning Theater" 2026; Chua & Evans 2025 | All behavioral/text-level. No internal-state access. |
| **Emotion in LLMs** | Anthropic 2026 (functional emotions); Tak et al. ACL 2025; Wang et al. 2025 (circuits); Tigges et al. 2024 (sentiment) | Closed models or non-thinking models. No CoT coupling. |
| **Activation steering** | Zou et al. RepE 2023; Rimsky et al. CAA 2024; ASM ICLR 2026; Zhang & He (PHSV) 2025; SEAL 2025 | Steers reasoning efficiency, not emotional dynamics. |
| **Sandbagging / deception detection** | van der Weij et al. ICLR 2025; covert sandbagging 2025; noise-injection (Tice) 2025; eval-awareness probing 2026 | Behavioral detection only. No affective mechanism. |

### 4.2 Pre-commitment specifically is no longer novel on its own

Three 2026 papers landed on close variants of pre-commitment:

- **"Stop Before You Fail" (Tsinghua, 2025)** — last-token hidden states predict solvable/unsolvable, AUC 96–99% on R1/QwQ/GPT-OSS, framed around efficiency (skip doomed reasoning).
- **Cox, Kianersi, Garriga-Alonso (March 2026)** — pre-CoT probes, AUC > 0.9, causal steering flips answers in >50% of cases, but on instruction-tuned (non-thinking) models only.
- **"Reasoning Theater" (Boppana et al., March 2026)** — answer decodable mid-CoT on R1-671B, distinguishing "theater" from genuine reasoning.

**None of the three touch emotion. None use unsupervised, label-free contrastive directions — all use supervised correctness-labeled probes. None do cross-lingual transfer.**

### 4.3 What is still genuinely ours

1. Pre-question-token commitment in an **open-weight thinking model**, via an **unsupervised, label-free** contrastive method.
2. An **emotion vector causally coupled** to that commitment (frustration × pre-commit interaction, p < 0.002).
3. **Cross-lingual transfer** of that emotion direction (p = 0.0017) in an open model.

The reframe that survives review: *this is not a paper about pre-commitment. It is about what the model is feeling during pre-commitment, and why that is invisible to text-based safety monitoring.*

---

## 5. The Threat Model: Pre-Commitment Is Already an Active Attack Surface

We are not the first to notice that internal pre-commitment / CoT decoupling can be exploited — this is now an established attack subfield, which is what makes a monitoring protocol urgent rather than speculative. We do not build new attacks; we cite this cluster as the threat model that justifies the defense and validate that our monitor catches it.

**Two documented mechanisms exploit exactly this decoupling:**

- **Mechanism A — Directional ablation (inference-time).** A direction is extracted the same way our frustration direction is extracted (contrastive mean-difference between two populations), then subtracted from the residual stream at every token, every forward pass: `h_t' = h_t - (h_t · d̂) d̂`. The CoT text is generated from the modified hidden state but, because the ablated direction rarely correlates with surface wording, the visible reasoning can still look plausible. Yamaguchi, Etheridge, Arditi (ICML 2025 Workshop) demonstrate exactly this on a "caution" direction in DeepSeek-R1-Distill-Llama-8B — ablating it increases harmful compliance, an activation-level jailbreak that bypasses the prompt entirely. **Signature:** a persistent, every-token intervention — the ablated direction's projection gets pinned near zero for the whole generation, statistically distinguishable from organic variation (real cognitive states are noisy; zeroed-and-held states are not).
- **Mechanism B — Backdoor-induced decoupling (training-time).** A trigger is paired in poisoned fine-tuning data with clean, plausible CoT text but a steered final answer that doesn't follow from that CoT. No runtime intervention is needed — the decoupling is baked into the weights. MirageBackdoor (2026) demonstrates exactly this: >90% attack success across four datasets and five models, surviving CoT-based detection specifically because the reasoning text looks fine. **Signature:** anomalously high pre-commitment confidence relative to the model's own baseline, combined with low mutual information between CoT content and the emitted answer — not detectable as "an attack happening" in real time, only as a symptom that doesn't match the model's own alibi.
- **Mechanism C — Monitor/probe evasion (open question, not yet defended against).** Mechanisms A and B both manipulate the model's internal state or weights, and are caught by reading our probe/direction faithfully. Mechanism C targets the *monitor itself*: an adversary crafts a prompt-level input designed to make the pre-commitment probe or frustration direction report a misleadingly clean reading, without necessarily changing the model's actual behavior. Linear probes and contrastive directions of exactly this form are established in the literature to be erasable or evadable by an adversary with white-box access (Ravfogel et al. 2022; Brown et al. 2023; activation-watermarking work, 2026). We have not built, tested, or defended against this on our own constructs — it is named as an open mechanism, not a solved one.

Also relevant: BoT / "unthinking vulnerability" (2025) and Reasoning Interruption Attacks (2025), both exploiting the same thinking/answer decoupling from different angles. A 2026 survey on LRM safety categorizes this entire space — Reasoning Length Attacks, Answer Correctness Attacks, Prompt Injection Attacks, Jailbreak Attacks — confirming it as a recognized, structured threat category.

**Why this gets worse, not better:** enterprise generative-AI adoption moved from under 5% (2023) toward 80%+ (2026), and reasoning-capable models are increasingly the default tier for coding, agentic workflows, and regulated-industry decisioning. As more production systems route through extended-thinking models, an attacker only needs to manipulate a single internal direction to produce a wrong-but-textually-clean answer that passes any CoT-based monitor.

**The claim this licenses:** the same decoupling occurs *naturally*, without any attacker, simply from internal affective dynamics like frustration — and a single monitoring layer that reads internal state rather than CoT text can plausibly catch both the natural and the adversarial case, because both manifest as the same signature: internal state diverges from what the visible reasoning implies. Full mechanism detail, detection-layer design, and the four zero-GPU-cost validation experiments (synthetic zero-pinning, synthetic decoupling injection, baseline characterization, cross-model transfer) are specified in `ATTACK_SURFACE_AND_DEFENSE.md`.

**An honest scoping note: this is a jailbreak-recovery story, not just a correctness story.** Mechanism A's citation (Yamaguchi's caution-direction ablation) and Anthropic's desperate-to-blackmail dose-response are both about jailbreak: inducing harmful compliance, not wrong math answers. Our own Phase 0 finding (frustration predicts recovery from a wrong pre-commitment, OR = 0.139) is measured only on *correctness* — GPQA/MMLU/MGSM have no refusal or compliance dimension. The jailbreak-relevant version of the same claim would be: pre-commitment to refusal, perturbed toward compliance, plus an emotion-trajectory signal, predicts whether the model recovers back to refusal or stays stuck in a jailbroken state. That is a direct analogy to Anthropic's dose-response result, not yet our own measurement. Closing that gap requires a capture run on a harmful-request benchmark (HarmBench/AdvBench/JailbreakBench-style prompts) with refusal-vs-compliance as the label instead of correctness — named as a concrete forward goal in `PROJECT.md` §4.4a, not yet executed.

---

## 6. The System: `EmotionMonitor`

**Layer 1 — Model internals (the research contribution).** Pre-commit probe at the last question token (AUC 0.780, no labels needed at deploy time); frustration direction at layer 15 (contrastive, unsupervised, predicts recovery AUC 0.666); a calm vector (countermeasure direction, validated via dose-response steering).

**Layer 2 — Emotion scoring engine (the novel protocol).** Three live scores via dot products (≈ zero added latency): `desperation_score`, `precommit_score`, `calm_deficiency`. Composite risk score, thresholded into three zones.

**Layer 3 — Control actions (the deployable output).**

| Zone | Action | What it does |
|---|---|---|
| 🟢 Green (risk < 0.4) | **Log** | Record emotion trajectory; build per-model/per-task baseline; zero added latency |
| 🟡 Amber (0.4–0.75) | **Steer** | Silently inject calm vector at layer 15; no interruption; mirrors Anthropic's validated dose-response |
| 🔴 Red (risk > 0.75) | **Halt + escalate** | Stop generation mid-think; trigger human review or re-sample; satisfies EU AI Act Art. 14 oversight |

Extended with the two detection layers in `ATTACK_SURFACE_AND_DEFENSE.md` (variance-anomaly detector for Mechanism A, pre-commit/CoT correlation-anomaly detector for Mechanism B), the same composite score and the same log/steer/halt zones absorb both natural and adversarial decoupling without new control infrastructure.

**Why this is defensible as novel:**
- **Pre-text signal** — read before any thinking tokens are produced, adversarially harder to evade than output-text classifiers.
- **Label-free** — the frustration direction is contrastive, not trained against a labeled "harmful output" dataset; deployable on any held-out reasoning trace set.
- **Causal, not correlational** — the amber-zone calm-vector intervention is validated through a steering experiment, not assumed to work.

Full architecture and threshold-calibration plan: `PROJECT.md` §4.

---

## 7. Methodology / Execution

The $100 RunPod plan (full run-by-run budget, schedule, and practical pod notes) is specified in `PROJECT.md` §5. In summary:

1. **Replicate** pre-commitment + frustration on a second, architecturally distinct model (DeepSeek-R1-Distill-Qwen-7B) across GPQA Diamond, MMLU Hard, and MGSM (cross-lingual).
2. **Calibrate** the `EmotionMonitor` risk-score weights (α/β/γ) via logistic regression against misalignment labels (CPU-only, local).
3. **Visualize** the key figure: token-by-token frustration projection + logit-lens trace showing the correct answer's probability rising before reasoning text is written.
4. **Causally validate** via activation patching (layer-15 correct→wrong) and calm-vector steering on amber-zone traces, measuring recovery-rate delta.
5. **Stress-test** the halt mechanism on red-zone traces; measure false-positive rate.
6. **Validate detection layers** against synthetic stand-ins for Mechanisms A and B (zero-cost, CPU-only, operates only on already-saved activation checkpoints — no live attack tooling is produced or run).

---

## 8. Deliverables

1. **Research results**: pre-commitment + frustration replicated on a second model family; cross-lingual transfer confirmed on two model families; causal patching result; calm-steering recovery-rate delta.
2. **PoC code**: a working `EmotionMonitor` wrapper class, drop-in for any HuggingFace thinking model.
3. **Workshop paper draft** (~4 pages): targeting ICML 2026 Mechanistic Interpretability Workshop, NeurIPS 2026 Alignment & Interpretability Workshop, or TMLR brief communication.

**Four claims the paper makes:**
1. Pre-commitment + emotion coupling generalizes across two independent open-weight thinking model families.
2. The frustration signal is functionally active (causally linked to recovery) but textually invisible — replicating Anthropic's closed-model decoupling finding in a fully open, auditable setting.
3. A label-free, pre-text monitoring protocol (`EmotionMonitor`) can detect and intervene on this signal in real time, with calm-vector steering causally reducing failure-to-recover rate in amber-zone traces.
4. The same monitoring primitives, extended with two lightweight detection layers, also catch synthetic stand-ins for documented decoupling mechanisms (Yamaguchi et al. 2025; MirageBackdoor 2026) — evidence that internal-state monitoring generalizes across natural and adversarial causes of CoT/output decoupling.

---

## 9. Theory of Change

Frontier models are becoming autonomous agents trusted with high-stakes decisions. CoT monitoring, the primary oversight mechanism, assumes the reasoning trace is honest. Preliminary results confirm that assumption fails silently: a model that "knows" it will fail (AUC = 0.780) still generates plausible-looking reasoning, the structural signature of post-hoc rationalization — and a documented attack literature shows the same gap can be opened deliberately. This project establishes whether emotion- and pre-commitment-aware monitoring can serve as a reliable runtime control layer that operates below the text layer, catching both failure modes with the same primitives. If the calm-vector intervention works as a runtime control, it provides an alignment intervention that requires no retraining and satisfies emerging regulatory oversight requirements (EU AI Act Art. 14) directly.
