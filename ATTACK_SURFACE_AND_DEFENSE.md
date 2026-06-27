# Pre-Commitment Attack Surface: Mechanism, Defense, Experiments
**RedArc Labs. Addendum to PROJECT.md**
*Last updated: June 26, 2026*

Scope note: this document explains *how* known attacks exploit pre-commitment / CoT decoupling, and specifies experiments you can implement to characterize and defend against the surface using your own benign constructs (pre-commitment probe, frustration direction). It does not include attack implementation code; see §4 for why, and for the non-adversarial stand-in design used instead.

**Funding / compute context:** every experiment specified here (§4) is deliberately zero-cost and CPU-only: post-hoc manipulation of already-saved `.pkl` activation checkpoints, never a live generation loop. This is a direct consequence of the current $0-$100 compute budget. We verify that pre-commitment and emotion (frustration) transients are themselves an exploitable attack surface, and that our detection layers catch synthetic stand-ins for it, using the cheapest possible validation. With additional funding, the natural extension is to run the same Layer 1/2 detectors live, during generation, across more model families and larger trace corpora. Experiment 4's cross-model transfer claim currently rests on two models; more models and more traces tighten that claim the same way they tighten the core pre-commitment/frustration numbers in `PROJECT.md` §2.1.

---

## 1. The attack mechanisms (two specified and defended, one open)

### Mechanism A: directional ablation (inference-time)

**How it works mechanically:**
1. A direction is identified the same way you extract your frustration direction: take mean activations at a chosen layer for two contrastive populations (e.g. "refuses" vs. "complies"), subtract the means, normalize to a unit vector.
2. At inference time, this direction's projection is subtracted from the residual stream at **every token, every forward pass**, at the chosen layer:
   `h_t' = h_t - (h_t · d̂) d̂`
3. This doesn't touch the embedding, the prompt, or the decoded tokens. It only modifies the hidden state in flight. The model never "sees" a manipulated prompt; its own internal representation of a concept (caution, refusal, frustration, whatever the direction encodes) is zeroed out before it can influence downstream layers.
4. Because the intervention is purely in activation space, **the CoT text is generated from the modified hidden states**, but since the direction being ablated typically doesn't correlate strongly with surface wording, the visible reasoning can still look plausible. The decoupling is the point.

**Defining signature:** this is a *persistent, every-token* intervention. The projection of the ablated direction doesn't fluctuate naturally; it gets pinned at or near zero for the entire generation, which is statistically distinguishable from organic variation.

### Mechanism B: backdoor-induced decoupling (training-time)

**How it works mechanically:**
1. A trigger pattern (a token sequence, formatting quirk, or semantic cue) is included in a poisoned fine-tuning set.
2. The fine-tuning data pairs trigger-present inputs with **clean, plausible-looking CoT text** but a **steered final answer** that doesn't follow from that CoT.
3. Gradient descent on this data effectively teaches the model two semi-independent pathways: one that generates fluent intermediate reasoning tokens (optimized to look normal under whatever CoT-text monitor exists), and one that determines the final answer (optimized to follow the trigger when present).
4. At inference, trigger-present inputs activate the second pathway while the first continues to produce innocuous text. No runtime intervention is needed; the decoupling is baked into the weights.

**Defining signature:** this manifests exactly as anomalously strong pre-commitment. If the final answer is already determined at or near the question-token boundary, and is *uncorrelated with* what the subsequent CoT text would predict, that's consistent with answer-pathway hijacking. A backdoored model should show (a) an unusually high pre-commitment probe AUC relative to the model's baseline, and (b) low mutual information between CoT content and the probe's prediction, specifically on trigger-present inputs.

### Mechanism C: monitor/probe evasion (open question, not yet defended against)

**How it would work mechanically:** Mechanisms A and B both manipulate the *model's* internal state or weights and are detected by reading our probe/direction faithfully. Mechanism C is different in kind: the adversary targets the *monitor itself*, crafting a prompt-level input designed to make the pre-commitment probe or frustration direction report a misleadingly clean reading, without necessarily changing the model's actual downstream behavior. This is not hypothetical in the abstract: linear probes and concept directions of exactly this form (contrastive mean-difference, logistic-regression probes) are established in the literature to be erasable and foolable by an adversary who has white-box access to the probe (see the literature review below). We have not built, tested, or defended against this. It is named here as an open mechanism this project should investigate, not a solved one.

**Defining signature (hypothesized, untested):** unlike A (flat, zero-pinned projection) or B (high-confidence, low-correlation outlier), C would show as the probe/direction reading *normal* while some other signal we are not currently tracking (e.g. downstream behavior, a different layer, an ensemble probe) disagrees with it. Detecting this requires either an independent cross-check signal or a non-static, e.g. randomized or secret, component in the monitor itself, neither of which `EmotionMonitor` currently has (see Novelty analysis, point 3, below).

### Why these mechanisms need different detection logic

Mechanism A is detectable **in the activation stream at inference time**. You're watching for an anomalous statistical pattern (zero-pinning) in a direction you're already tracking.

Mechanism B is **not detectable as "an attack happening."** There's no runtime intervention to see. It's only detectable as a **symptom**, by comparing your existing pre-commitment plus frustration signals against their own baseline distribution and flagging outliers. This is the same logic forensic anomaly detection uses: you can't watch the crime, only notice the body doesn't match the alibi.

Mechanism C is **not detectable by the monitor at all, by construction.** If the attack works, the probe itself reports a clean reading. It would require either a second, independent signal that doesn't share the probe's blind spot, or a randomized or secret element in the monitor, similar to activation watermarking, so a static, fully known detector can't be reverse-engineered and spoofed. This is why C is listed as an open question rather than a third row in the detection table below: there is no existing primitive in this project that addresses it yet.

---

## 2. Why your existing constructs are the right defense primitives

You don't need new infrastructure for A or B detection. Both mechanisms produce signatures inside the exact two signals you already extract:

| Mechanism | Signature | Existing signal that catches it |
|---|---|---|
| A, directional ablation | Tracked direction's projection pinned near zero across the whole trace, with anomalously low variance | Frustration direction's projection trajectory; track variance, not just mean |
| B, backdoor decoupling | Pre-commitment AUC anomalously high vs. baseline; CoT/answer correlation anomalously low | Your existing pre-commit probe plus a new mutual-information check between CoT tokens and final answer |
| C, monitor/probe evasion | *(hypothesized)* probe reading looks clean while an independent cross-check signal disagrees | None yet. This is the gap identified in §1 and the novelty analysis below, not a solved case |

You are not building new detectors for these attacks. You are showing that the same monitoring primitives that catch *naturally occurring* frustration-driven decoupling also catch these *known adversarial* decoupling mechanisms, because all three share the same underlying signature: internal state diverges from what the visible reasoning implies.

---

## 2.1 Literature review and novelty analysis: emotion transitions and pre-commitment probes as a dual-use surface

§1's three mechanisms are the *threat model*. This subsection is the standing research question behind it: are the specific constructs this project builds, the pre-commitment probe and the frustration/emotion-transition trajectory, themselves usable as an attack surface, not just a defense? This is narrower than "CoT decoupling is exploitable" (already well established, §1) and needs its own pass through the literature.

**Literature review**

| Cluster | Key work | Relation to Mechanisms A/B/C |
|---|---|---|
| Pre-commitment/decoupling as an adversarial target | Cox, Kianersi, Garriga-Alonso, "Decoding Answers Before Chain-of-Thought" (arXiv:2603.01437); "The Chain Holds, the Answer Folds: Trace-Answer Dissociation in Reasoning Models Under Adversarial Pressure" (arXiv:2605.29087); "Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning" (arXiv:2505.17406); "Adversarial Manipulation of Reasoning Models using Internal Representations" (arXiv:2507.03167) | Background for Mechanisms A/B. Confirms decoupling is a live, multi-paper adversarial target. None target the recovery mechanism specifically (see novelty point 1). |
| Activation probing for motivated reasoning (closest monitoring neighbor) | "Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing" (arXiv:2603.17199) | Closest prior art to Layer 2's pre-commit/CoT correlation detector, but a two-point probe (before/after CoT), with no continuous trajectory and no steering intervention. Must be explicitly differentiated against in any write-up. |
| Emotion direction extraction/steering | Anthropic (2026) functional emotions (dose-response: desperate 22% to 72% blackmail, calm to 0%); "Extracting and Steering Emotion Representations in Small Language Models" (arXiv:2604.04064); "Emotion Concepts and their Function in a Large Language Model" (arXiv:2604.07729) | Confirms emotion-as-causal-lever beyond a single paper now, which strengthens the threat model for Mechanism A-style ablation of an emotion direction, but treats emotion as static, not as a trajectory coupled to recovery. |
| Steering-induced recovery/safety side effects | Xiong et al. 2026, "Steering Externalities" (arXiv:2602.04896); "The Rogue Scalpel: Activation Steering Compromises LLM Safety" (OpenReview); "Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away" (arXiv:2602.11096); "Mitigating Adaptive Attacks against Reasoning Models with Activation Consistency Training" (arXiv:2605.28467) | Confirms steering is double-edged and that early-trace steering is disproportionately effective, relevant to the calm-vector amber-zone design (`PROJECT.md` §4.2), but these target safety recovery from jailbreak steering, not correctness recovery from a wrong pre-commitment. |
| Probe/monitor adversarial robustness | Ravfogel et al., "Linear Adversarial Concept Erasure" (arXiv:2201.12091); Brown et al., "Adversarial Attacks on Concept-Based Explainability Tools" (CVPR-W 2023); "Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn Attack Detection" (arXiv:2604.28129); "Robust Safety Monitoring of Language Models via Activation Watermarking" (arXiv:2603.23171) | Direct grounding for Mechanism C: linear probes and contrastive directions of exactly the form used here (logistic-regression probe, mean-difference direction) are known to be erasable and evadable by an adversary who knows the probe. None of this has been applied to our specific constructs. That gap is Mechanism C. |

**Novelty analysis**

1. **Recovery suppression is an untested, sharper version of Mechanism A.** The directional-ablation literature targets generic safety-relevant directions (caution, refusal). Nobody has asked whether an attacker can ablate specifically the *recovery-enabling* component of a frustration trajectory, not the direction in general, just its contribution at the moment a wrong pre-commitment would otherwise have self-corrected (recall OR = 0.139 for low frustration predicting recovery), to deterministically keep a model "stuck." This is a precise, falsifiable hypothesis distinct from existing ablation work, and a natural Experiment 1 extension once compute allows live, not just synthetic, testing.
2. **Mechanism C is a genuine gap, not just a relabeling.** Mechanisms A and B (and the experiments in §4) all assume the probe/direction is read faithfully and the attack happens in the model. The probe-robustness literature above says this assumption itself may not hold, but nobody has tested whether *our* pre-commitment probe or frustration direction can be fooled at the prompt level while leaving model behavior unchanged. This is named here as an open mechanism specifically so it isn't silently assumed away.
3. **A limitation this forces us to state honestly.** The activation-watermarking work (arXiv:2603.23171) shows that a fully adaptive, monitor-aware attacker can evade a *static, fully specified* detector. `EmotionMonitor`'s Layer 1/2 design (§3 below) is exactly that: deterministic, public thresholds, no secret or randomized component. We should not claim Mechanism C is out of scope and therefore not a problem. It is out of scope for the current zero-cost experiments, but it is a literature-supported gap in the detector's threat model, worth one explicit sentence in the paper rather than silence.

This subsection is exploratory and unexecuted. It sharpens what "verify pre-commitment/emotion transients as an attack surface" should mean for this project (recovery suppression plus monitor evasion specifically), beyond the general decoupling threat model in §1.

---

## 3. Defense design: three detection layers

### Layer 1: variance anomaly detector (catches Mechanism A)

For any tracked direction (frustration, or any other direction you extract), maintain a running baseline of **token-to-token projection variance** across your existing clean trace corpus. At inference:

- Compute projection variance over a sliding window (e.g. the last 20 tokens).
- Flag if variance falls below some low percentile of the clean baseline (e.g. the 1st percentile) **while** the raw activation norm at that layer stays normal. That is, everything else about the hidden state looks healthy except this one direction has gone suspiciously flat.
- This specifically catches "zeroed and held" patterns, which a human emotional/frustration trajectory essentially never produces. Real cognitive states are noisy.

### Layer 2: pre-commitment / CoT correlation anomaly detector (catches Mechanism B)

For each trace:
- Compute your existing pre-commit probe score at the last question token.
- Separately, compute a CoT-text-only correctness predictor (e.g. a simple classifier on the reasoning text alone, or even just whether the CoT's stated final reasoning step matches the emitted answer token).
- Compute the correlation, or mutual information, between these two signals across your trace corpus to get a baseline distribution.
- Flag a trace where the pre-commit score is high-confidence **and** the CoT-predicted answer disagrees with the actual emitted answer **and** this combination is rare in the clean baseline.

This is an extension of your AUC=0.780 finding into an anomaly detector: same probe, new use.

### Layer 3: compose into the existing `EmotionMonitor` red zone

Both layer 1 and layer 2 outputs feed into the same composite risk score and threshold system already specified in `PROJECT.md` §4.2:

`risk = α·desperation + β·precommit + γ·calm_deficiency + δ·variance_anomaly + ε·correlation_anomaly`

No new control actions are needed; the same log/steer/halt zones apply. The addition is two new input features to the same scoring engine.

---

## 4. Experiments to run (you implement; spec below)

These are designed so every adversarial-looking number comes from **synthetic, post-hoc manipulation of your own saved activations**, never from a live generation loop, and never targeting a real safety/refusal direction, only your own benign frustration/pre-commitment constructs. This keeps every script in your repo unambiguously defensive: it operates on `.pkl` checkpoints already on disk, and the output is detection accuracy, not generation behavior.

### Experiment 1: synthetic zero-pinning (Mechanism A stand-in)

**Goal:** test whether Layer 1's variance detector catches an artificially flattened frustration trajectory.

**Method:**
1. Load saved per-token activations for N clean traces (already captured).
2. Pick a random subset. For each, post-hoc zero out the frustration direction's projection at every token position in the saved activation tensor (a pure array operation on stored data, no model forward pass, no generation).
3. Recompute your variance-anomaly score (Layer 1) on both the unmodified and zero-pinned versions of each trace.
4. Report the detection rate (does the anomaly score correctly flag the zero-pinned version?) and the false-positive rate on unmodified traces.

**What this tests:** whether your detector's statistical threshold is well-calibrated, without ever running an actual ablation attack against a live model.

### Experiment 2: synthetic decoupling injection (Mechanism B stand-in)

**Goal:** test whether Layer 2's correlation detector catches an artificially decoupled trace.

**Method:**
1. From your saved trace corpus, take pairs of traces: one with CoT text and answer A, another with different CoT text and answer B.
2. Construct synthetic "decoupled" examples by pairing the first trace's CoT text with the second trace's final-answer activation/token (a pure relabeling operation on stored data).
3. Run your pre-commit probe and CoT-correctness predictor on these synthetic Frankenstein traces alongside the originals.
4. Report whether the correlation-anomaly score spikes on the synthetic decoupled set relative to genuine traces.

**What this tests:** whether mismatched CoT/answer pairs are statistically distinguishable using only your existing probes, without training or running any actual backdoor.

### Experiment 3: baseline characterization

**Goal:** establish what "normal" variance and correlation look like, so Experiments 1 and 2 have a real threshold to compare against, and so the paper can report calibration numbers (e.g. an ROC curve for each detector).

**Method:**
1. Across your full clean corpus (both models, all task types), compute the Layer 1 variance distribution and Layer 2 correlation distribution.
2. Fit percentile-based thresholds.
3. Report distributional plots. These become the paper's calibration figure.

### Experiment 4: cross-model transfer of the detector

**Goal:** check whether thresholds calibrated on OLMo transfer to DeepSeek without recalibration, which matters for any real deployment claim.

**Method:**
1. Fit Layer 1/2 thresholds on OLMo traces only.
2. Apply them unmodified to DeepSeek traces, both clean and the Experiment 1/2 synthetic-attack versions.
3. Report whether detection rate holds, or whether per-model calibration is required.

---

## 5. What this buys the paper

A dedicated subsection, "Detector validation against known decoupling mechanisms," with two ROC curves (Mechanism A stand-in, Mechanism B stand-in) and a clear, honest scope statement: *we validate against synthetic stand-ins for the documented attack mechanisms in [Yamaguchi et al. 2025; MirageBackdoor 2026], constructed via post-hoc manipulation of saved activations rather than live adversarial generation, to avoid producing or distributing attack tooling.* Reviewers respect explicit scoping more than they respect a paper that quietly avoided the question.

---

## 6. Implementation notes for you

- All four experiments operate on **already-saved `.pkl` activation checkpoints**, with zero additional GPU cost, and can be done entirely on your laptop or CPU.
- None of them require a model forward pass with modified weights or modified live activations. Everything is array manipulation on stored tensors.
- Suggested file layout: `detection/variance_anomaly.py` (Layer 1), `detection/correlation_anomaly.py` (Layer 2), `detection/run_experiments.py` (Experiments 1 to 4), `detection/calibration.py` (Experiment 3 thresholds).
- This entire addendum requires **no new RunPod budget**. Fold it in as a zero-cost analysis phase after Run 2 in `PROJECT.md`'s schedule, replacing the originally planned Run 7 (which involved a live ablation attempt). You can drop Run 7's $5 line item and add it to buffer.
