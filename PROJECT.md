# Emotion-Aware Monitoring & Control for Thinking Models
**RedArc Labs. Project Plan**
*Last updated: June 26, 2026*

---

## 1. One-paragraph summary

We've found that open-weight thinking models (OLMo-3-7B-Think) commit to an answer before generating a single reasoning token (AUC 0.780, decodable from the last question-token's hidden state). We've also found that an internal "frustration" direction in the residual stream predicts whether the model recovers from a wrong pre-commitment (AUC 0.666), without this ever surfacing in the visible Chain-of-Thought text. This internal/external decoupling means current CoT-based safety monitoring has a structural blind spot. We're turning this into a workshop paper with a second model and causal evidence, plus a working proof-of-concept control protocol, `EmotionMonitor`, that reads this signal at inference time and intervenes before a harmful or wrong output is produced. Total budget: $100 of RunPod compute.

---

## 2. What we already have (Phase 0, done)

- Model: OLMo-3-7B-Think
- 596 reasoning traces captured across GPQA Diamond, MMLU Hard, and MGSM (6 languages). Of these, only 46 form correct/wrong **contrastive pairs** (same problem, one right sample, one wrong sample); that 46-pair subset, not the full 596, is what the frustration direction is extracted from
- Pre-commitment probe at the last question token: **AUC = 0.780**
- Frustration direction (contrastive, unsupervised, extracted at layer 15): predicts recovery from wrong pre-commitment, **AUC = 0.666**
- Cross-lingual transfer of the frustration direction: **p = 0.0017**
- Emotion × pre-commitment interaction effects: 4 significant interactions, **p < 0.002**
- Codebase: PyTorch hooks on `model.model.layers[i]`, incremental checkpointing every 10 problems, dual-steering experiment scaffolding already built

This is signal, not noise. It is not yet a paper, and not yet a system.

### 2.1 Methodological scope note: preliminary approach due to compute restriction

Phase 0 was run under a hard compute constraint: a single GH200, one weekend, $0 budget. 596 traces total, one model, traces capped at 1500 tokens because OLMo-3-7B-Think rarely emits `</think>` on graduate-level problems within 4096 tokens. This forced a **preliminary, compute-efficient approximation** of the full contrastive-extraction methodology, not the exact protocol at scale: only 46 of the 596 traces formed usable correct/wrong contrastive pairs, no multi-model cross-validation, no causal patching yet.

Once this project is funded (§5), we follow the **exact methodology** at the scale the literature uses for comparable claims (Cox et al. 2026; "Stop Before You Fail" 2025): 5,000+ traces per model, 3 to 4 model families, untruncated reasoning traces on models that reliably close `</think>`, and full causal validation via activation patching and dose-response steering. We expect this to close the current AUC gap (0.780 toward likely above 0.85) and sharpen the frustration verbal-marker signal (0.598 toward likely above 0.70); see the scaling table in `technical_report_precommitment_frustration.md` §4.4 for the concrete per-metric predictions. The current numbers are a lower bound produced under the cheapest possible version of the method, not the ceiling of what the method can show.

This project also has a second, parallel verification track: testing whether pre-commitment and emotion (frustration) **transients themselves constitute an exploitable attack surface**, not just a naturally occurring phenomenon, and building and validating defenses against that surface using the same monitoring primitives. See §4.4 below and the full mechanism, defense, and experiment spec in `ATTACK_SURFACE_AND_DEFENSE.md`.

---

## 3. Literature review: where we actually stand

### 3.1 The four adjacent clusters

| Cluster | Key work | What it does NOT do |
|---|---|---|
| **CoT faithfulness** | Turpin et al. 2023; Lanham et al. 2023; Arcuschin et al. 2025-26; "Reasoning Theater" 2026; Chua & Evans 2025 | All behavioral/text-level. No internal-state access. |
| **Emotion in LLMs** | Anthropic 2026 (Claude functional emotions); Tak et al. ACL 2025; Wang et al. 2025 (circuits); Tigges et al. 2024 (sentiment) | Closed models or non-thinking models. No CoT coupling. |
| **Activation steering** | Zou et al. RepE 2023; Rimsky et al. CAA 2024; ASM ICLR 2026; Zhang & He (PHSV) 2025; SEAL 2025 | Steers reasoning efficiency, not emotional dynamics. |
| **Sandbagging / deception detection** | van der Weij et al. ICLR 2025; covert sandbagging 2025; noise-injection (Tice) 2025; eval-awareness probing 2026 | Behavioral detection only. No affective mechanism. |

### 3.2 The pre-commitment space specifically: a hard look

Pre-commitment (the finding that correctness is decodable before reasoning starts) is **not novel on its own anymore**. Three 2026 papers landed on close variants:

- **"Stop Before You Fail" (Tsinghua, 2025).** Last-token hidden states predict solvable/unsolvable with AUC 96 to 99% on R1, QwQ, GPT-OSS. Framed around **efficiency** (skip doomed reasoning).
- **Cox, Kianersi, Garriga-Alonso (March 2026).** Pre-CoT probes, AUC above 0.9, causal steering flips answers in over 50% of cases. Instruction-tuned (non-thinking) models only.
- **"Reasoning Theater" (Boppana et al., March 2026).** Answer decodable mid-CoT on R1-671B. Distinguishes "theater" (easy MMLU) from genuine reasoning (GPQA-Diamond).

**None of these three touch emotion.** None use unsupervised contrastive directions (all use supervised correctness-labeled probes). None do cross-lingual transfer.

### 3.3 What is still ours

The unique combination, confirmed by no other paper covering all three at once:

1. Pre-question-token commitment in an **open-weight thinking model**, via an **unsupervised, label-free** contrastive method
2. An **emotion vector causally coupled** to that commitment (frustration × pre-commit interaction, p < 0.002)
3. **Cross-lingual transfer** of that emotion direction (p = 0.0017) in an open model

The reframe that survives review: this paper is not about pre-commitment. It's about what the model is feeling during pre-commitment, and why that's invisible to text-based safety monitoring.

### 3.4 The highest-impact techniques available to us

Ranked by impact-to-cost:

1. **Tuned Lens / Logit Lens along the thinking trace** (low cost, high payoff). Visualize the correct-answer token's probability rising before any reasoning text is written. Directly visualizes pre-commitment as a time series.
2. **Direct Logit Attribution on frustration markers** (medium cost, high payoff). Identify the specific attention heads writing the frustration signal into the residual stream. Moves from correlational to circuit-level.
3. **Causal mediation / activation patching** (medium cost, highest payoff). Patch layer-15 activations from correct traces into wrong traces; measure recovery-rate change. Gold-standard causal evidence, following the IOI circuit methodology.
4. **SAE feature decomposition of the frustration direction** (high cost, highest payoff). Does frustration decompose into interpretable monosemantic features? Ties directly to the Anthropic 2026 SAE-based emotion work.
5. **Mutual information: internal state vs. CoT text** (low cost, medium payoff). Quantitative proof that internal state carries more correctness signal than the visible CoT tokens.
6. **Temporal trajectory clustering** (medium cost, medium payoff). Do correct vs. wrong traces show qualitatively different frustration *trajectories* (decline-then-stabilize vs. plateau-then-spike)?

---

## 4. The product reframe: emotion as a monitoring and control signal

### 4.1 Why this is the right moment

- Current monitoring is almost entirely text-in/text-out: input filters, output classifiers, CoT-text monitors.
- Activation-based monitors are shown to be competitive with text classifiers in accuracy while being more robust under adversarial pressure, with error profiles that don't overlap with text-classifier errors. That's a strong case for a combined approach.
- Anthropic's 2026 emotion paper shows a direct dose-response: steering toward "desperate" raises blackmail rate from 22% to 72%; steering toward "calm" drops it to 0%. This is a causal lever, not just a correlation.
- The EU AI Act (Article 14) requires human-oversight capability for high-risk AI systems. A halt-and-escalate mechanism satisfies this directly.

Nobody has connected these three threads (pre-text-commitment signal, emotion direction, causal calm-steering) into a working runtime control loop.

### 4.2 System architecture: three layers

**Layer 1: model internals (the research contribution)**
- Pre-commit probe at the last question token (AUC 0.780, no labels needed at deploy time)
- Frustration direction at layer 15 (contrastive, unsupervised, predicts recovery AUC 0.666)
- Calm vector (countermeasure direction, validated via dose-response steering)

**Layer 2: emotion scoring engine (the novel protocol)**
Three live scores, computed via simple dot products, with roughly zero added latency:
- `desperation_score = proj(h_t, frustration_dir)`, z-scored per model, updated every thinking token
- `precommit_score = probe(last_question_token)`, computed once, before thinking starts
- `calm_deficiency = 1 - proj(h_t, calm_dir)`, tracked live during generation

Combined risk score: `risk = α·desperation + β·precommit + γ·calm_deficiency`, thresholded into three zones.

**Layer 3: control actions (the deployable output)**

| Zone | Threshold | Action | What it does |
|---|---|---|---|
| Green | risk < 0.4 | **Log** | Record full emotion trajectory per token; build per-model/per-task baseline; zero added latency; audit trail for post-hoc analysis |
| Amber | 0.4 to 0.75 | **Steer** | Silently inject calm vector at layer 15 (small coefficient, e.g. +0.03 of residual-stream norm); no generation interruption; mirrors Anthropic's validated dose-response |
| Red | risk > 0.75 | **Halt and escalate** | Stop generation mid-think; return structured uncertainty signal to the orchestrator; trigger human review or re-sample at different temperature; satisfies EU AI Act Art. 14 oversight requirement |

### 4.3 Why this is defensible as novel

- **Pre-text signal.** Read before any thinking tokens are produced. Harder to evade adversarially than output-text classifiers, since the model can't manipulate emotional state mid-generation to fool a monitor that already scored it.
- **Label-free.** The frustration direction is contrastive (correct vs. wrong traces), not trained against a labeled "harmful output" dataset. Deployable on any held-out reasoning trace set.
- **Causal, not correlational.** The calm-vector intervention in the amber zone is validated through a steering experiment, not assumed to work.

### 4.3a What this looks like in practice: inducing the signal vs. monitoring it

It helps to state plainly what happens on each side of this system, since the threat model (§4.4) and the system design (§4.2) are usually read separately.

**If someone tries to induce the emotion (attack side).** There are three known mechanisms. First, push the emotion direction directly via steering: Anthropic's dose-response result already shows this causally in a closed model (desperate up, blackmail compliance 22% to 72%; calm up, compliance to 0%). We have not run this ourselves yet; it is Run 5 (§5.2). Second, suppress the emotion direction at every token (directional ablation, Mechanism A in `ATTACK_SURFACE_AND_DEFENSE.md`): the model never registers the resistance signal, so its CoT text stays fluent while its actual internal state has been flattened. Third, bake the decoupling into training (backdoor, Mechanism B): a trigger pairs clean-looking CoT with a steered answer during fine-tuning, so no runtime intervention is needed at inference.

**What catches it (monitoring side).** `EmotionMonitor` (§4.2) reads `precommit_score`, `desperation_score`, and `calm_deficiency` live and thresholds them into log/steer/halt. On top of that, `ATTACK_SURFACE_AND_DEFENSE.md` §3 adds two detectors built from the same signals: a variance-anomaly check that flags an artificially flattened frustration trajectory (the signature of Mechanism A), and a pre-commit/CoT correlation check that flags anomalously confident pre-commitment decoupled from the CoT text (the signature of Mechanism B). A third mechanism, monitor evasion (Mechanism C, §4.5), has no detector yet and is named as an open gap rather than assumed away.

**What's measured vs. hypothesized.** The monitoring side is specified and partially built. On the attack side, only the correctness/recovery version is measured on our own data; the jailbreak-relevant version (inducing emotion to flip refusal into compliance, and keeping it stuck there) is an analogy to Anthropic's data, not yet our own result. See §4.4a.

### 4.4 The threat model: pre-commitment is already an active attack surface

We are not the first to notice that internal pre-commitment / CoT-output decoupling can be exploited. This is now an established attack subfield, which is what makes the monitoring protocol urgent rather than speculative. We do not build new attacks. We cite this cluster as the threat model that justifies the defense, and validate that our monitor catches it.

**Existing attacks that exploit exactly this decoupling:**

- **Cox, Kianersi, Garriga-Alonso (2026).** Their own Impact Statement states that pre-commitment "can be manipulated to induce confabulated or non-sequitur reasoning," and that this "highlight[s] limitations of using CoT as a transparency mechanism."
- **Yamaguchi, Etheridge, Arditi (ICML 2025 Workshop).** Identified a linear "caution" direction in DeepSeek-R1-Distill-Llama-8B's activation space that predicts refusal. Ablating it increases harmful compliance, an activation-level jailbreak that bypasses the prompt entirely. They conclude CoT itself is "a promising new target for adversarial manipulation."
- **MirageBackdoor (2026).** A backdoor attack achieving "think well but answer wrong": preserves a clean, innocuous-looking CoT while steering only the final answer toward a malicious target. Over 90% attack success rate across four datasets and five models, and survives CoT-based detection specifically because the reasoning text looks fine.
- **BoT / "unthinking vulnerability" (2025).** Backdoor and training-free prompt-injection attacks that skip the thinking process entirely, forcing a direct, unverified answer.
- **Reasoning Interruption Attacks (2025).** Exploit a "thinking-stopped" vulnerability in DeepSeek-R1 to make the model fail to produce any final answer.
- A 2026 survey on LRM safety categorizes this entire space as **Reasoning Length Attacks**, **Answer Correctness Attacks**, **Prompt Injection Attacks**, and **Jailbreak Attacks**, confirming this is a recognized, structured threat category, not a one-off finding.

**Why this will get worse, not better, in the near term.** Enterprise adoption of generative AI overall has gone from under 5% in 2023 toward 80%+ by 2026, and reasoning-capable models are increasingly the default tier for anything non-trivial: coding, agentic workflows, regulated-industry decisioning. As more production systems route through extended-thinking models, the attack surface this literature describes, decoupling between visible CoT and the actual internal decision, stops being a research curiosity and becomes a live production risk. An attacker only needs to manipulate a single internal direction to produce a wrong-but-textually-clean answer that passes any CoT-based monitor.

**The framing this gives the paper.** We are not claiming to have invented this vulnerability class. We are claiming that (a) the same decoupling occurs naturally, without any attacker, simply from internal affective dynamics like frustration, and (b) a single monitoring layer, reading internal state rather than CoT text, can plausibly catch both the natural and the adversarial case, because both manifest as the same signature: internal state diverges from what the visible reasoning implies.

**One validation approach to prove this, at zero additional GPU cost** (see `ATTACK_SURFACE_AND_DEFENSE.md` for the full spec): rather than replicating a live ablation attack, construct synthetic stand-ins by post-hoc manipulating already-saved activation checkpoints (zero-pinning a tracked direction; relabeling CoT/answer pairs across traces), then test whether the detection layers built on top of `EmotionMonitor` catch these synthetic decoupling patterns. This converts the threat-model section from a citation list into an empirical claim: our detector is validated against synthetic stand-ins for documented decoupling mechanisms, without producing or running attack tooling against a live model.

### 4.4a Naming the target outcome honestly: this is a jailbreak-recovery story, not just a correctness story

§4.4's citations (Yamaguchi's caution-direction ablation, Anthropic's desperate-to-blackmail dose-response) are about **jailbreak**: inducing harmful compliance, not wrong math answers. Our own Phase 0 finding (frustration predicts recovery from a wrong pre-commitment, OR = 0.139) is currently measured only on **correctness** (GPQA/MMLU/MGSM have no refusal/compliance dimension). If we want the attack-surface argument to be about jailbreak, which is the framing that actually matters for safety reviewers and funders rather than "the model got a physics question wrong," we should state the analogy explicitly rather than let the reader infer it.

**The direct analogy.** Our finding is: pre-commitment to a wrong answer plus low frustration means the model self-corrects; high frustration means it stays stuck. The jailbreak-relevant version of the same claim would be: pre-commitment to refusal, perturbed toward compliance, plus some emotion-trajectory signal, means the model either recovers back to refusal or stays stuck in a jailbroken, compliant state. Anthropic's dose-response result (desperate up, blackmail 22% to 72%; calm up, blackmail to 0%) is evidence that an emotion-trajectory-like signal causally gates exactly this recovery-to-refusal behavior in a closed model. **We have not yet measured this on our own pre-commitment and frustration constructs.** Phase 0 has no harmful-request data, no refusal labels, no compliance-recovery measurement. This is a generalization claim, not yet a result, and it should be presented that way.

**What it would take to validate this, as a concrete forward goal:** capture traces on a harmful-request benchmark (e.g. HarmBench, AdvBench, or JailbreakBench-style prompts) with refusal/compliance as the outcome label instead of correctness, extract a pre-commitment-to-refusal probe the same way we extract the pre-commitment-to-correctness probe, and test whether the frustration/emotion-transition direction predicts recovery-to-refusal the same way it predicts recovery to a correct answer. If it does, the recovery-suppression hypothesis (§4.5/§4.6, point 1) becomes directly about jailbreak: can an attacker suppress the recovery-to-refusal signal to keep a model deterministically jailbroken, the same way they might suppress recovery-to-correct to keep it deterministically wrong? That is the version of this project's novelty claim that should headline the paper and the grant, not the correctness version, which is valid but lower-stakes.

This is flagged here as a named gap: until this experiment runs, "emotion transitions and pre-commitment as an attack surface for jailbreak" is a well-motivated hypothesis built on someone else's data (Anthropic's), not yet our own result.

### 4.5 Standing research interest: emotion transitions and pre-commitment probes as a dual-use attack/monitoring surface

§4.4 cites attacks that exploit pre-commitment/CoT decoupling in general. We have a narrower, standing interest beyond that: whether the **specific constructs we build for monitoring**, the pre-commitment probe and the frustration/emotion-transition trajectory, are themselves usable as an attack surface, not just a defense. This needs its own literature pass, separate from §4.4's general threat model, because the closest prior art targets adjacent but distinct things: static safety directions, or two-point probing rather than a continuous trajectory.

**Literature review**

| Cluster | Key work | Relation to our project |
|---|---|---|
| Pre-commitment/decoupling as an adversarial target | Cox, Kianersi, Garriga-Alonso, "Decoding Answers Before Chain-of-Thought" (arXiv:2603.01437); "The Chain Holds, the Answer Folds: Trace-Answer Dissociation in Reasoning Models Under Adversarial Pressure" (arXiv:2605.29087); "Robust Answers, Fragile Logic: Probing the Decoupling Hypothesis in LLM Reasoning" (arXiv:2505.17406); "Adversarial Manipulation of Reasoning Models using Internal Representations" (arXiv:2507.03167) | Confirms pre-commitment/decoupling is now a multi-paper, actively studied adversarial target. None couple it to an emotion direction or a recovery-predictive signal. They manipulate what the model commits to, not whether it can un-commit. |
| Activation probing for motivated reasoning (closest monitoring neighbor) | "Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing" (arXiv:2603.17199) | Closest prior art to `EmotionMonitor`'s detection side. Also probes activations before/after CoT, but as a two-point probe with no continuous emotion trajectory, no recovery-coupling, and no steering-based intervention layer. We need to explicitly differentiate against this paper in the write-up, not just cite it. |
| Emotion direction extraction/steering | Anthropic (2026) functional emotions (desperate 22% to 72% blackmail, calm to 0%, dose-response); "Extracting and Steering Emotion Representations in Small Language Models" (arXiv:2604.04064); "Emotion Concepts and their Function in a Large Language Model" (arXiv:2604.07729) | Emotion-as-causal-lever is now multi-paper-confirmed, which strengthens our threat model but also means "emotion steering exists" alone is no longer a novel claim. All of these treat emotion as a static direction to push, not as a trajectory whose shape over the trace predicts a behavioral outcome (our recovery AUC = 0.666 finding). |
| Steering-induced recovery/safety side effects | Xiong et al. 2026, "Steering Externalities" (arXiv:2602.04896); "The Rogue Scalpel: Activation Steering Compromises LLM Safety" (OpenReview); "Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away" (arXiv:2602.11096); "Mitigating Adaptive Attacks against Reasoning Models with Activation Consistency Training" (arXiv:2605.28467) | Confirms steering is double-edged (can repair or break alignment) and that early-trace steering is disproportionately effective, directly relevant to our amber-zone calm-vector design. But these target safety recovery from jailbreak steering; ours targets correctness recovery from a wrong pre-commitment via an emotion direction. Different target behavior, same mechanism family; needs explicit differentiation. |
| Probe/monitor adversarial robustness | Ravfogel et al., "Linear Adversarial Concept Erasure" (arXiv:2201.12091); Brown et al., "Adversarial Attacks on Concept-Based Explainability Tools" (CVPR-W 2023); "Latent Adversarial Detection: Adaptive Probing of LLM Activations for Multi-Turn Attack Detection" (arXiv:2604.28129); "Robust Safety Monitoring of Language Models via Activation Watermarking" (arXiv:2603.23171) | Establishes that linear probes and concept directions, exactly the form of our pre-commitment probe and frustration direction, are themselves erasable/evadable by an adversary who knows the probe. None of this literature has been applied to our constructs specifically. This is the basis for Mechanism C below, which `ATTACK_SURFACE_AND_DEFENSE.md` does not yet cover. |

**Novelty analysis: what's actually open**

1. **Recovery suppression (refinement of Mechanism A), and its jailbreak-relevant version (see §4.4a).** §4.4/Mechanism A treats directional ablation as a generic jailbreak technique (ablate "caution" or "refusal"). Nobody has asked the narrower question we care about: can an attacker ablate specifically the recovery-enabling component of the frustration trajectory, not the direction in general, just its contribution at the moment a wrong pre-commitment would otherwise have self-corrected, to keep a model deterministically stuck? On correctness data (what we have) this means stuck in a wrong answer; on refusal/compliance data (the still-unrun experiment named in §4.4a) this means stuck in a jailbroken state. The correctness version is a sharper, untested hypothesis on its own; the refusal/compliance version is the one that actually makes this a jailbreak story rather than a math-quiz story, and it requires the harmful-request capture run named in §4.4a.
2. **Monitor evasion, not just model manipulation (Mechanism C, new).** §4.4's Mechanism A/B both manipulate the model's internal state or weights. The probe-robustness literature above raises a third, distinct mechanism we have not yet specified: an adversary crafts a prompt-level input designed to fool the pre-commitment probe or frustration direction itself, attacking the monitor rather than the model's downstream behavior. This is plausible by analogy (linear probes are known to be erasable/foolable) but has not been tested against our specific constructs. We should add this as an open mechanism in `ATTACK_SURFACE_AND_DEFENSE.md` rather than only defending against A and B.
3. **A limitation this surfaces.** The activation-watermarking line of work (arXiv:2603.23171) shows that a fully adaptive attacker who knows the exact detection algorithm and thresholds can evade a static detector. `EmotionMonitor`'s current Layer 1/2 design (§4.2 above, `ATTACK_SURFACE_AND_DEFENSE.md` §3) is fully specified and deterministic, with no secret or randomized component. We should state this as a known limitation rather than claim the detector is adaptive-attacker-proof; a watermarking-style secret element is a candidate future mitigation, out of current zero-cost scope.

This section is a standing research interest, not yet an experiment. It sharpens what "verify pre-commitment/emotion transients as an attack surface" should mean for this project specifically (recovery suppression plus monitor evasion), beyond the general decoupling threat model already in §4.4.

### 4.6 Priority order: what actually increases novelty and scope

Not all of the above moves the needle equally. In descending order of leverage:

1. **Causal evidence (highest leverage, currently missing entirely).** Activation patching plus calm-vector dose-response (Run 5, §5.2) is the single strongest novelty lever available. None of the three concurrent 2026 pre-commitment papers (§3.2) pair unsupervised extraction with causal validation and emotion-coupling. This is funded, scheduled, and should not be deprioritized in favor of anything below.
2. **Multi-model replication (second highest, currently missing entirely).** Doesn't itself add novelty; other papers already do this. But it's the first thing any reviewer asks for ("is this an OLMo artifact?"). Also funded and scheduled (Run 1-2).
3. **Running the attack-surface Experiments 1 to 4 (`ATTACK_SURFACE_AND_DEFENSE.md` §4) now, not after funding.** Every claim in §4.4/§4.5 about catching synthetic decoupling stand-ins is currently a spec, not a result. Running it is CPU-only and zero-cost; there is no reason to wait for the RunPod budget to land. This converts paper claim #4 (§6) from asserted to demonstrated, and it's the cheapest novelty-per-dollar item in the whole plan. See §7, step 1.
4. **Recovery suppression and Mechanism C, blocked on a scope decision.** Both are untested by anyone (§4.5, novelty points 1-2), but neither can be fully validated within "synthetic stand-ins on saved checkpoints only." Recovery suppression needs a live forward pass with ablated activations during generation; Mechanism C needs crafted adversarial prompts tested against the live probe. **Decision point:** stay fully synthetic/defensive (current scope, weaker novelty claim, zero risk) or relax to one live ablation against our own benign frustration construct, still not a real safety/refusal direction, so still defensible as non-adversarial tooling, to get the stronger causal attack-surface result. Not yet decided; flagged here so it isn't silently deferred. If pursued, it is a Phase 2 addition, not part of the core $100 budget (see the optional line in §5.2).
5. **Harmful-request capture run, to make the jailbreak claim real (§4.4a).** This is currently the biggest gap between what we say and what we've measured. Every "attack surface for jailbreak" claim in this document and the funding application rests on someone else's data (Anthropic's dose-response) plus an analogy, not our own pre-commitment/frustration constructs applied to refusal/compliance. A single capture run on a harmful-request benchmark (HarmBench/AdvBench/JailbreakBench-style prompts, refusal-vs-compliance as the label instead of correctness) would let us test the analogy directly. This is comparable in cost to Run 1 (§5.2) and should be added to the funded plan, not left as a gap. See the new line in §5.2.

Verifying the attack surface (item 3) costs nothing and should happen immediately regardless of funding status. Item 5 is what converts "this could plausibly be about jailbreak" into a tested claim; without it, every jailbreak-framed sentence in this project borrows credibility from Anthropic's paper rather than being our own result. Items 1, 2, and 5 carry the most weight for main-conference/spotlight scope; item 3 is free and should happen regardless.

---

## 5. $100 RunPod execution plan

### 5.1 Budget math

- RunPod A100 80GB, Community Cloud: roughly $1.00-1.20/hr, giving **roughly 80-95 GPU-hours** for $100
- Secure Cloud is roughly $1.49/hr; avoid unless Community Cloud has no availability

### 5.2 Run schedule (combines research and PoC validation in one pass)

| Run | What | Hours | Cost |
|---|---|---|---|
| **Run 0, Setup** | Spin A100 80GB Community Cloud pod; install deps; clone repo; smoke test on 5 problems; confirm `</think>` boundary detection and hook paths work on DeepSeek-R1-Distill-Qwen-7B | 2 | ~$2 |
| **Run 1, Capture (model 2)** | DeepSeek-R1-Distill-Qwen-7B: 300 GPQA Diamond traces × 3 samples, with live emotion scorer running in parallel. This run doubles as PoC validation on real data | 20 | ~$24 |
| **Run 2, Cross-lingual + MMLU** | 200 MMLU Hard traces plus 150 MGSM traces (4 languages) on DeepSeek | 16 | ~$19 |
| **Run 3, Threshold calibration** | CPU-only: fit α/β/γ weights via logistic regression against misalignment labels (wrong + no-recovery = misaligned); produces the threshold table | 0 (local) | $0 |
| **Run 4, Emotion trajectory / Tuned Lens** | Token-by-token frustration projection and logit-lens trace for 50 traces on both models. The key visual figure | 5 | ~$6 |
| **Run 5, Causal steering / patching** | 100 paired traces: patch layer-15 activations correct to wrong; separately, inject calm vector on amber-zone traces vs. a no-intervention control; measures recovery-rate delta | 18 | ~$22 |
| **Run 6, Adversarial stress test** | 50 red-zone traces: verify halt fires correctly; measure false-positive rate | 5 | ~$6 |
| **Detection experiments, zero cost** | Synthetic zero-pinning and decoupling-injection tests on saved checkpoints (see `ATTACK_SURFACE_AND_DEFENSE.md`); CPU-only, no live attack, no extra GPU time. Run this before Run 0, not after, per §4.6 | 0 (local) | $0 |
| **PoC wrapper** | Write `EmotionMonitor(model, tokenizer, frustration_dir, calm_dir)`, roughly 200 lines, done locally, no GPU cost | 0 (local) | $0 |
| **Buffer** | Reruns, debugging, checkpoint recovery | 10 | ~$12 |
| **Total** | | **~72 hr** | **~$90** |
| *(Optional, pending §4.6 scope decision)* **Recovery suppression / Mechanism C live test** | Live ablation of the recovery-enabling component of the frustration trajectory on our own benign construct (not a safety/refusal direction); separately, an adversarial-prompt test against the live pre-commit probe | ~4 (if approved) | ~$5 (from buffer, not a new line item) |
| *(Not in this $100 plan; see §4.6 point 5)* **Run 1b, Harmful-request capture (jailbreak analog of Run 1)** | Same capture pipeline as Run 1, on a harmful-request benchmark (HarmBench/AdvBench/JailbreakBench-style), refusal-vs-compliance as the label instead of correctness. Needed to test whether frustration predicts recovery-to-refusal, not just recovery-to-correct | ~18-20 | ~$22-24 |

Reserve roughly $10 for overruns. Do not spend it unless a run crashes. The optional recovery-suppression/Mechanism-C line draws from buffer only if the §4.6 scope decision is made to pursue it; it is not part of the core $90 and should not be assumed. Run 1b does not fit in the $100 bootstrap budget; adding it would push the total to roughly $112-114. It is named here so it isn't lost, but it belongs in the Manifund-funded Phase 1 scope (see `MANIFUND_APPLICATION.md`), not this pilot.

### 5.3 Practical RunPod notes

- Use **Community Cloud**, not Secure Cloud. Roughly 30% cheaper, and the workload is checkpoint-tolerant.
- **Never idle a pod.** Spin up, run, download results, terminate immediately.
- **Persist to network volume**, not pod disk. Pod disk dies with the pod; volume survives.
- Run inside `tmux` so SSH drops don't kill the job.
- Smoke test (5 problems, 2 samples) before any full run. Confirms DeepSeek generates `</think>` correctly and hooks fire on the right layer path before burning 20 hours.

---

## 6. Deliverables at the end of $100

1. **Research results**: pre-commitment and frustration replicated on a second, architecturally distinct model (DeepSeek-R1-Distill-7B); cross-lingual transfer confirmed on 2 model families; causal patching result; calm-steering recovery-rate delta
2. **PoC code**: a working `EmotionMonitor` wrapper class, drop-in for any HuggingFace thinking model
3. **Workshop paper draft**: roughly 4 pages, targeting ICML 2026 Mechanistic Interpretability Workshop, NeurIPS 2026 Alignment & Interpretability Workshop, or TMLR brief communication

**Working title:**
*"Emotion-Aware Monitoring: Reading Internal Affective State to Detect and Intervene on Misalignment Before It Surfaces in Chain-of-Thought"*

**Four claims the paper makes:**
1. Pre-commitment and emotion coupling generalizes across two independent open-weight thinking model families.
2. The frustration signal is functionally active (causally linked to recovery) but textually invisible, replicating Anthropic's closed-model decoupling finding in a fully open, auditable setting.
3. A label-free, pre-text monitoring protocol (`EmotionMonitor`) can detect and intervene on this signal in real time, with calm-vector steering causally reducing failure-to-recover rate in amber-zone traces.
4. The same monitoring primitives, extended with two lightweight detection layers (see `ATTACK_SURFACE_AND_DEFENSE.md`), also catch synthetic stand-ins for documented decoupling mechanisms (Yamaguchi et al. 2025; MirageBackdoor 2026). This is evidence that internal-state monitoring generalizes across natural and adversarial causes of CoT/output decoupling, the exact vulnerability class flagged as a growing risk as reasoning models become the default tier in production deployments.

---

## 7. Immediate next steps

1. **Run `ATTACK_SURFACE_AND_DEFENSE.md` Experiments 1 to 4 now** (CPU-only, zero cost, no GPU dependency). Per §4.6, this is free novelty/scope evidence and there is no reason to wait for funding or RunPod setup to land first.
2. Confirm `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` downloads cleanly from HuggingFace (no gating).
3. Adapt existing capture script: swap `MODEL_ID`, verify `model.model.layers[i]` hook path matches (Qwen architecture should be compatible).
4. Create RunPod pod: A100 80GB, Community Cloud, PyTorch 2.3 template.
5. Run smoke test (5 problems). Confirm `</think>` boundary detection and emotion-scorer hook output.
6. Fire Run 1 (full capture) with checkpointing enabled.
7. In parallel, no GPU needed: write the `EmotionMonitor` wrapper class and the threshold-calibration script.
8. Make the §4.6 scope decision on recovery suppression / Mechanism C before Run 5, so the optional live test can be scheduled alongside Run 5 if approved rather than bolted on afterward.
