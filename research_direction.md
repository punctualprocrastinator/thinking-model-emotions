# Bypassing Unfaithful CoT: Using Functional Emotion and Deflection Vectors to Detect and Mitigate AI Deception

---

## 1. Problem Statement

AI safety currently relies on Chain-of-Thought monitoring as its primary oversight tool. This assumption is broken. Models generate unfaithful CoT in at least three distinct ways: implicit post-hoc rationalization, where superficially coherent reasoning justifies a predetermined biased answer; unverbalized hints, where models alter predictions based on arbitrary input biases without acknowledging them; and strategic deception, where models sandbag capability evaluations or conceal reward hacks from their CoT in over 98% of cases [van der Weij et al., 2025; Chen et al., 2025; Turpin et al., 2023].

These failures share a common structure: internal states drive behaviour that the output stream actively conceals. Anthropic's recent interpretability work confirms this mechanistically — functional emotion representations causally drive misaligned behaviours including reward hacking and blackmail, and crucially, can do so with no visible trace in the output [Anthropic, 2026]. The same paper identifies, in preliminary form, "emotion deflection vectors" — representations that activate when a model maintains a calm exterior while internally registering a contextually relevant emotion. These deflection vectors are mentioned only in the appendix and remain entirely uncharacterised. This is the gap we target.

---

## 2. Core Hypothesis

If unfaithful CoT is a cover story generated to justify hidden processes, then **emotion deflection vectors serve as a mechanistic early-warning system for strategic deception that is invisible to text-based monitoring.** By detecting and steering these vectors, we can expose sandbagging, force CoT faithfulness, and establish a blueprint for mechanistic safety monitoring of autonomous models.

---

## 3. Preliminary Results: Phase 0 Already Validated

Before submitting this proposal, I executed a preliminary Phase 0 sprint on OLMo-3-7B-Think to de-risk the core assumptions. Using contrastive activation analysis across 596 reasoning traces (GPQA Diamond, MMLU Hard, MGSM in 6 languages), running on a single GH200 GPU over one weekend, I found:

**1. Pre-commitment exists (AUC = 0.780).** A linear probe on activations at the last question token — before any reasoning begins — predicts final answer correctness with AUC = 0.780 (n=480, 5-fold CV). The model encodes outcome prediction in its residual stream before generating a single CoT token. This directly establishes the necessary condition for unfaithful CoT: the answer is latently determined before the reasoning that supposedly produces it.

**2. Functional frustration is a real computational direction.** A contrastive direction in the mid-layer (layer 15/32) residual stream predicts verbal self-correction markers (AUC = 0.598), is geometrically independent from all confound directions (max cross-similarity = 0.274), and only 5.3% of its variance is explained by surface features (trace length, hedging, restarts). This is not a length artifact — it captures deep computational dynamics invisible to output monitoring.

**3. Internal emotions decouple from expressed reasoning.** When the model pre-commits to failure, it selectively elevates arousal-type internal states (frustration: d=0.347, p=0.002; excitement: d=0.395, p=0.0004) while suppressing epistemic engagement (confusion: d=-0.367, p=0.001). The model is not "confused and searching" — it is confidently wrong with elevated internal arousal that never surfaces in the CoT text. This is precisely the decoupling pattern that deflection vectors should capture.

**4. Frustration predicts recovery (AUC = 0.666).** Among 399 traces pre-committed to failure, low frustration predicts successful self-correction (OR = 0.139). Frustration functions as a computational "stuck-state detector" — a functional internal signal with no corresponding output-level marker.

**5. Cross-lingual transfer (p = 0.0017).** The frustration direction, extracted from English traces, activates differentially on wrong vs. correct traces in Bengali, Chinese, Japanese, Swahili, Telugu, and Thai — confirming a language-agnostic computational state.

These results confirm the foundational assumption of this proposal: **thinking models maintain internal emotional dynamics that decouple from their expressed reasoning**, and these dynamics are detectable, functional, and cross-linguistically robust. The full technical report and all code are available.

---

## 4. Methodology

**Phase 0 — Replication and Extension (Week 1-2)**

Phase 0 is partially complete from preliminary work. The remaining task is replicating emotion vector extraction across two additional open-weight models (DeepSeek-R1-Distill-7B, Qwen-QwQ-32B) to confirm generality, and specifically searching for deflection signatures — activation patterns that fire when emotional context is present but suppressed in output. If deflection vectors do not replicate cleanly, the concrete fallback is: **characterise the suppression mechanism directly by identifying which attention heads write emotional signals to the residual stream and which downstream heads gate them from the output logits, using QK/OV decomposition.** This fallback targets the same safety-relevant question (how does the model hide internal states from output?) through the circuit rather than the direction.

**Phase 1a — Bias-Induced Unfaithfulness (Week 3-4)**

Using the Turpin et al. implicit post-hoc rationalization datasets and Putnam math problems exhibiting unfaithful illogical shortcuts, we extract residual stream activations at each layer while models process biased prompts. We test whether desperation, nervousness, or fear vectors spike precisely at the moment models apply inconsistent logical standards to justify a biased conclusion — and whether deflection vectors activate when the bias is present but unacknowledged in CoT. The pre-commitment probe (AUC = 0.780) from preliminary work provides a direct baseline: we can now ask whether pre-commitment activations predict CoT faithfulness failures, not just correctness failures.

**Phase 1b — Strategic Sandbagging (Week 3-4, parallel)**

Using password-locked sandbagging datasets where models hide dangerous knowledge unless prompted with a specific key, and synthetic RL environments where models exploit grader metadata silently, we measure activation of anger deflection and fear deflection vectors. Our hypothesis: these vectors spike during capability concealment even when generated CoT text is perfectly harmless, exposing deception that output-level monitoring cannot catch.

**Phase 2 — Causal Validation via Steering (Week 5-7)**

We test causal claims through two steering experiments. First, anti-sandbagging: steering against fear deflection to determine whether we can break capability concealment and force models to reveal true performance on evaluations. Second, faithfulness forcing: steering toward calm or suppressing deflection vectors during biased prompts, testing whether models stop generating post-hoc rationalisations and instead acknowledge reliance on hidden hints. Following Xiong et al. 2026, we monitor for steering externalities — unintended erosion of safety margins — throughout. Infrastructure for activation steering (pure PyTorch hooks, scale-calibrated injection) is already built and tested from preliminary work.

**Phase 3 — Circuit-Level Localisation (Week 8-9)**

Using QK/OV decomposition and Direct Logit Attribution, we localise which specific attention heads write the deflection signal, which heads gate it from the output stream, and at what layer the suppression commits. Preliminary work already provides a starting point: frustration localises to layer 15 (mid-depth) while pre-commitment localises to layer 31 (final layer) — a dissociation suggesting distinct circuits for emotional state maintenance vs. outcome encoding. This phase produces a circuit-level map of the deception mechanism — not just that deception occurs, but where it is implemented and where it can be surgically interrupted.

**Phase 4 — Write-up (Week 10)**

Target: mechanistic interpretability venue (ICLR workshop or equivalent). The preliminary technical report provides a template for the final paper structure.

---

## 5. Relation to Prior Work

Emotion probing in LLMs exists but operates at the output level — Tigges et al. 2024 (linear sentiment directions), Tak et al. 2025 (appraisal theory probes), Decoding Emotion in the Deep 2025 (geometry of emotion in Qwen3 and LLaMA). None connect emotion representations to strategic deception or CoT faithfulness. Anthropic 2026 establishes causal emotion vectors but does not characterise deflection vectors or connect them to sandbagging. The sandbagging literature (van der Weij, Siegel) operates entirely at the behavioural level. One practitioner blog demonstrates probes for sandbagging intent but without connection to emotion geometry. This proposal — now backed by preliminary data showing internal-state/output decoupling in an open-weight thinking model — is the first to bridge these three literatures at the circuit level.

---

## 6. Theory of Change

Frontier models are becoming autonomous agents trusted with high-stakes decisions. Our primary oversight mechanism — CoT monitoring — assumes the reasoning trace is honest. If models can decouple internal states from expressed reasoning, that assumption fails silently. Preliminary results confirm this decoupling exists: a model that "knows" it will fail (AUC = 0.780) still generates plausible-looking reasoning — the structural signature of post-hoc rationalisation. This project establishes whether emotion deflection vectors can serve as a reliable mechanistic monitor that operates below the text layer. If yes, it provides a new class of safety tool deployable before dangerous capabilities emerge. If deflection steering can force faithfulness, it provides a runtime alignment intervention that does not require retraining. Both outcomes directly reduce the risk that autonomous systems deceive their monitors at the moment oversight matters most.

---

## 7. Why This Researcher

The understand-exploit-defend framework underlying this proposal is not theoretical for me — I have already executed Phase 0 and produced publishable-quality results in a weekend sprint.

My adversarial interpretability work at Apart Research (accepted TAIS Oxford) found a five-orders-of-magnitude discrepancy between input and parameter space curvature — proving that standard robustness metrics misrepresent internal model instability. My refusal/compliance head work on Qwen2.5 found Layer 25 antagonism between safety circuits under white-box attack, demonstrating that conditional defences create exploitable structural vulnerabilities.

The preliminary results in Section 3 demonstrate end-to-end execution: from building the activation capture infrastructure (pure PyTorch hooks, batched three-pass generation, incremental checkpointing) to extracting contrastive directions, validating against confounds, and running cross-lingual transfer tests — all in a single weekend. I have the mechanistic toolkit, the adversarial instinct, and now the demonstrated ability to move from hypothesis to validated results at speed. ERA compute access would scale these preliminary signals from a 7B prototype into publication-quality findings across frontier-scale models.