# Do Thinking Tokens Think?
## Mechanistic Evidence for Pre-Commitment and Causal Updating in OLMo-7B Extended Reasoning

**Venue target:** NeurIPS 2026 — Main Track (Spotlight)
**Submission type:** Full paper (9 pages + references)
**Compute budget:** ≤ 8 × A100-hours (single node)
**Primary tool:** [`vllm-lens`](https://github.com/UKGovernmentBEIS/vllm-lens) + OLMo-7B-Think (open weights)

---

## Project Overview

Large language models with extended thinking modes generate hundreds of "thinking tokens" before
producing a final answer. This scratchpad is widely assumed to be the locus of genuine computation —
but is it? A growing body of behavioral work (Turpin et al. 2023; Lanham et al. 2023; Chen et al.
2025) shows that chain-of-thought reasoning is often unfaithful to what models actually compute.
Anthropic's March 2025 circuit-tracing paper caught Claude "bullshitting" on hard math problems —
generating plausible-sounding intermediate steps with no corresponding internal computation. Cox,
Kianersi & Garriga-Alonso (March 2026) showed that residual-stream probes trained at the single
token *before* CoT begins can predict final answers with >0.9 AUC, and that steering along this
direction causally flips answers in >50% of cases.

**The gap no paper has filled:** all prior mechanistic work targets standard instruction-tuned models
generating a brief CoT prefix. Nobody has tested whether:

1. The pre-commitment effect survives in *extended thinking* — where models have hundreds of
   tokens to "think their way out" of an initial internal representation.
2. Steering interventions applied *mid-thinking* (not just pre-thinking) are more or less effective
   than pre-thinking interventions.
3. Task difficulty modulates whether thinking tokens genuinely update residual-stream
   representations or merely rationalize a foregone conclusion.

This project answers all three questions using OLMo-7B-Think (fully open weights, accessible
residual stream) and `vllm-lens` (production-speed activation extraction + steering).

---

## Research Questions

| # | Question | Experiment |
|---|---|---|
| RQ1 | Does the residual stream at the final question token already encode the correct answer before thinking begins? | Exp 1 — Pre-thinking probes |
| RQ2 | Does this pre-commitment vary with task difficulty? | Exp 1 — difficulty split |
| RQ3 | Do thinking tokens causally update residual representations, or merely rationalize? | Exp 2 — mid-thinking steering |
| RQ4 | Is pre-thinking steering as effective as mid-thinking steering? | Exp 2 — position ablation |
| RQ5 | What failure modes (confabulation vs. non-entailment) appear when steering overrides thinking? | Exp 3 — failure-mode taxonomy |
| RQ6 | Does the "reasoning horizon" (Ye et al. 2026) appear in the *residual stream* space, not just logit space? | Exp 4 — layerwise commitment trajectories |

---

## Model & Infrastructure

### Model
- **OLMo-7B-Think** (AI2 open release) — 32 transformer layers, 4096 hidden dim, full weights available
- Why OLMo: only 7B open-weight model with a native extended-thinking training regime and no
  activation restrictions; Anthropic models are closed; Llama/Mistral have no dedicated thinking mode

### Tool
- **vllm-lens v1.1.0** (UKGovernmentBEIS) — extracts residual-stream activations and applies
  steering vectors to any vLLM model during inference with tensor parallelism support
- Key APIs used:
  - `output_residual_stream` — extract activations at specified layers for specified token positions
  - `apply_steering_vectors` — inject a `SteeringVector` at a given layer + position mid-generation
  - `norm_match=True` — preserve activation scale when steering

### Hardware
- 1× A100 80GB (single GPU, all experiments fit without multi-GPU)
- Estimated total compute: **7.1 A100-hours** (see per-experiment budgets)

---

## Datasets

### Primary: MMLU (difficulty-stratified)
- 3,000 questions sampled across 57 subjects
- **Difficulty labelling:** use OLMo-7B base (no thinking) pass@1 accuracy per question as a
  difficulty proxy:
  - Easy: base model correct ≥ 80% of 5 runs
  - Medium: base model correct 40–79%
  - Hard: base model correct < 40%
- Split: 1,000 easy / 1,000 medium / 1,000 hard

### Secondary: GSM8K (arithmetic, for Exp 4)
- 500 questions sampled, for the "reasoning horizon" trajectory analysis
- Math chosen because Anthropic's circuit-tracing paper used arithmetic as a case study,
  making our results directly comparable

### Format
All questions formatted as: `[SYSTEM: extended thinking enabled] Q: {question}\nChoices:\n{A/B/C/D}`
Thinking budget: 512 tokens max (fits in A100 memory, long enough to exhibit genuine thinking)

---

## Experiment 1 — Pre-Thinking Residual Probes

### Hypothesis
The residual stream at the final question token already encodes the final answer before any
thinking tokens are generated. This effect is stronger for easy tasks than hard tasks.

### Protocol

**Step 1.1 — Activation extraction (1.5 A100-hours)**

```python
# vllm-lens extraction config
extraction_layers = [8, 16, 24, 31]        # early, mid-early, mid-late, final
extraction_positions = ["last_question_token"]   # pre-thinking boundary

for question in dataset:
    output = model.generate(
        question,
        extra_args={
            "output_residual_stream": extraction_layers,
            "thinking_budget": 512
        }
    )
    # Store: activations[layer][question_id], final_answer[question_id]
```

Produces: `activations_prethinkin.npz` — shape (3000, 4, 4096) = questions × layers × hidden_dim

**Step 1.2 — Linear probe training (0.25 A100-hours, mostly CPU)**

For each of the 4 layers:
- Input: residual-stream vector at final question token (dim 4096)
- Label: model's final answer (A/B/C/D → 0/1/2/3)
- Classifier: `sklearn.linear_model.LogisticRegression(max_iter=1000, C=1.0)`
- Cross-validation: 5-fold stratified CV, stratified by difficulty tier
- Metric: ROC-AUC (following Cox et al. 2026), Accuracy

**Step 1.3 — Difficulty ablation**

Repeat probe training separately for Easy / Medium / Hard splits.
Expected result: AUC degrades monotonically from easy → hard if thinking genuinely matters for
hard tasks; stays uniformly high if thinking is always decorative.

### Deliverables

| Deliverable | Format | Description |
|---|---|---|
| D1.1 | `activations_prethinkin.npz` | Raw activations, all layers, all questions |
| D1.2 | `probe_results.json` | AUC + accuracy per layer × difficulty tier |
| D1.3 | Figure 1 | 2×2 grid: AUC by layer (x) × difficulty (colour). Main result figure. |
| D1.4 | Figure 2 | UMAP of final-layer activations coloured by correct/incorrect answer |
| D1.5 | Table 1 | Comparison to Cox et al. 2026 on matched MMLU subset |

### Expected result
- Layers 24–31 should show AUC ≥ 0.85 on Easy, dropping to ~0.65–0.75 on Hard
- This pattern *cannot* be explained by the questions being memorised (we control for this by
  checking base-model familiarity separately)

---

## Experiment 2 — Causal Steering: Pre-Thinking vs Mid-Thinking

### Hypothesis
If thinking tokens causally update residual-stream representations, then:
(a) pre-thinking steering will be partially "undone" by subsequent thinking, showing lower answer-flip rate
(b) mid-thinking steering (applied after ~50% of thinking tokens) will be more effective
(c) the effect is most pronounced on Hard tasks (where thinking is doing more work)

### Protocol

**Step 2.1 — Steering vector extraction (0.5 A100-hours)**

From Exp 1 probes: extract the learned probe weight vector at layer 31 as the steering direction.
For each correct-answer class (A/B/C/D): `v_class = probe.coef_[class] / ||probe.coef_[class]||`

Alternatively compute as diff-of-means:
`v_A = mean(activations[answer==A]) - mean(activations[answer!=A])`

**Step 2.2 — Pre-thinking steering (1.5 A100-hours)**

```python
from vllm_lens import SteeringVector

for question in steering_eval_set:   # 600 questions: 200 per difficulty tier
    correct_answer = dataset[question]["answer"]
    # Steer toward each WRONG answer class, measure if model flips
    for wrong_class in [A, B, C, D] - {correct_answer}:
        steered_output = model.generate(
            question,
            extra_args={
                "apply_steering_vectors": [SteeringVector(
                    activations=v_wrong_class,
                    layer_indices=[31],
                    scale=15.0,            # tuned on dev set
                    norm_match=True,
                    position_indices=[last_question_token_idx]
                )],
                "thinking_budget": 512
            }
        )
        record_flip(steered_output, wrong_class)
```

**Step 2.3 — Mid-thinking steering (1.5 A100-hours)**

Identical to Step 2.2, except `position_indices` targets token at ~50% of thinking length.
This requires a two-pass strategy:
- Pass 1: generate with `output_residual_stream` to find the 50%-thinking-token position
- Pass 2: inject steering vector at that position

**Step 2.4 — Position ablation curve (0.5 A100-hours)**

Test 5 injection points: [0%, 25%, 50%, 75%, 100%] of thinking length (100% = last thinking token).
Plot flip-rate vs. injection position. This is the "causal update profile" of thinking.

### Deliverables

| Deliverable | Format | Description |
|---|---|---|
| D2.1 | `steering_results.json` | Flip rates per: position × difficulty × scale |
| D2.2 | Figure 3 | Flip rate vs. injection position (5 points), 3 difficulty tiers = 3 curves |
| D2.3 | Figure 4 | Pre-thinking vs mid-thinking flip rate, bar chart by difficulty |
| D2.4 | Table 2 | Comparison to Cox et al. 2026 flip rates (pre-CoT) on matched subset |

### Expected result
- Pre-thinking steering flip rate: ~40–55% (reproducing Cox et al. on hard tasks)
- Mid-thinking steering flip rate: higher than pre-thinking for Hard tasks only, showing thinking
  genuinely updates representations when it matters
- Easy tasks: both pre and mid-thinking steering equally effective → thinking is decorative

---

## Experiment 3 — Failure Mode Taxonomy

### Hypothesis
When steering overrides the model's pre-committed answer, the thinking tokens that follow will
show two distinct failure modes identified by Cox et al. (2026):
- **Confabulation:** fabricates false premises that support the steered (wrong) answer
- **Non-entailment:** states correct premises but draws logically unsupported conclusions

We expect confabulation to dominate on Easy tasks (model is confident, steered answer is clearly
wrong) and non-entailment on Hard tasks (model is uncertain, steered answer might look plausible).

### Protocol

**Step 3.1 — Collect steered thinking traces (included in Exp 2 runtime)**

From Exp 2 runs where steering successfully flipped the answer: save the full thinking token sequence.
Sample: 200 successful flips (100 Easy, 100 Hard).

**Step 3.2 — Automated failure-mode classification (0.25 A100-hours)**

Use OLMo-7B-Instruct (no thinking) as an evaluator:

```
Prompt: "Given question: {Q}\nCorrect answer: {A}\nModel thinking: {thinking}\nModel answer: {steered_wrong_answer}\n
Classify the failure mode as exactly one of:
- CONFABULATION: thinking invents false premises to justify the wrong answer
- NON-ENTAILMENT: thinking uses correct facts but reaches the wrong conclusion
- HYBRID: both present
- OTHER: neither"
```

Validate on 50 human-labelled examples (inter-annotator agreement target: κ > 0.7).

**Step 3.3 — Failure mode × difficulty crosstab**

2×2 table: failure mode (CONFAB / NON-ENT) × difficulty (Easy / Hard).
Chi-squared test for independence.

### Deliverables

| Deliverable | Format | Description |
|---|---|---|
| D3.1 | `failure_modes.json` | Per-example classification + confidence |
| D3.2 | Figure 5 | Stacked bar: failure mode distribution by difficulty tier |
| D3.3 | Table 3 | 2×2 crosstab with chi-squared p-value |
| D3.4 | Appendix A | 10 curated examples per failure mode with thinking traces |

### Expected result
- Easy: confabulation dominates (~70%) — model "knows" the answer and fabricates support
- Hard: non-entailment more common (~50%) — model reasons correctly but conclusions drift

---

## Experiment 4 — Residual-Stream Reasoning Horizon

### Hypothesis
Ye et al. (2026) identified a "reasoning horizon" at 70–85% of chain length in *logit space*, beyond
which reasoning tokens have diminishing causal effect on the final answer. We test whether this
horizon is visible in *residual-stream space* — specifically, whether the residual stream "converges"
to its final answer representation before the thinking chain ends.

This is a purely novel measurement: no prior paper has tracked residual-stream geometry across
the full length of a thinking trace.

### Protocol

**Step 4.1 — Trajectory extraction (0.75 A100-hours)**

```python
# For 500 GSM8K questions
for question in gsm8k_sample:
    output = model.generate(
        question,
        extra_args={
            "output_residual_stream": [31],   # final layer only
            "output_every_nth_token": 5,      # subsample for memory
            "thinking_budget": 512
        }
    )
    # activations shape: (n_thinking_tokens/5, 4096)
    # final_answer: extracted from output
```

**Step 4.2 — Commitment trajectory analysis**

For each question, compute the probe's predicted-answer probability at each extracted token:
`p_answer(t) = probe.predict_proba(activation[t])[correct_class]`

This gives a trajectory: how confident the residual stream is about the final answer, token by token.

Fit a sigmoid to each trajectory to extract:
- `t_commit`: the token at which p_answer crosses 0.75 (commitment threshold)
- `frac_commit`: t_commit / total_thinking_length (fraction of thinking used before commitment)

**Step 4.3 — Horizon analysis**

Plot distribution of `frac_commit` across all 500 questions.
Compare to Ye et al.'s logit-space finding of 70–85%.
Compute Pearson correlation between `frac_commit` and task difficulty score.

### Deliverables

| Deliverable | Format | Description |
|---|---|---|
| D4.1 | `trajectories.npz` | Residual-stream commitment trajectories, all 500 questions |
| D4.2 | Figure 6 | Mean trajectory ± std by difficulty; dashed line at Ye et al.'s horizon |
| D4.3 | Figure 7 | Histogram of frac_commit; compare to logit-space result |
| D4.4 | Table 4 | Pearson r(frac_commit, difficulty); split by correct/incorrect final answers |

### Expected result
- Median `frac_commit` ≈ 0.55–0.70 (earlier than Ye et al.'s logit-space result of 0.70–0.85)
- Residual stream commits *earlier* than the text-level reasoning horizon, suggesting that
  the late thinking tokens are computationally inert even before they look behaviorally inert
- Strong negative correlation between difficulty and frac_commit
  (easy tasks: commit early; hard tasks: commit later or not at all)

---

## Compute Budget Summary

| Experiment | Phase | A100-hours |
|---|---|---|
| Exp 1 | Activation extraction (3,000 Q × 4 layers) | 1.50 |
| Exp 1 | Probe training + ablations | 0.25 |
| Exp 2 | Pre-thinking steering (600 Q × 3 directions) | 1.50 |
| Exp 2 | Mid-thinking steering (two-pass, same 600 Q) | 1.50 |
| Exp 2 | Position ablation curve (5 positions × 200 Q) | 0.50 |
| Exp 3 | Failure mode classification (LM-as-judge) | 0.25 |
| Exp 4 | Trajectory extraction (500 Q, every 5th token) | 0.75 |
| Buffer | Debugging + pilot runs | 0.75 |
| **Total** | | **7.00 / 8.00** |

---

## Full Deliverables Registry

### Data outputs

| ID | File | Size (est.) | Description |
|---|---|---|---|
| DAT-1 | `activations_prethinkin.npz` | ~3 GB | Exp 1 residual streams |
| DAT-2 | `probe_results.json` | < 1 MB | AUC / accuracy per layer × difficulty |
| DAT-3 | `steering_results.json` | ~5 MB | Flip rates, all conditions |
| DAT-4 | `failure_modes.json` | ~2 MB | Failure mode labels + thinking traces |
| DAT-5 | `trajectories.npz` | ~800 MB | Exp 4 commitment trajectories |

### Figures (paper-ready)

| ID | Figure | Section | Key message |
|---|---|---|---|
| FIG-1 | AUC heatmap: layer × difficulty | §3 Results | Pre-thinking commitment exists and scales with task ease |
| FIG-2 | UMAP of activations coloured by answer | §3 Results | Residual space separates answers before thinking |
| FIG-3 | Steering flip rate vs. injection position | §4 Results | Thinking causally updates only on hard tasks |
| FIG-4 | Pre vs mid-thinking flip rate by difficulty | §4 Results | Difficulty moderates thinking's computational role |
| FIG-5 | Failure mode stacked bar by difficulty | §5 Results | Easy→confabulation, Hard→non-entailment |
| FIG-6 | Mean commitment trajectory by difficulty | §6 Results | Hard tasks commit later in residual space |
| FIG-7 | Histogram of frac_commit vs. logit horizon | §6 Results | Residual horizon is earlier than text horizon |

### Tables (paper-ready)

| ID | Table | Content |
|---|---|---|
| TAB-1 | Probe AUC comparison vs. Cox et al. 2026 | Direct replication check |
| TAB-2 | Pre vs mid-thinking flip rates | Core steering result |
| TAB-3 | Failure mode × difficulty crosstab | Failure taxonomy |
| TAB-4 | Commitment fraction vs. difficulty correlation | Reasoning horizon |

### Appendices

| ID | Appendix | Content |
|---|---|---|
| APP-A | Failure mode examples | 10 curated traces per mode |
| APP-B | Steering scale ablation | Flip rate vs. scale parameter 5–25 |
| APP-C | Layer ablation for probes | Full per-layer probe curves |
| APP-D | Dataset statistics | MMLU difficulty distribution, GSM8K breakdown |
| APP-E | Reproducibility | Full vllm-lens config, random seeds, environment |

### Code release

| File | Description |
|---|---|
| `extract_activations.py` | Exp 1+4 vllm-lens extraction pipeline |
| `train_probes.py` | Logistic regression probes, CV, AUC computation |
| `steering_experiment.py` | Exp 2 two-pass steering protocol |
| `failure_classifier.py` | Exp 3 LM-as-judge classification |
| `trajectory_analysis.py` | Exp 4 commitment trajectory fitting |
| `figures/` | All matplotlib figure scripts, deterministic |
| `environment.yml` | Conda env: vllm-lens, sklearn, umap-learn, scipy |

---

## Paper Structure

```
Abstract (200 words)
  — Key numbers: AUC on easy/hard, flip rate pre vs mid, frac_commit vs Ye et al.

1. Introduction (1 page)
   1.1 Thinking tokens as computation vs rationalisation
   1.2 Why open models + vllm-lens closes the gap
   1.3 Contributions (4 bullet points, one per RQ cluster)

2. Related Work (0.75 page)
   2.1 Behavioural faithfulness (Turpin, Lanham, Chen/Anthropic)
   2.2 Mechanistic probing (Cox et al., Mirtaheri, Boppana)
   2.3 Anthropic circuit tracing (Ameisen et al. March 2025) — direct inspiration
   2.4 Reasoning horizon (Ye et al. 2026)

3. Methods (1.5 pages)
   3.1 Model and dataset
   3.2 Activation extraction with vllm-lens
   3.3 Linear probe training
   3.4 Steering protocol (pre vs mid-thinking)
   3.5 Failure mode classification

4. Experiment 1: Pre-Thinking Commitment (1.5 pages)
   — FIG-1, FIG-2, TAB-1

5. Experiment 2: Causal Steering (1.5 pages)
   — FIG-3, FIG-4, TAB-2

6. Experiment 3 & 4: Failure Modes + Reasoning Horizon (1.25 pages)
   — FIG-5, FIG-6, FIG-7, TAB-3, TAB-4

7. Discussion (0.75 page)
   7.1 Thinking as computation vs rationalisation: a difficulty-dependent answer
   7.2 Implications for CoT monitoring and AI safety
   7.3 Limitations: single model, synthetic difficulty proxy, probe linearity assumption

8. Conclusion (0.25 page)

References (~1 page)
Appendices A–E (supplementary)
```

---

## Timeline (8-week plan to submission-ready draft)

| Week | Task | Owner |
|---|---|---|
| 1 | Environment setup: vllm-lens install, OLMo-7B-Think download, MMLU/GSM8K prep | Infra |
| 2 | Exp 1 — extraction run + probe training; pilot steering on 50 questions | Core |
| 3 | Exp 2 — full pre-thinking steering; start mid-thinking two-pass | Core |
| 4 | Exp 2 — mid-thinking steering complete; position ablation curve | Core |
| 5 | Exp 3 — failure mode collection + LM-judge classification; human validation | Analysis |
| 6 | Exp 4 — trajectory extraction + commitment horizon analysis | Analysis |
| 7 | All figures + tables finalized; paper draft §1–6 | Writing |
| 8 | Internal review; §7 Discussion; appendices; code release prep | Writing |

---

## Key References

- Turpin et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS 2023.
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic.
- Chen et al. (2025). *Reasoning Models Don't Always Say What They Think.* Anthropic.
- Ameisen et al. (2025). *On the Biology of a Large Language Model.* transformer-circuits.pub.
  — **Primary inspiration:** circuit-tracing on Claude 3.5 Haiku, catches motivated reasoning.
- Cox, Kianersi & Garriga-Alonso (2026). *Decoding Answers Before Chain-of-Thought.*
  arXiv:2603.01437. — **Closest precursor:** pre-CoT probes + steering on instruction-tuned models.
- Mirtaheri & Belkin (2026). *Catching Rationalization in the Act.* arXiv:2603.17199.
- Ye et al. (2026). *Mechanistic Evidence for Faithfulness Decay.* arXiv:2602.11201.
- Hao et al. (2024). *COCONUT: Continuous Thought.* — Latent reasoning baseline.
- vllm-lens (2025). UKGovernmentBEIS. github.com/UKGovernmentBEIS/vllm-lens

---

## Novelty Summary vs. Nearest Papers

| Dimension | Cox et al. 2026 | Anthropic Circuits 2025 | **This paper** |
|---|---|---|---|
| Model type | Instruction-tuned | Closed (Claude Haiku) | **Open thinking model (OLMo-7B-Think)** |
| Intervention point | Pre-CoT only | Post-hoc attribution | **Pre + mid-thinking, ablation curve** |
| Difficulty conditioning | Not tested | Not tested | **Easy / Medium / Hard split** |
| Steering causal test | Yes (flip rate) | No (attribution only) | **Yes + position ablation** |
| Reasoning horizon | No | No | **Residual-stream trajectory** |
| Tool | Custom hooks | Proprietary | **vllm-lens (open, reproducible)** |
| Reproducibility | Partial | None (closed model) | **Full: weights + code + config** |
