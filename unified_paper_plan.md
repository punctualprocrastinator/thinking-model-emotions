# Feeling the Answer Before Thinking It Through
## Pre-Commitment, Emotional Dynamics, and the Computational Role of Extended Thinking

**Venue:** NeurIPS 2026 — Main Track  
**Model:** OLMo 3-Think 7B · **Infra:** transformers + PyTorch hooks · **Budget:** ≤8 hrs A100

---

## 1. Core Thesis (Revised)

Extended thinking models produce two simultaneous internal phenomena:

1. **Pre-commitment** — the residual stream often encodes the final answer *before* thinking tokens begin
2. **Emotional dynamics** — a structured landscape of functional emotional states (frustration, confidence, anxiety, curiosity, etc.) emerges *during* thinking

> [!IMPORTANT]
> The central question is NOT "frustration = pre-commitment failure" (too reductive).
> It is: **What is the full bidirectional relationship between pre-commitment state and emotional dynamics during reasoning?**

### Bidirectional Analysis

```
Forward:   Pre-commitment state → Which emotions activate during thinking?
Backward:  Emotional dynamics   → Does thinking succeed in updating the pre-committed answer?
```

### What makes this defensible

- We test **10 emotions**, not just frustration — avoiding cherry-picking
- We use **regression decomposition** to separate pre-commitment from difficulty, length, and other confounds
- We report the full **emotion × commitment interaction matrix** — surprising nulls (e.g., curiosity being independent of commitment) are findings too

---

## 2. Research Questions

| # | Question | Experiment |
|---|---|---|
| RQ1 | Does the residual stream pre-commit to the final answer before thinking begins? | Exp 1 |
| RQ2 | Do multiple functional emotional states emerge during thinking traces? | Exp 2 |
| RQ3 | Which emotions track pre-commitment state? Which are independent? | Exp 3 (forward) |
| RQ4 | Which emotions predict whether thinking succeeds in correcting a wrong pre-commitment? | Exp 3 (backward) |
| RQ5 | What independently drives each emotion — pre-commitment failure, difficulty, or something else? | Exp 3 (regression) |
| RQ6 | Does the temporal ordering matter — does emotion emerge before or after commitment shifts? | Exp 4 |
| RQ7 | Can steering emotions and pre-commitment independently and jointly affect accuracy? | Exp 5 |
| RQ8 | Where in pretraining do these emotional representations originate? | Exp 6 |

---

## 3. Model & Infrastructure

- **Model:** OLMo 3-Think 7B (AI2) — 32 layers, 4096 hidden dim, fully open weights + training data
- **Activation capture:** PyTorch `register_forward_hook()` on `model.model.layers[i]`
- **Steering:** PyTorch forward hooks that modify `output[0]` in-place during generation
- **Hardware:** 1× A100 80GB

---

## 4. Datasets

| Dataset | N | Model accuracy | Role |
|---|---|---|---|
| MATH Level 4-5 | 350 × 3 samples | ~55-65% | Primary: contrastive pairs, probe training, steering |
| BIG-Bench Hard | 500 × 2 samples | ~55-65% | Cross-domain transfer test |
| AIME 2024+2025 | 60 × 3 samples | ~15-25% | Extreme frustration regime (qualitative) |
| Reference emotion corpus | 10 emotions × 30 stories | — | Emotion direction validation |

---

## 5. Experiments

### Exp 1 — Pre-Thinking Commitment Probes

**Goal:** Establish whether and how strongly the model pre-commits before thinking.

**Method:**
1. For each MATH problem, capture residual stream at **last question token** (before `<think>`) at layers [7, 15, 23, 31]
2. Train logistic regression probes → predict final answer from pre-think activations
3. Report AUC by layer

**Key outputs:**
- `probe_auc_by_layer`: Does pre-commitment exist? (Expected: AUC > 0.8 at layer 31)
- `pre_commitment_label` per trace: Was the pre-committed answer correct? (binary label for Exp 3)
- `probe_confidence` per trace: How confident was the pre-commitment? (continuous score for Exp 3)

**Code:** Migrate probe logic from `exp1_prethinkin_probes_p2.py` to PyTorch hooks  
**Budget:** 0.5 hr (shared capture with Exp 2)

---

### Exp 2 — Multi-Emotion Direction Extraction & Validation

**Goal:** Extract and validate directions for **10 functional emotional states**, not just frustration.

**Emotions:** frustration, confusion, confidence, anxiety, curiosity, boredom, satisfaction, desperation, calm, excitement

**Method (per emotion):**

**Step A — Task-derived directions (contrastive):**
These come from the thinking traces themselves:

| Direction | Construction | What it captures |
|---|---|---|
| `frustration_v` | `mean(wrong_traces) - mean(correct_traces)` | Failure-linked arousal |
| `confidence_v` | `mean(correct_traces) - mean(wrong_traces)` | Success-linked certainty (≈ anti-frustration) |
| `confusion_v` | `mean(long_wrong_traces) - mean(short_wrong_traces)` | Prolonged uncertainty |
| `desperation_v` | `mean(multi_restart_traces) - mean(single_attempt_traces)` | Repeated failure |

**Step B — Reference-derived directions (corpus-based):**
From the reference emotion corpus (30 stories × 10 emotions):

```python
for emotion in ["frustration", "confidence", "anxiety", "curiosity", ...]:
    acts = [forward_pass(story) for story in stories[emotion]]
    ref_vectors[emotion] = mean(acts) - global_mean
```

**Step C — Validation per direction:**
For each of the 10 directions, run the 5-level protocol:
1. **Geometry:** cos_sim to matching reference vector
2. **Verbal markers:** Does it predict emotion-specific text patterns?
3. **Confound independence:** Matrix of pairwise cos_sim between all 10 directions
4. **Cross-domain:** Does it transfer from MATH to BBH?
5. **Causal:** Does steering along it produce emotion-consistent behavior?

**Key output:** A validated set of 10 emotion direction vectors, each with confidence scores

**Code:** `experiment1_frustration_vector.py` already does this for frustration — extend the loop  
**Budget:** 1.5 hr (capture) + 0.5 hr (extraction + validation)

---

### Exp 3 — Bidirectional Interaction Analysis *(THE KEY EXPERIMENT)*

**Goal:** Map the full relationship between pre-commitment and emotional dynamics.

#### Part A — Forward: Pre-commitment → Emotions

For each trace, measure all 10 emotion activations. Group by pre-commitment state:

```python
# The full 10 × 2 interaction matrix
matrix = {}  # emotion → {"pre_correct": [...], "pre_wrong": [...]}

for trace in all_traces:
    pre_correct = probe.predict(trace.pre_think_act) == trace.gold_answer
    
    for emotion, direction in emotion_vectors.items():
        activation = np.dot(mean_pool(trace), direction)
        matrix[emotion][pre_correct].append(activation)

# For each emotion: t-test between pre_correct and pre_wrong groups
# Expected: frustration, anxiety, desperation → significantly higher when pre_wrong
#           confidence, calm → significantly higher when pre_correct  
#           curiosity, boredom → possibly INDEPENDENT of pre-commitment (interesting null!)
```

**Deliverable:** Table 1 — The 10 × 2 matrix with effect sizes and p-values

| Emotion | Pre-commit ✅ (mean ± se) | Pre-commit ❌ (mean ± se) | Cohen's d | p-value | Verdict |
|---|---|---|---|---|---|
| Frustration | ? | ? | ? | ? | Tracks failure? |
| Confidence | ? | ? | ? | ? | Tracks success? |
| Anxiety | ? | ? | ? | ? | Tracks failure? |
| Curiosity | ? | ? | ? | ? | Independent? |
| ... | | | | | |

#### Part B — Backward: Emotions → Thinking success

Among traces where pre-commitment was **wrong**, which emotions predict recovery?

```python
wrong_precommit_traces = [t for t in traces if probe_was_wrong(t)]

for emotion, direction in emotion_vectors.items():
    activations = [np.dot(mean_pool(t), direction) for t in wrong_precommit_traces]
    recovered   = [t.is_correct for t in wrong_precommit_traces]  # final answer correct?
    
    # Logistic regression: P(recovery) ~ emotion_activation
    # Does high frustration HELP or HURT recovery?
    # Does high curiosity help?
```

**Deliverable:** Table 2 — Which emotions predict recovery from wrong pre-commitment?

| Emotion | Recovery OR (odds ratio) | 95% CI | p-value | Interpretation |
|---|---|---|---|---|
| Frustration (low) | ? | ? | ? | Sweet spot helps? |
| Frustration (high) | ? | ? | ? | Panic hurts? |
| Curiosity | ? | ? | ? | Exploration helps? |
| Confidence | ? | ? | ? | False confidence hurts? |

#### Part C — Regression decomposition: What drives each emotion?

For frustration specifically, decompose its variance:

```python
# What independently predicts frustration activation?
import statsmodels.api as sm

X = pd.DataFrame({
    "pre_commit_wrong":  [int(not probe_correct(t)) for t in traces],
    "problem_difficulty": [difficulty_score(t) for t in traces],
    "trace_length":      [len(t.think_tokens) for t in traces],
    "restart_count":     [count_restarts(t) for t in traces],
    "hedge_density":     [count_hedges(t) / len(t.think_tokens) for t in traces],
})
y = [np.dot(mean_pool(t), frustration_v) for t in traces]

model = sm.OLS(y, sm.add_constant(X)).fit()
# Report β coefficients, R², VIFs
# Key question: is pre_commit_wrong significant AFTER controlling for difficulty?
```

**Deliverable:** Table 3 — Regression table for frustration (and optionally each emotion)

| Predictor | β | SE | t | p | VIF |
|---|---|---|---|---|---|
| Pre-commitment failure | ? | ? | ? | ? | ? |
| Problem difficulty | ? | ? | ? | ? | ? |
| Trace length | ? | ? | ? | ? | ? |
| Restart count | ? | ? | ? | ? | ? |
| Hedge density | ? | ? | ? | ? | ? |
| **R²** | | | | | |

**Code:** ~80 lines of new analysis code (pure numpy/statsmodels, CPU only)  
**Budget:** 0.2 hr

---

### Exp 4 — Temporal Dynamics with Dual Overlay

**Goal:** When does each emotion emerge relative to commitment shifts?

For each trace, plot **two curves** over normalized position:

1. **Probe confidence** in final answer (from Exp 1 probe applied token-by-token)
2. **Emotion activation** (dot product with each emotion direction)

```python
for trace in traces:
    for t in range(n_tokens):
        # Probe confidence at position t
        probe_conf[t] = probe.predict_proba(trace.acts[t])[correct_class]
        
        # Emotion activation at position t  
        for emotion, direction in emotion_vectors.items():
            emotion_act[emotion][t] = np.dot(trace.acts[t], direction)
```

**Key findings to look for:**

1. **Temporal ordering:** Does frustration rise BEFORE probe confidence drops? (= model "feels" failure before the representation shifts)
2. **Emotion divergence point:** At what % of thinking do correct and wrong traces diverge for each emotion?
3. **Recovery signatures:** In traces that recover from wrong pre-commitment, what does the emotion trajectory look like?

**Deliverable:** Figure 2 (signature figure) — Multi-panel temporal dynamics

```
Panel A: Probe confidence         Panel B: Frustration
    correct: ──────────               correct: ──────────
    wrong:   ──╲                      wrong:        ╱────
                ╲──────                          ╱
                                             ╱───
    
Panel C: Confidence               Panel D: Curiosity
    correct: ──────────               correct: ────╱╲────
    wrong:   ──╲                      wrong:   ────╱╲────  ← same? (independent!)
                ╲──────
```

**Code:** Extend `compute_temporal_dynamics()` in `experiment1_frustration_vector.py`  
**Budget:** 0.3 hr

---

### Exp 5 — Dual Steering: Emotion × Commitment

**Goal:** Causally test whether emotions and pre-commitment have independent effects.

**Design:** 2 × 3 factorial

| | No emotion steer | Suppress frustration (scale -1.5) | Amplify frustration (scale +2.0) |
|---|---|---|---|
| **No answer steer** | Baseline accuracy | +Xpp? | -Xpp? |
| **Steer toward wrong answer** | Y% flip rate | Higher flip? (less resistance) | Lower flip? (more resistance) |

**Plus: test top-3 emotions from Exp 3** (whichever showed strongest commitment interaction)

If curiosity is independent of commitment (Exp 3), then:
- Amplifying curiosity should NOT change flip rate (no interaction)
- But it might change trace *quality* (longer, more exploratory)

**Additional controls from P1:**
- Steer at 5 positions: [0%, 25%, 50%, 75%, 100%] of thinking
- Report flip rate curve per emotion × position

**Deliverable:** 
- Table 4: Full 2×3 accuracy/flip-rate table  
- Figure 3: Flip rate curves by position for top emotions
- Key test: Is the emotion × steering interaction significant? (2-way ANOVA)

**Code:** Extend `steer_inference()` — already uses PyTorch hooks  
**Budget:** 2.5 hrs

---

### Exp 6 — Training Data Attribution *(OLMo-unique)*

**Goal:** Where in Dolma 3 pretraining data did each emotional direction originate?

For the top-3 validated emotions, score 1M Dolma 3 chunks:

```python
for emotion, direction in top_3_emotions.items():
    scores = []
    for chunk in dolma_sample:
        act = forward_pass(chunk, layer=TARGET_LAYER)
        scores.append((chunk, np.dot(act, direction)))
    
    top_500 = sorted(scores, reverse=True)[:500]
    categorize(top_500)  # fiction, math, dialogue, news, etc.
```

**Predictions:**
- **Frustration direction:** Activated by narrative fiction depicting characters stuck/failing — NOT by mathematical error examples
- **Confidence direction:** Activated by authoritative exposition, textbook writing
- **Curiosity direction:** Activated by mystery fiction, scientific discovery narratives

**Deliverable:** Figure 5 — Stacked bar of document categories at different activation quantiles, per emotion

This test determines whether emotions are **learned from human-authored narratives** (method actor hypothesis) or from **task-relevant patterns** (functional hypothesis). Both are interesting findings.

**Code:** New script, ~200 lines  
**Budget:** 2.0 hrs

---

## 6. Compute Budget (8 hrs A100)

| Hour | Task |
|---|---|
| 0.0–1.5 | Capture: MATH L4-5 (350×3) with per-token activations + pre-think extraction |
| 1.5–2.5 | Capture: BBH (500×2) |
| 2.5–3.0 | Capture: AIME (60×3) + reference emotion corpus (10×30 stories) |
| 3.0–3.5 | Exp 1: Probe training + Exp 2: Multi-emotion extraction + validation |
| 3.5–3.7 | Exp 3: Interaction analysis (CPU — matrix, regression, backward) |
| 3.7–4.0 | Exp 4: Temporal dynamics with dual overlay |
| 4.0–6.5 | Exp 5: Dual steering sweep (emotions × commitment × positions) |
| 6.5–8.0 | Exp 6: Dolma attribution + figure generation |

---

## 7. Paper Outline (9 pages + appendices)

```
1. Introduction                                           (~1 page)
   - Thinking models: computation or rationalization?
   - Two lenses: commitment (logical) and emotion (affective)
   - Key question: what is the relationship between them?
   - OLMo uniquely enables full analysis (open weights + training data)

2. Background                                             (~1 page)
   2.1  Pre-commitment in CoT (Cox et al. 2026)
   2.2  Functional emotions in LLMs (Anthropic 2026)
   2.3  Steering vectors in reasoning (Venhoff, Ward 2025)
   2.4  OLMo 3-Think 7B

3. Methods                                                (~1.5 pages)
   3.1  Datasets: MATH L4-5, BBH, AIME
   3.2  Activation capture via PyTorch hooks
   3.3  Pre-commitment probes (logistic regression on pre-think activations)
   3.4  Multi-emotion direction extraction (contrastive + reference corpus)
   3.5  Validation protocol (5 levels)

4. Pre-Commitment Exists in Thinking Models (Exp 1)       (~0.75 page)
   - Probe AUC by layer
   - Establishes the baseline: yes, models often know the answer before thinking
   - FIG 1: AUC heatmap

5. A Structured Emotional Landscape (Exp 2)               (~1 page)
   - 10 validated emotion directions
   - Pairwise independence (confusion matrix)
   - Not all are just "difficulty" or "error" in disguise
   - FIG 2: Radar charts per emotion (validation scores)

6. The Emotion–Commitment Nexus (Exp 3 + 4)              (~2 pages)
   ** THE CORE CONTRIBUTION **
   6.1  Forward: which emotions track pre-commitment? (TABLE 1)
   6.2  Backward: which emotions predict recovery? (TABLE 2)
   6.3  Regression: what drives frustration independently? (TABLE 3)
   6.4  Temporal ordering: emotions precede commitment shifts (FIG 3 — signature figure)
   6.5  Surprising independence: curiosity doesn't track commitment

7. Causal Steering (Exp 5)                                (~1 page)
   - 2×3 factorial: emotion steer × commitment steer
   - Frustration as dual role: hurts normally, protects against adversarial steer
   - Independent emotions (curiosity) show no interaction (control)
   - FIG 4: Steering interaction plot, TABLE 4: Full factorial results

8. Training Data Origins (Exp 6)                          (~0.75 page)
   - Different emotions trace to different document types
   - Frustration → fiction; confidence → exposition
   - The method actor hypothesis: models acquire emotional repertoires from narrative
   - FIG 5: Document category breakdown per emotion

9. Discussion                                             (~0.75 page)
   - We do NOT claim subjective experience
   - "Functional emotional states" = directions that (a) align with human emotion concepts,
     (b) causally affect behavior, (c) have identifiable training origins
   - Safety: emotional dynamics during reasoning as a monitoring signal
   - Limitation: single model family, linear analysis
   
10. Conclusion                                            (~0.25 page)
```

---

## 8. Key Figures & Tables

| ID | Content | Section | Key message |
|---|---|---|---|
| FIG 1 | Probe AUC heatmap (layer × dataset) | §4 | Pre-commitment exists |
| FIG 2 | Radar charts: 10 emotion validation scores | §5 | Multiple emotions validated, not just frustration |
| FIG 3 | **Temporal dynamics (4-panel): probe confidence + 3 emotions over trace position** | §6 | **Emotions precede commitment shifts** (signature figure) |
| FIG 4 | 2×3 steering interaction plot | §7 | Emotions and commitment have independent causal effects |
| FIG 5 | Dolma document categories by emotion | §8 | Different emotions trace to different pretraining sources |
| TAB 1 | 10×2 emotion × commitment interaction matrix | §6 | Which emotions track commitment? |
| TAB 2 | Recovery prediction: emotion → P(correct \| wrong pre-commit) | §6 | Which emotions help recovery? |
| TAB 3 | Regression decomposition of frustration drivers | §6 | Pre-commitment vs difficulty vs length |
| TAB 4 | 2×3 factorial steering results | §7 | Causal independence test |

---

## 9. Anticipated Reviewer Objections & Responses

| Objection | Response |
|---|---|
| "Frustration is just difficulty" | TABLE 3 shows pre-commitment failure is significant AFTER controlling for difficulty (β₁ significant, VIF < 3) |
| "Pre-commitment is just confidence" | TABLE 1 shows confidence tracks commitment, but frustration, anxiety, desperation do too — it's a structured landscape, not one emotion |
| "Cherry-picking frustration" | We test ALL 10 emotions. Some track commitment, some don't. We report the full matrix including nulls |
| "Why not more emotions?" | 10 is the same order as Anthropic's initial battery. We prioritize depth (5-level validation each) over breadth |
| "Single model" | OLMo is the ONLY open thinking model with full training data. This is a necessary first study; multi-model is future work |
| "Linear directions can't capture emotions" | We use linear probes as a first-order approximation, consistent with Anthropic (2026) and the superposition literature. Non-linear extensions are future work |

---

## 10. Why This Is Stronger Than Either Project Alone

| Criterion | P1 alone | P2 alone | **Unified** |
|---|---|---|---|
| Core claim | "Pre-commitment exists" (known) | "Frustration exists" (novel but isolated) | **"Emotions and commitment form a structured bidirectional system"** |
| Confound handling | None for emotions | None for commitment | **Full regression decomposition** |
| Cherry-picking risk | N/A | "Why only frustration?" | **10-emotion matrix, including nulls** |
| Unique contribution | 0 (replication) | 1 (Dolma attribution) | **3+ (interaction, temporal ordering, dual steering, attribution)** |
| Scoop risk | Very high | Low | **Very low** (nobody has both lenses) |
| Reviewer reception | "Nice extension" | "Interesting but one-directional" | **"Complete mechanistic story"** |
