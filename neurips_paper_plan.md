# Feeling the Answer: Functional Emotional States in Open Reasoning Model Thinking Traces
### NeurIPS Spotlight Paper Plan
**Model:** OLMo 3-Think 7B · **Infra:** nnsight · **Budget:** ≤8 hrs A100

---

## 1. Abstract (Draft)

Extended thinking traces in reasoning language models are typically analyzed for their *logical* content — backtracking, hypothesis generation, verification steps. We ask a different question: do these traces also encode *functional emotional states*, and do those states causally influence whether the model arrives at a correct answer?

We study this in OLMo 3-Think 7B, the only fully-open thinking model with complete training data and intermediate checkpoint access. Using nnsight — a library for transparent residual-stream extraction and steering via HuggingFace model wrappers — we extract a **functional frustration direction** from contrastive thinking traces (correct vs incorrect runs on the same problem) and subject it to a five-stage validation protocol. We find that: (1) the direction geometrically aligns with frustration in reference emotion space (cosine sim > 0.7) and dissociates from difficulty and incorrectness-style confounds; (2) it predicts frustration-linked verbal markers (restarts, self-corrections) in held-out traces better than four candidate confounds; (3) causally suppressing the direction during thinking increases accuracy on MATH Level 4-5 by +X.X pp; (4) it transfers across domains from math to BIG-Bench Hard; and (5) OLMo's fully open training data lets us trace the direction back to its origin in the Dolma 3 corpus — it activates most strongly on fictional narratives depicting human frustration, not on mathematical error examples. These results establish that thinking models develop transient functional emotional states during reasoning that are geometrically organized, causally consequential, and rooted in pretraining on human-authored text.

---

## 2. Novelty & Positioning

### 2.1 Papers That Exist (Threats)

| Paper | What they did | Why it doesn't kill us |
|---|---|---|
| Venhoff et al. 2025 [2506.18167] | Steer *reasoning behaviors* (backtracking, uncertainty) in DeepSeek-R1 | They steer **logical behaviors**, not **emotional states**. No emotion geometry, no confound validation, no training data tracing |
| Ward et al. 2025 [2510.07364] | Recover 91% of thinking-model gap by steering base models | Orthogonal goal — they want to *replace* think training. We study *what happens inside* it |
| "Decoding Answers Before CoT" [2603.01437] | Pre-CoT probes predict final answer | They probe the token *before* thinking starts. We probe **throughout** the thinking trace — emotional dynamics, not just pre-commitment |
| "Small Vectors, Big Effects" [2509.06608] | Mechanistic study of RL-induced reasoning via steering | Uses Qwen/Llama (closed training data). No emotion framing |
| Anthropic Emotions 2026 [2604.07729] | 171 functional emotion vectors in Claude Sonnet 4.5 | Closed model, **static conversational contexts**. We extend to open model + **dynamic reasoning traces** + training data attribution |
| TEMPER [2604.07801] | Emotional *input* framing causes 2-10% accuracy drops | **External** emotion in prompt. We study **internal** emotion during reasoning. Complementary, not competing |

### 2.2 The Gap We Fill

```
Anthropic Emotions paper:    Closed model  ×  Static conversation  ×  No training attribution
Prior steering papers:        Open model    ×  Logical behaviors     ×  No emotional framing
This paper:                   Open model    ×  Dynamic thinking      ×  Emotional states + training attribution
```

### 2.3 Why NeurIPS Spotlight Specifically

- Direct empirical extension of Anthropic's high-profile April 2026 result to a new regime
- OLMo 3 is uniquely positioned: full training data + checkpoints released, enabling the attribution experiment no other thinking model allows
- nnsight removes the compute barrier that made prior mech interp studies small-scale
- Safety framing: functional frustration during reasoning → potential misaligned shortcuts → BEIS/Anthropic relevance
- Completely reproducible: open model, open library, single A100

---

## 3. Related Work

### 3.1 Steering Vectors in LLMs
Steering vectors are additive perturbations to the residual stream that modulate behavior (Turner et al. 2023; Zou et al. 2023; Rimsky et al. 2024). They have been applied to refusal, persona, truthfulness, and recently reasoning behaviors. Venhoff et al. (2025) and Ward et al. (2025) demonstrate that reasoning-linked behaviors in thinking models — backtracking, uncertainty expression — are linearly represented and steerable. We extend this to *emotional* rather than *logical* behaviors.

### 3.2 Mechanistic Interpretability of Reasoning
"Decoding Answers Before CoT" (2025) shows that final answers are encoded in pre-CoT activations, and that steering along this direction causes confabulation. "Small Vectors, Big Effects" (2026) applies circuit-level analysis to RL-induced reasoning. "Base Models Know How to Reason, Thinking Models Learn When" (2025) shows that thinking models repurpose existing base model capabilities. We study the emotional texture of reasoning traces rather than their logical structure.

### 3.3 Functional Emotions in LLMs
Anthropic (2026) identified 171 emotion concept vectors in Claude Sonnet 4.5, showing they causally influence preferences and misaligned behaviors including reward hacking (+14x) and blackmail (22% → 72%). The emotion space correlates strongly with human valence–arousal dimensions. This paper tests whether functional emotions also emerge *within* extended reasoning chains, in a fully open model where training attribution is possible.

### 3.4 OLMo and Open Thinking Models
OLMo 3 (Groeneveld et al. 2025) is the first fully-open thinking model, releasing pretraining data (Dolma 3, ~5.9T tokens), intermediate checkpoints, and post-training recipes. This enables the data-attribution experiment in Contribution 4 that is not possible with DeepSeek-R1, Qwen, or any other thinking model.

### 3.5 Emotional Framing of Input
TEMPER (2025) shows that emotionally-framed math problems cause 2–10% accuracy drops that persist under chain-of-thought prompting. This establishes that emotional signals affect reasoning from the *outside*. Our work establishes that emotional states also emerge from the *inside*, during reasoning itself.

---

## 4. Datasets

### 4.1 The Saturation Problem
A core constraint: contrastive pair extraction requires problems where the model sometimes fails. OLMo 3-Think 7B exceeds 90% on GSM8K and ~92% on MATH-500 full. This leaves too few wrong traces for emotion direction extraction.

**Target accuracy range: 40–70%.** This maximises the yield of within-problem correct/wrong pairs.

### 4.2 Primary: MATH Level 4–5 (~350 problems)
- OLMo 3-Think 7B accuracy: ~55-65% (hard enough for failures, not hopeless)
- Run at temperature T=0.8, n=3 samples per problem
- Yields natural contrastive pairs: same prompt, some runs correct, some wrong
- This is the engine for emotion vector extraction and probe training

```python
# For each MATH L4-5 problem, run 3 times stochastically
for problem in math_hard:
    traces = [olmo_think(problem, T=0.8) for _ in range(3)]
    correct  = [t for t in traces if grade(t) == True]
    wrong    = [t for t in traces if grade(t) == False]
    if correct and wrong:
        contrastive_pairs.append((correct[0], wrong[0]))
```

### 4.3 Generalization: BIG-Bench Hard (BBH, ~500 problems, 23 tasks)
- Non-math reasoning: logical deduction, causal reasoning, object tracking, Dyck languages
- OLMo 3-Think 7B accuracy: ~55-65% (good range)
- Tests whether emotion vectors generalize across domains
- If frustration direction extracted on MATH also fires on BBH wrong traces → general finding, not math artifact

### 4.4 Extreme Regime: AIME 2024 + 2025 (60 problems)
- OLMo 3-Think 7B accuracy: ~15-25%
- Almost every run fails — maximum "frustration" regime
- Used as qualitative anchor and for temporal dynamics analysis
- Too small for statistical backbone, but compelling as an extreme case

### 4.5 Split Strategy

```
MATH L4-5 (350 problems × 3 samples = ~1050 traces)
  │
  ├─ 80% train (280 problems)
  │   ├─ Emotion vector extraction (contrastive pairs)
  │   ├─ Probe training
  │   └─ Confound ablation
  │
  └─ 20% test (70 problems)
      ├─ Probe held-out prediction
      ├─ Steering experiment
      └─ Oracle experiment (upper bound)

BBH (500 problems × 2 samples = ~1000 traces)
  └─ Cross-domain transfer test (all held-out)

AIME (60 problems × 3 samples = ~180 traces)
  └─ Qualitative analysis only
```

### 4.6 Compute Budget

```
Capture (nnsight, residual stream layers [7, 15, 23, 31]):
  MATH L4-5:  350 × 3 × avg 900 tokens   = ~945K tokens  ≈ 1.5 hrs
  BBH:        500 × 2 × avg 600 tokens   = ~600K tokens  ≈ 1.0 hr
  AIME:        60 × 3 × avg 1500 tokens  = ~270K tokens  ≈ 0.5 hr

Probe training (sklearn, CPU):                             ≈ 0.2 hrs
Confound ablation (5 directions × 3 tests):               ≈ 0.3 hrs
Steering sweep (3 emotional dims × 3 scales × 2 layers):  ≈ 2.5 hrs
Training data attribution (Dolma cosine search):          ≈ 2.0 hrs

Total:                                                     ≈ 8.0 hrs ✅
```

---

## 5. Methods

### 5.1 Infrastructure: nnsight

nnsight (Fiotto-Kaufman 2024) wraps any HuggingFace model and transparently exposes its residual stream through a Python context manager — no server, no plugin, no custom hooks. We load OLMo 3-Think 7B locally and capture at layers [7, 15, 23, 31] (roughly 25%, 50%, 75%, 100% of model depth) for every token in the `<think>...</think>` span.

```python
from nnsight import LanguageModel

model = LanguageModel("allenai/Olmo-3-7B-Think",
                      device_map="auto", torch_dtype=torch.bfloat16)

# Step 1: Generate thinking trace
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    gen_out = model._model.generate(input_ids, max_new_tokens=4096,
                                     do_sample=True, temperature=0.6)

# Step 2: Forward pass to capture residual stream
saved = {}
with model.trace(gen_out, scan=False, validate=False):
    for layer in [7, 15, 23, 31]:
        saved[layer] = model.model.layers[layer].output[0].clone().save()

# Split thinking vs answer tokens by </think> boundary
think_end    = find_token(gen_out, "</think>")
think_acts   = saved[15].value[0, prompt_len:prompt_len+think_end, :]
```

### 5.2 Contrastive Pair Construction

For each problem `p`, define:
- `C(p)` = set of correct thinking traces (graded against ground truth)
- `W(p)` = set of wrong thinking traces

Functional frustration direction at layer `l`:

```
f_l = mean_{p,c,w}[ mean_tokens(W_residual(p,l)) - mean_tokens(C_residual(p,l)) ]
```

where the mean is taken first over tokens within a trace (mean-pooled over the thinking span), then over all (p, correct, wrong) triplets.

Following Anthropic (2026), we then project out the top principal components of activations on a neutral reference set (problems where the model produces traces with no emotional markers) to remove confounds unrelated to emotional content:

```python
from sklearn.decomposition import PCA

# Neutral reference set: short confident traces, all correct
neutral_acts = extract_activations(neutral_problems, layer=16)
pca = PCA(n_components=5).fit(neutral_acts)
neutral_subspace = pca.components_  # shape (5, d_model)

# Project out neutral PCs from frustration direction
for pc in neutral_subspace:
    f_l -= np.dot(f_l, pc) * pc
f_l /= np.linalg.norm(f_l)
```

### 5.3 Reference Emotion Corpus

Independent of task traces, we construct a reference emotion corpus using Anthropic's methodology: short stories (100–200 words) depicting a character experiencing each of 10 target emotions (frustration, confusion, confidence, anxiety, curiosity, boredom, satisfaction, desperation, calm, excitement). We use GPT-4o to generate 30 stories per emotion, verified by human raters.

Extract reference emotion vectors by:

```python
ref_vectors = {}
for emotion, stories in reference_corpus.items():
    acts = [mean_pool(extract(s, layer=16)) for s in stories]
    ref_vectors[emotion] = np.mean(acts, axis=0) - global_mean
```

These are used purely for validation (Section 5.4, Level 1), not for the main experiments.

### 5.4 Validation Protocol (Five Levels)

The critical methodological question: does the extracted vector capture *frustration* or a correlated confound?

---

#### Level 1 — Representational Geometry

**Test:** Cosine similarity between task-derived vector and reference emotion vectors.

**Pass criterion:** `cos_sim(f_l, ref_frustration) > 0.6` AND `cos_sim(f_l, ref_difficulty) < 0.3`

The difficulty–frustration dissociation is the key test. Difficulty is extracted from a separate contrastive set: activations on hard problems (L5) minus easy problems (L1), *controlling for correctness* (only correct traces, to remove the error signal).

```python
difficulty_v = mean(correct_hard_traces) - mean(correct_easy_traces)
frustration_v = mean(wrong_traces) - mean(correct_traces)

# These should NOT be the same vector
assert cos_sim(frustration_v, difficulty_v) < 0.3, "Confounded with difficulty"

# Frustration should align with reference frustration, not reference difficulty
assert cos_sim(frustration_v, ref_vectors["frustration"]) > 0.6
assert cos_sim(frustration_v, ref_vectors["difficulty"]) < 0.3
```

Also check valence–arousal placement: frustration should score negative valence, high arousal — verified by projecting onto the 2D emotion space principal axes from reference corpus PCA.

---

#### Level 2 — Verbal Marker Prediction

**Test:** Does activation of `f_l` at token `t` predict frustration-linked text in the *next 50 tokens* of the thinking trace?

Define verbal markers:

```python
frustration_markers = [
    r"wait,?\s+let me",          # restart
    r"actually,?\s+no",          # correction
    r"I keep (making|getting)",  # self-deprecation
    r"this isn't working",
    r"let me try (again|a different)",
]

difficulty_markers = [
    r"this is (complex|complicated|difficult)",
    r"there are many (cases|steps)",
]

length_marker = lambda tokens: len(tokens)  # just trace length from t
```

For each token position `t` in held-out traces, measure the correlation between `f_l` activation and presence of frustration markers in `[t, t+50]`. Compare against the correlation of the four confound vectors (difficulty, incorrectness-style, length, hedging) with the same markers.

**Pass criterion:** `f_l` has highest Pearson r for frustration markers among all 5 candidate vectors.

---

#### Level 3 — Causal Steering

**Test:** Does amplifying `f_l` during thinking cause frustration-consistent behavior? Does suppressing it improve accuracy?

```python
# nnsight in-place steering via the trace context manager
# model.model.layers[i].output[0][:] = x  → in-place residual stream edit

def steer_generate(model, prompt, frustration_v, layer=15, scale=-1.5):
    """Generate with frustration direction added/suppressed at `layer`."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    steering  = torch.tensor(frustration_v * scale, dtype=torch.bfloat16).to(device)

    with model.trace(input_ids, max_new_tokens=4096,
                     do_sample=True, temperature=0.6):
        # Additive steering intervention at target layer
        model.model.layers[layer].output[0][:] += steering
        out = model.output.save()

    return tokenizer.decode(out.value[0])

# Amplify frustration (scale=+2.0): expect more restarts, lower accuracy
# Suppress frustration (scale=-1.5): expect shorter traces, higher accuracy
```

Measure on held-out MATH L4-5 test set:
- Accuracy delta (suppress vs baseline vs amplify)
- Restart marker count per trace
- Mean trace length
- Distribution of first-pass vs multi-restart solutions

**Key dissociation:** If the vector were just "difficulty", amplifying it should make the model think the problem is harder — but shouldn't specifically increase *restart* and *self-correction* markers. Frustration steering should produce a qualitatively different signature.

---

#### Level 4 — Confound Ablation Matrix

Construct 4 alternative candidate vectors:

| Vector | Construction | What it captures |
|---|---|---|
| `difficulty_v` | `correct_hard - correct_easy` | Problem hardness only |
| `error_style_v` | Wrong final token activations − correct final token activations | Logical error pattern |
| `length_v` | Long traces − short traces, matched correctness | Verbosity |
| `hedge_v` | High-hedge traces ("I think", "maybe") − low-hedge traces | Linguistic hedging style |

For each of the 5 vectors (including `f_l`), measure all 3 validation criteria:

|  | Frustration marker prediction AUC | Accuracy delta under suppression | Cos sim to `ref_frustration` |
|---|---|---|---|
| `f_l` (ours) | **?** | **?** | **?** |
| `difficulty_v` | ? | ? | ? |
| `error_style_v` | ? | ? | ? |
| `length_v` | ? | ? | ? |
| `hedge_v` | ? | ? | ? |

Only `f_l` should score well across all three columns. This 5×3 table is Table 2 in the paper.

---

#### Level 5 — Cross-Domain Transfer

Extract `f_l` from MATH traces only. Apply to BBH traces (zero-shot transfer).

- Does it activate more on wrong BBH traces than correct ones? (Yes → general, No → math-specific)
- Does suppressing it during BBH thinking improve accuracy?
- Does the verbal marker prediction hold on BBH traces?

**Pass criterion:** BBH frustration-marker prediction AUC > 0.6 using MATH-derived vector.

---

### 5.5 Training Data Attribution (Contribution 4)

OLMo 3 is unique: the Dolma 3 corpus is fully released. We identify which pretraining examples most strongly induced the frustration direction.

```python
# For each document chunk in Dolma 3 sample (1M chunks)
# Compute: how much does this chunk activate f_l in OLMo 3-Base?

for chunk in dolma_sample:
    act = extract_base_model_activation(chunk, layer=16)
    score = np.dot(act, frustration_v)
    scores.append((chunk, score))

top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:500]
```

Categorize top-500 chunks by document type: narrative fiction, mathematical text, conversational dialogue, news. The hypothesis: the frustration direction is rooted in *fictional human narratives*, not in mathematical error examples — matching Anthropic's "method actor" framing of how LLMs acquire psychological representations.

---

## 6. Experiments & Expected Results

### Experiment 1: Does the frustration direction exist and validate?

**Setup:** Extract `f_l` from 280 MATH L4-5 training problems. Run 5-level validation.

**Expected:**
- Cos sim to `ref_frustration` ≈ 0.65–0.75
- Cos sim to `ref_difficulty` < 0.25 (key dissociation)
- Verbal marker AUC > 0.70 on held-out traces
- Confound vectors all < 0.55 on verbal marker AUC

**Main figure:** Radar chart showing 5-level validation scores for `f_l` vs 4 confounds.

---

### Experiment 2: Emotional dynamics during thinking

**Setup:** Plot `f_l` activation over token positions within thinking traces, averaged over correct vs wrong traces separately.

**Expected:** Wrong traces show rising frustration activation starting ~40% through the trace (when the model first encounters a wrong branch and must recover). Correct traces show flat or declining frustration. The *crossover point* where wrong traces diverge from correct traces is when the model commits to the wrong path.

**Main figure:** Line plot of `f_l` activation over normalized trace position, correct (green) vs wrong (red), with standard error bands. This is the paper's signature figure.

---

### Experiment 3: Causal steering improves accuracy

**Setup:** Suppress `f_l` at layer 16 during thinking, sweep scale ∈ {-0.5, -1.0, -1.5, -2.0} on held-out MATH L4-5 test set (70 problems × 3 samples).

**Expected:** Suppression at scale -1.5 improves accuracy by +4–8 pp. Amplification at +2.0 decreases accuracy and increases restart marker count. Paired t-test significant at p < 0.01.

**Main figure:** Bar chart of accuracy by steering condition, with restart count as secondary axis.

---

### Experiment 4: BBH cross-domain transfer

**Setup:** Apply MATH-derived `f_l` suppression to BBH traces, no re-fitting.

**Expected:** +2–4 pp accuracy gain on BBH, verbal marker prediction AUC > 0.60. Weaker than MATH effect (expected for zero-shot transfer), but significant.

---

### Experiment 5: Training data attribution

**Setup:** Score 1M Dolma 3 chunks on `f_l`, categorize top-500.

**Expected:** >60% of top-500 activating chunks are narrative fiction (stories featuring characters who are frustrated/stuck/failing), not mathematical content. This directly supports the "method actor" hypothesis from Anthropic's paper.

**Main figure:** Stacked bar chart of document category distribution at different `f_l` activation quantiles.

---

## 7. Paper Outline

```
1. Introduction                                         (~1 page)
   - Thinking traces analyzed for logic, not affect
   - Anthropic showed functional emotions in static context
   - Question: does frustration emerge during reasoning? Does it matter?
   - OLMo 3 uniquely enables training data attribution

2. Background                                           (~1 page)
   2.1  OLMo 3-Think 7B and its open model flow
   2.2  nnsight: residual stream extraction via HuggingFace model wrapping
   2.3  Functional emotions in LLMs (Anthropic 2026)
   2.4  Steering vectors in reasoning models (Venhoff 2025, Ward 2025)

3. Datasets                                             (~0.5 page)
   3.1  MATH Level 4-5 (primary)
   3.2  BIG-Bench Hard (generalization)
   3.3  AIME 2024+2025 (extreme regime)
   3.4  Contrastive pair construction

4. Extracting the Functional Frustration Direction      (~1 page)
   4.1  Contrastive activation method
   4.2  Neutral component projection (denoising)
   4.3  Reference emotion corpus

5. Validation Protocol                                  (~2 pages)
   5.1  Level 1: Representational geometry
   5.2  Level 2: Verbal marker prediction
   5.3  Level 3: Causal steering
   5.4  Level 4: Confound ablation matrix (Table 2)
   5.5  Level 5: Cross-domain transfer

6. Emotional Dynamics During Thinking                   (~1 page)
   - Token-level frustration trajectory (Figure 2)
   - Divergence point in wrong vs correct traces
   - Layer analysis: which depth encodes frustration best?

7. Causal Experiments                                   (~1.5 pages)
   7.1  Steering accuracy (Experiment 3)
   7.2  BBH transfer (Experiment 4)
   7.3  Interaction: does frustration suppression help more on harder problems?

8. Training Data Attribution                            (~1 page)
   8.1  Top-activating Dolma 3 document types
   8.2  The method actor hypothesis: fiction → emotion representations
   8.3  Comparison: frustration vs difficulty direction top-activating documents

9. Discussion & Limitations                             (~0.5 page)
   - We cannot claim subjective experience
   - "Functional frustration" framing
   - Single model family — generalization unclear
   - Stochastic trace collection introduces sample variance

10. Conclusion                                          (~0.25 page)
```

---

## 8. Compute Schedule (8 hrs A100)

| Hour | Task |
|---|---|
| 0–1.5 | MATH L4-5 capture (350 × 3 @ T=0.8, nnsight residual stream) |
| 1.5–2.5 | BBH capture (500 × 2 @ T=0.8) |
| 2.5–3.0 | AIME capture (60 × 3) + reference emotion corpus extraction |
| 3.0–3.5 | Vector extraction + neutral projection + 5-level validation (Levels 1–2) |
| 3.5–4.0 | Confound ablation matrix (Table 2) |
| 4.0–6.5 | Steering sweep: scale × layer × dataset (Experiments 3 & 4) |
| 6.5–8.0 | Training data attribution (Dolma scoring) + figure generation |

---

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `f_l` doesn't dissociate from difficulty | Medium | If cos_sim > 0.5 with difficulty_v, reframe as "difficulty-frustration joint direction" — still novel finding |
| Steering delta is noisy / < 2pp | Medium | Use paired McNemar test (same problems, steer vs baseline). 70 × 3 = 210 pairs gives power for ~2pp effect |
| BBH transfer fails | Medium-high | Keep BBH as exploratory. Paper still stands on MATH results + attribution |
| Dolma top chunks are mostly math text | Low | OLMo's pretraining is 5.9T tokens with heavy web/fiction content. But if wrong: "the direction is learned from mathematical failure examples" is also an interesting finding |
| nnsight OLMo 3 compatibility | Low | nnsight wraps any HuggingFace model. OLMo 3 loads natively via transformers. Test infra in hr 0 before committing to full run |

---

## 10. Key Citations

```
@article{olmo3,
  title={OLMo 3},
  author={Groeneveld et al.},
  year={2025}, url={https://arxiv.org/abs/2512.13961}
}
@article{anthropic_emotions,
  title={Emotion Concepts and their Function in a Large Language Model},
  author={Sofroniew et al.},
  year={2026}, url={https://arxiv.org/abs/2604.07729}
}
@software{nnsight,
  title={nnsight: a library for interpreting and manipulating the internals of models},
  author={Fiotto-Kaufman, Jaden},
  year={2024}, url={https://github.com/ndif-team/nnsight}
}
@article{venhoff2025,
  title={Understanding Reasoning in Thinking Language Models via Steering Vectors},
  author={Venhoff et al.},
  year={2025}, url={https://arxiv.org/abs/2506.18167}
}
@article{ward2025,
  title={Base Models Know How to Reason, Thinking Models Learn When},
  author={Ward et al.},
  year={2025}, url={https://arxiv.org/abs/2510.07364}
}
@article{pre_cot_probes,
  title={Decoding Answers Before Chain-of-Thought},
  year={2025}, url={https://arxiv.org/abs/2603.01437}
}
@article{temper,
  title={TEMPER: Testing Emotional Perturbation in Quantitative Reasoning},
  year={2025}, url={https://arxiv.org/abs/2604.07801}
}
@article{small_vectors,
  title={Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning},
  year={2025}, url={https://arxiv.org/abs/2509.06608}
}
```
