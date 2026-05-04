

Good — this is a significant evolution. Let me do a full gap analysis against what we've already built.

---

## Gap Analysis: Current Code vs Unified Plan

### 🔴 Critical Conflicts

**1. Dataset accuracy is wrong in the plan**
The plan still uses MATH L4-5 (~55-65%) and BBH (~55-65%) from before we checked the HF model card. Reality:

| Dataset in plan | Actual OLMo-3-7B-Think acc | Status |
|---|---|---|
| MATH L4-5 | **95.1%** | ❌ Too saturated — almost no wrong traces |
| BBH | **86.6%** | ❌ Too saturated |
| AIME 2024+2025 | **64-72%** | ✅ Usable as secondary |
| OMEGA | **37.8%** | ✅ Best primary corpus |
| ZebraLogic | **66.5%** | ✅ Good transfer test |

The plan's entire dataset section needs updating before any experiment runs.

**2. `exp1_prethinkin_probes_p2.py` is referenced but never shared**
The plan says "migrate probe logic from `exp1_prethinkin_probes_p2.py`" — this file doesn't exist in the codebase. We need to write it from scratch.

---

### 🟡 Structural Mismatches

**3. Current `experiment1_frustration_vector.py` does the wrong experiment**
In the unified plan, **Exp 1 = pre-commitment probes** (last question token → probe final answer). Our current file does emotion direction extraction, which is now **Exp 2**. The file needs renaming and the pre-commitment probe needs to be the new Exp 1.

**4. Emotion extraction is frustration-only, plan requires 10 emotions**
Current code only builds `frustration_v`. Plan needs all 10 directions with distinct construction methods:

| Emotion | Construction method in plan | In current code? |
|---|---|---|
| frustration | `mean(wrong) - mean(correct)` | ✅ |
| confidence | `mean(correct) - mean(wrong)` | ❌ |
| confusion | `mean(long_wrong) - mean(short_wrong)` | ❌ |
| desperation | `mean(multi_restart) - mean(single_attempt)` | ❌ |
| anxiety, curiosity, boredom, satisfaction, calm, excitement | Reference corpus only | ❌ |

**5. Exp 3 interaction matrix is entirely new code (~200 lines)**
The `emotion × commitment` bidirectional analysis, regression decomposition (statsmodels OLS), and recovery odds ratios don't exist anywhere yet.

**6. Exp 4 temporal dynamics needs dual overlay**
Current `compute_temporal_dynamics()` plots frustration only. Plan needs **probe confidence + 3 emotions simultaneously** on a 4-panel figure.

**7. Exp 5 dual steering is a 2×3 factorial, current steering is 1D**
`steer_inference()` only steers emotion. Plan requires simultaneously steering the answer direction (commitment steer) as a second axis.

**8. Exp 6 Dolma attribution is not yet written**

---

### ✅ What Carries Over Cleanly

| Component | Status |
|---|---|
| nnsight `capture_batch()` | ✅ Works as-is — same API for all experiments |
| `token_level_markers()` | ✅ Carry over directly |
| `cosine_sim()`, `validate_level1/2/4/5()` | ✅ Carry over, just run in a loop over 10 emotions |
| `compute_temporal_dynamics()` | ✅ Extend to multi-emotion |
| `steer_inference()` | ✅ Extend to dual-axis |
| Reference corpus loading | ✅ Carry over |
| Figure utilities | ✅ Extend |

---

## Proposed File Structure

```
experiments/
  shared/
    capture.py          ← nnsight engine (from current exp1)
    datasets.py         ← OMEGA / AIME / ZebraLogic loaders (fixed)
    markers.py          ← verbal marker patterns + token_level_markers()
    utils.py            ← cosine_sim, mean_pool, save/load helpers

  exp1_precommitment.py ← NEW: pre-think probes, probe_auc_by_layer, per-trace labels
  exp2_emotions.py      ← RENAME+EXTEND: 10-emotion extraction + 5-level validation
  exp3_interaction.py   ← NEW: forward matrix, backward recovery, OLS regression
  exp4_temporal.py      ← EXTEND: 4-panel dual overlay
  exp5_steering.py      ← EXTEND: 2×3 factorial dual steering
  exp6_dolma.py         ← NEW: Dolma 3 attribution
  run_all.py            ← orchestrator: runs 1→6 in sequence, saves shared outputs
```

---

## Updated Compute Budget (with correct datasets)

| Hour | Task | Change from plan |
|---|---|---|
| 0–2.0 | Capture: OMEGA 400×3 + AIME 60×3 (replaces MATH L4-5) | Dataset swap |
| 2.0–3.0 | Capture: ZebraLogic 300×2 (replaces BBH) | Dataset swap |
| 3.0–3.5 | Capture: reference emotion corpus (10×30 stories) | No change |
| 3.5–4.0 | Exp 1: pre-commitment probes | No change |
| 4.0–4.5 | Exp 2: 10-emotion extraction + validation | Broader than current |
| 4.5–4.7 | Exp 3: interaction matrix + regression (CPU) | No change |
| 4.7–5.0 | Exp 4: temporal dynamics 4-panel | Minor extension |
| 5.0–7.0 | Exp 5: 2×3 dual steering sweep | New |
| 7.0–8.0 | Exp 6: Dolma attribution | No change |

---

t with the most critical missing piece, which is **`exp1_precommitment.py`**?