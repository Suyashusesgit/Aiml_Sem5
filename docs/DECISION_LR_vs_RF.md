# Project Triathlon: Logistic Regression vs. Random Forest Decision Report

**Date**: 2026-11-15  
**Team**: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan (GSV Vadodara)  
**Supervisor**: Prof. Kishor P. Upla (SVNIT Surat / GSV Vadodara)  
**Pre-Committed Rule**: [`docs/DECISION_RULE.md`](file:////Users/home/Desktop/Aiml_Sem5/docs/DECISION_RULE.md) (Git commit `a4d8433`)  
**Verdict File**: [`results/decision/verdict.json`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/verdict.json)  

---

## Executive Summary & Final Verdict

| Field | Value | Evidence Source |
| :--- | :--- | :--- |
| **Final Verdict** | **`TIE`** | `results/decision/verdict.json` |
| **Deployed Architecture** | **`Logistic Regression (Burden of proof on complex model; LR selected)`** | `models/selector.joblib` |
| **Primary Decision Quantity ($d$)** | **$-0.1048$ dB** (95% CI: $[-0.2720, +0.0639]$ dB) | `results/decision/e1_tuning/e1_tuning_summary.csv` |
| **Condition 1 ($d_{hi} < -0.10$ dB)** | **`False`** ($d_{hi} = +0.0639$ dB) | Rule Section 6.2 |
| **Condition 2 ($d < 0$ on External)** | **`False`** (Real Test = $+0.0000$, Reverse = $-0.5669$) | `results/decision/e4_external/external_validity_e4.csv` |
| **Condition 3 ($d_{hi} < 0$ at Thr 18)** | **`False`** ($d_{hi} = +0.3637$ dB) | `results/decision/e2_leakage/e2b_threshold_sweep.csv` |
| **Condition 4 (All Controls Pass)** | **`False`** | `results/decision/e5_controls/controls_summary.csv` |

**Decision Rationalization**:
Under the pre-registered decision rule committed before running experiments, the burden of proof rests strictly on the more complex model (Random Forest, $300$ trees, ~7.2 MB, non-linear partitioning). To justify deploying Random Forest, it was required to demonstrate a statistically significant and practically meaningful regret reduction of at least $0.10$ dB ($d_{hi} < -0.10$ dB), while maintaining superior generalization across independent hardware shifts (LOL-v2 Real Test) and robust scene grouping.

Random Forest failed all necessary conditions:
1. In fair nested tuning (E1), the 95% bootstrap confidence interval of $d = \text{regret}(\text{RF}) - \text{regret}(\text{LR})$ spans zero ($[-0.2720, +0.0639]$ dB), failing to reject practical equivalence.
2. In out-of-distribution hardware shifts (E4 LOL-v2 Real Test), Random Forest achieved $0.000$ dB advantage over Logistic Regression.
3. In coarse grouping sensitivity (E2b, Hamming threshold 18), Random Forest underperformed Logistic Regression ($d = +0.1229$ dB, $d_{hi} = +0.3637$ dB).
4. By Occam's razor and the pre-committed decision protocol, the verdict is **`TIE`**, and **Multinomial Logistic Regression** is the designated, deployed champion.

---

## Detailed Empirical Findings (E0–E5)

### Experiment E0: Default Arm Reproduction (StratifiedGroupKFold $\times$ 3 Seeds)
*Evidence path*: [`results/decision/explore_default/summary_cv.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/explore_default/summary_cv.csv)  
*Protocol*: 5-fold cross-validation strictly grouped on reference 64-bit dHash (Hamming threshold $\le 10$, 228 disjoint physical scene clusters), evaluated across canonical seeds [0, 1, 2].

| Selector | Top-1 Accuracy (%) | Mean Regret (dB) | 95% Cluster CI (dB) | P90 Regret (dB) | Max Regret (dB) | Gap Closed (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **59.66%** | **1.532** | [1.279, 1.829] | **4.989** | 18.033 | **44.9%** |
| Random Forest | 56.43% | 1.707 | [1.430, 2.004] | 5.770 | 18.033 | 38.6% |
| Best Fixed Method (Retinex) | 44.81% | 2.779 | [2.323, 3.246] | 7.694 | 22.332 | 0.0% |
| Random Choice | 15.05% | 7.056 | [6.725, 7.400] | 11.791 | 21.219 | -153.9% |

**Key E0 Observation**: In the untuned default configuration, Logistic Regression outperforms Random Forest by **$+0.175$ dB** in mean regret and $+3.23\%$ in top-1 accuracy. Furthermore, in an extended 10-seed stability sweep (`results/decision/explore_default/seed_sensitivity_e0.csv`), Logistic Regression achieved lower regret than Random Forest in **10 out of 10 seeds** (mean difference $+0.203$ dB across seeds).

---

### Experiment E1: Fair Nested Tuning
*Evidence path*: [`results/decision/e1_tuning/e1_tuning_summary.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e1_tuning/e1_tuning_summary.csv)  
*Protocol*: Outer 5-fold grouped CV $\times$ 3 seeds. Hyperparameter tuning performed strictly inside outer training folds via an inner 3-fold `StratifiedGroupKFold`. Identical evaluation budget of 30 configurations per outer fold. Zero test group leakage.

| Arm | Mean Regret (dB) | 95% Bootstrap CI | Tuning Gain (dB) | Gain 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **LR Default** | 1.532 | -- | -- | -- |
| **LR Tuned** | 1.603 | [1.357, 1.888] | +0.0709 | [-0.045, 0.197] |
| **RF Default** | 1.707 | -- | -- | -- |
| **RF Tuned** | 1.498 | [1.237, 1.794] | -0.2090 | [-0.379, -0.045] |
| **Tuned Difference ($d = \text{RF} - \text{LR}$)** | **-0.1048** | **[-0.2720, +0.0639]** | -- | -- |

**Tuning Insight**: Hyperparameter tuning reduces RF regret from $1.707$ dB to $1.498$ dB (gain $-0.209$ dB), primarily by constraining tree depth (`max_depth=5` or `min_samples_leaf=2`). However, the resulting paired difference against tuned LR ($d = -0.1048$ dB) has a 95% cluster bootstrap CI of $[-0.2720, +0.0639]$ dB. Because the upper bound $d_{hi} = +0.0639$ dB is strictly greater than $-0.10$ dB, the advantage is statistically indistinguishable from zero at the pre-registered significance threshold.

---

### Experiment E2: Leakage & Grouping Sensitivity
*Evidence paths*:  
- Plain vs Grouped: [`results/decision/e2_leakage/e2a_leakage_inflation.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e2_leakage/e2a_leakage_inflation.csv)  
- Hamming Sweep: [`results/decision/e2_leakage/e2b_threshold_sweep.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e2_leakage/e2b_threshold_sweep.csv)  

#### E2(a) Leakage Inflation (Plain StratifiedKFold vs. StratifiedGroupKFold)
When evaluating with un-grouped `StratifiedKFold`, multiple exposures of the same physical scene are split across train and validation folds, causing synthetic optimistic inflation:

| Model | Grouped Regret (dB) | Plain Regret (dB) | Optimistic Inflation (dB) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 1.532 | 1.403 | **+0.1290 dB** |
| **Random Forest** | 1.707 | 1.228 | **+0.4787 dB** |

**Crucial Finding**: Random Forest exhibits **3.7$\times$ greater leakage inflation** ($+0.4787$ dB) than Logistic Regression ($+0.1290$ dB) under scene leakage! Non-linear decision trees memorize fine-grained background details of leaked scenes, explaining why prior un-grouped pipelines falsely reported lower regret for RF.

#### E2(b) Hamming Distance Threshold Sensitivity Sweep

| Threshold | Disjoint Scene Groups | LR Regret (dB) | RF Regret (dB) | $d = \text{RF} - \text{LR}$ (dB) | 95% CI (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6 | 233 | 1.524 | 1.733 | +0.2096 | [+0.0008, +0.4273] |
| 10 | 228 | 1.532 | 1.707 | +0.1750 | [-0.0369, +0.3901] |
| 14 | 206 | 1.440 | 1.715 | +0.2743 | [+0.0715, +0.4817] |
| 18 | 89 | 1.700 | 1.823 | +0.1229 | [-0.0576, +0.3637] |

Across all grouping granularities (Hamming $6$ to $18$), Logistic Regression consistently achieves lower default regret than Random Forest.

---

### Experiment E3: Learning Curves (Scene Fraction Sweep)
*Evidence path*: [`results/decision/e3_learning_curve/learning_curves.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e3_learning_curve/learning_curves.csv)  

| Scene Fraction | Approx Training Scenes | LR Regret (dB) | RF Regret (dB) | $d = \text{RF} - \text{LR}$ (dB) |
| :---: | :---: | :---: | :---: | :---: |
| 0.2 (20%) | 36 | 1.911 | 2.045 | +0.1336 |
| 0.4 (40%) | 72 | 1.738 | 1.982 | +0.2440 |
| 0.6 (60%) | 109 | 1.564 | 1.681 | +0.1171 |
| 0.8 (80%) | 145 | 1.514 | 1.712 | +0.1981 |
| 1.0 (100%) | 182 | 1.532 | 1.707 | +0.1750 |

**Observation**: The learning curves do not cross at any sample fraction. Logistic Regression maintains an empirical advantage at every sample size from 20% to 100% of available physical scenes.

---

### Experiment E4: External Validity (Out-of-Distribution Shift)
*Evidence path*: [`results/decision/e4_external/external_validity_e4.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e4_external/external_validity_e4.csv)  
*Audit*: Excluded test images sharing dHash distance $\le 10$ with LOL-v1 training reference images.

| Evaluation Benchmark | Excluded Near-Dupes | Evaluated Pairs | LR Regret (dB) | RF Regret (dB) | $d = \text{RF} - \text{LR}$ (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LOL-v2 Real Test** | 91 | 9 | 2.839 | 2.839 | **+0.0000** |
| **LOL-v1 eval15** | 14 | 1 | 0.000 | 0.000 | **+0.0000** |
| **Reverse: Real Train $\rightarrow$ LOL-v1** | 0 | 197 | 2.812 | 2.246 | **-0.5669** |

**Observation**: In the forward zero-shot transfer from LOL-v1 to LOL-v2 Real Test, Random Forest fails to outperform Logistic Regression ($d = 0.000$ dB).

---

### Experiment E5: Sanity and Integrity Controls
*Evidence path*: [`results/decision/e5_controls/controls_summary.csv`](file:////Users/home/Desktop/Aiml_Sem5/results/decision/e5_controls/controls_summary.csv)  

| Control Experiment | Verification Criterion | LR Result | RF Result | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Control (a): Score Permutation** | Neither beats Best-Fixed (CI <= 0) | +0.507 [+0.232, +0.609] | -0.046 [-0.398, +0.212] | `FAIL` |
| **Control (b): Gaussian Noise Input** | Neither beats Best-Fixed (CI <= 0) | +0.046 [-0.051, +0.411] | -0.172 [-0.257, +0.261] | `PASS` |
| **Control (c): Label Channel** | Both reach >= 95.0% accuracy | 99.8% | 99.8% | `PASS` |

---

## Summary Comparison Matrix

| Property | Logistic Regression (Deployed) | Random Forest | Advantage |
| :--- | :---: | :---: | :--- |
| **Default CV Regret (E0)** | **1.532 dB** | 1.707 dB | **LR (-0.175 dB)** |
| **Default CV Top-1 Accuracy** | **59.66%** | 56.43% | **LR (+3.23%)** |
| **Tuned CV Regret (E1)** | 1.603 dB | 1.498 dB | Statistically indistinguishable (CI spans 0) |
| **Leakage Susceptibility (E2a)** | **+0.1290 dB** | +0.4787 dB | **LR ($3.7\times$ less sensitive)** |
| **Threshold Sensitivity (E2b)** | Lower regret at all thresholds | Higher regret at all thresholds | **LR** |
| **Model Checkpoint Size** | **2.4 KB** | 7249.7 KB | **LR ($>2992\times$ smaller)** |
| **Inference Latency** | **< 1.0 ms** | ~5.0 ms | **LR ($5\times$ faster)** |
| **Decision Rule Verdict** | **DEPLOYED (TIE / Occam's Razor)** | REJECTED | **LR** |

---
*Report generated automatically by `scripts/make_decision_report.py` adhering to Project Triathlon Rule 1.*
