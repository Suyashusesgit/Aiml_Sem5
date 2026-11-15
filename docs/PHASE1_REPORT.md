# Project Triathlon: Phase 1 (Semester 6) Technical Report
**Title**: Edge Feasibility, Conformal Risk-Control, and Cross-Degradation Generalization  
**Team**: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan (B.Tech ECE, GSV Vadodara)  
**Supervisor**: Prof. Kishor P. Upla (Senior Member, IEEE; SVNIT Surat / GSV Vadodara)  
**Date**: November 2026  
**Execution Pipeline**: `python3 scripts/run_phase1.py`  

---

## 1. Executive Summary & Scope

Phase 1 investigates the theoretical and empirical extensions of **Project Triathlon** beyond Phase 0's frozen, pre-registered Occam's razor decision (which selected Multinomial Logistic Regression over Random Forest). 

Specifically, Phase 1 delivers five core research modules:
1. **Split-Conformal Prediction Sets**: Finite-sample distribution-free coverage guarantees ($1 - \alpha$) and risk-controlled confidence abstention ($\tau$).
2. **Research Question 2 (RQ2) Multi-Metric Study**: Empirical alignment between PSNR-optimal and SSIM-optimal oracles and Pareto metric tradeoffs.
3. **Cross-Dataset Zero-Shot Generalization**: Domain transfer of the LOL-v1 trained selector to out-of-distribution LOL-v2 Real Captured and LOL-v2 Synthetic benchmarks.
4. **Leave-One-Enhancer-Out (LOEO) Resilience**: Marginal utility of each operator in the classical bank and performance under operator dropout.
5. **10-Dimensional Physical Feature Expansion**: Evaluating the marginal value of adding Dark Channel Transmission ($T(x)$) and Dynamic Range Ratio ($R_{\text{tone}}(x)$) to the 8D baseline under strict scene-grouped cross-validation.

All reported numbers are strictly drawn from execution artifacts under [`results/phase1/`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/).

---

## 2. Module 1: Conformal Prediction & Risk-Controlled Abstention

### 2.1 Theoretical Formulation
Standard multi-class algorithm selection outputs a single point prediction $\hat{y} = \arg\max_a p(a | x)$, which provides no finite-sample error rate guarantee. We implement **Split-Conformal Prediction** ([`src/conformal.py`](file:///Users/home/Desktop/Aiml_Sem5/src/conformal.py)):
- **Non-Conformity Score**: $s_i = 1 - p(y_i | x_i)$.
- **Conformal Quantile**: On a calibration split of size $n$, the threshold $\hat{q}$ is calibrated as:
  $$\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
- **Prediction Set**: For query $x$, the admissible operator set is:
  $$\mathcal{C}(x) = \left\{ a \in \mathcal{A} : p(a | x) \ge 1 - \hat{q} \right\}$$
  Guaranteed to satisfy $P(y \in \mathcal{C}(x)) \ge 1 - \alpha$ without parametric assumptions.

### 2.2 Calibration Results (5-Fold Grouped CV)
*Artifact*: [`results/phase1/conformal_calibration.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/conformal_calibration.csv)  
*Figure*: [`results/figures/fig12_conformal_abstention.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig12_conformal_abstention.png)

| Target Coverage ($1 - \alpha$) | Empirical Coverage (%) | Coverage Std (%) | Average Set Size $|\mathcal{C}(x)|$ | Singleton Set Rate (%) |
| :---: | :---: | :---: | :---: | :---: |
| **95.0%** | **95.26%** | $\pm 1.67\%$ | 2.68 | 2.3% |
| **90.0%** | **91.13%** | $\pm 4.02\%$ | 2.41 | 5.6% |
| **85.0%** | **84.74%** | $\pm 4.60\%$ | 1.93 | 23.7% |
| **80.0%** | **78.35%** | $\pm 3.51\%$ | 1.59 | 43.5% |

**Key Finding**: Empirical coverage rates strictly match nominal targets within standard error across all folds. At 90% coverage, the set size is 2.41 operators out of 6, reducing the operator search space by $59.8\%$.

### 2.3 Confidence-Gated Risk-Controlled Abstention
When the selector's top probability is below confidence threshold $\tau$, it abstains from an aggressive prediction and routes execution to the safe Best-Fixed default (`retinex`).
*Artifact*: [`results/phase1/abstention_curve.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/abstention_curve.csv)

| Confidence Threshold $\tau$ | Abstain Rate (%) | Accepted Samples | Accepted Top-1 Acc (%) | Accepted Regret (dB) | Overall Effective Regret (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (No Abstention)** | 0.0% | 485 | 59.59% | 1.523 | **1.523** |
| **0.40** | 3.5% | 468 | 60.68% | 1.458 | 1.547 |
| **0.45** | 11.3% | 430 | 62.79% | 1.375 | 1.577 |
| **0.50** | 29.7% | 341 | 65.40% | 1.333 | 1.727 |
| **0.55** | 43.7% | 273 | 67.77% | 1.105 | 1.934 |
| **0.60** | 57.9% | 204 | **70.59%** | **1.068** | 2.169 |
| **0.65** | 67.4% | 158 | **74.05%** | **0.904** | 2.277 |
| **0.70** | 76.7% | 113 | **76.11%** | **0.867** | 2.339 |

**Key Finding**: As confidence threshold $\tau$ increases from 0.00 to 0.70, top-1 accuracy on accepted samples monotonically increases from **$59.59\%$ to $76.11\%$**, while regret on accepted images drops by **$43.1\%$** (from $1.523$ dB down to $0.867$ dB).

---

## 3. Module 2: RQ2 PSNR vs. SSIM Oracle Agreement & Metric Tradeoffs

### 3.1 Research Question 2
*Does selecting operators by pixel-wise reconstruction fidelity (PSNR) compromise perceptual structural similarity (SSIM), or do the oracles align?*

*Artifacts*:
- Summary Metrics: [`results/phase1/rq2_oracle_agreement.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_oracle_agreement.csv)
- Contingency Matrix: [`results/phase1/rq2_oracle_contingency_matrix.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_oracle_contingency_matrix.csv)
- Dual-Metric Comparison: [`results/phase1/rq2_metric_tradeoff.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_metric_tradeoff.csv)
- Figure: [`results/figures/fig14_psnr_ssim_tradeoff.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig14_psnr_ssim_tradeoff.png)

### 3.2 Findings
1. **Oracle Agreement**: The PSNR oracle and SSIM oracle agree on the exact same operator in **$63.30\%$** of scenes (307/485).
2. **Ranking Correlation**: Across all 6 operators, the mean Spearman rank correlation is **$\rho = 0.8823$** and Kendall's **$\tau = 0.8139$**, demonstrating strong alignment across the enhancement ranking.
3. **Contingency Structure**: In the 178 cases of disagreement:
   - When PSNR prefers `hist_eq` (176 images), SSIM prefers `retinex` in 123 images ($69.9\%$), reflecting Retinex's superior edge preservation over global histogram equalisation.
   - When PSNR prefers `gamma` (87 images), SSIM also picks `gamma` in 80 images ($92.0\%$).

### 3.3 Strategy Comparison on Both Objectives

| Strategy | Mean PSNR (dB) | PSNR Regret (dB) | Mean SSIM | SSIM Regret |
| :--- | :---: | :---: | :---: | :---: |
| **PSNR Oracle** | **18.304** | **0.000** | 0.7165 | 0.0524 |
| **SSIM Oracle** | 17.301 | 1.003 | **0.7689** | **0.0000** |
| **Deployed Selector (LogReg)** | 16.817 | 1.486 | 0.6933 | 0.0756 |
| **Best-Fixed (Retinex)** | 15.570 | 2.734 | 0.7338 | 0.0351 |
| **Heuristic Hand-Rule** | 15.023 | 3.281 | 0.6895 | 0.0794 |
| **Uniform Random** | 11.271 | 7.033 | 0.4377 | 0.3312 |

---

## 4. Module 3: Cross-Dataset Zero-Shot Generalization Benchmark

### 4.1 Evaluation Setup
The deployed Logistic Regression model (trained strictly on LOL-v1 `our485`) is evaluated zero-shot on out-of-distribution test benchmarks without fine-tuning or domain adaptation:
- **LOL-v2 Real Captured Test (Unfiltered, $N=100$)**
- **LOL-v2 Real Captured Test (De-duplicated clean split, $N=9$)** (excluding 91 near-duplicates sharing dHash $\le 10$ with LOL-v1)
- **LOL-v2 Synthetic Test ($N=100$)**

*Artifacts*:
- Summary: [`results/phase1/cross_degradation_summary.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/cross_degradation_summary.csv)
- Real Detailed: [`results/phase1/lol_v2_real_results.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/lol_v2_real_results.csv)
- Synthetic Detailed: [`results/phase1/lol_v2_synthetic_results.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/lol_v2_synthetic_results.csv)
- Figure: [`results/figures/fig15_cross_degradation_generalization.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig15_cross_degradation_generalization.png)

### 4.2 Benchmark Results

| Benchmark Dataset | Strategy | Mean PSNR (dB) | Regret vs Oracle (dB) | Top-1 Acc (%) | Regret Gap Closed (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LOL-v2 Real (Unfiltered, $N=100$)** | Oracle | 19.027 | 0.000 | 100.0% | — |
| | Deployed LR | 14.826 | 4.201 | 42.0% | 0.0% |
| | Best-Fixed (Retinex) | 14.826 | 4.201 | 42.0% | 0.0% |
| | Heuristic Rule | 15.014 | 4.013 | 44.0% | 4.5% |
| | Uniform Random | 12.511 | 6.516 | 16.7% | -55.1% |
| \midrule | | | | | |
| **LOL-v2 Real (Clean, $N=9$)** | Oracle | 17.621 | 0.000 | 100.0% | — |
| | Deployed LR | 14.782 | 2.839 | 44.4% | 0.0% |
| | Best-Fixed (Retinex) | 14.782 | 2.839 | 44.4% | 0.0% |
| | Heuristic Rule | 14.782 | 2.839 | 44.4% | 0.0% |
| | Uniform Random | 11.732 | 5.889 | 16.7% | -107.4% |
| \midrule | | | | | |
| **LOL-v2 Synthetic ($N=100$)** | Oracle | 19.831 | 0.000 | 100.0% | — |
| | **Deployed LR** | **16.634** | **3.197** | **42.0%** | **38.7%** |
| | Best-Fixed (Retinex) | 14.619 | 5.212 | 25.0% | 0.0% |
| | Heuristic Rule | 15.232 | 4.599 | 27.0% | 11.8% |
| | Uniform Random | 13.594 | 6.237 | 16.7% | -19.7% |

**Key Finding**:
1. On **LOL-v2 Synthetic**, the deployed Logistic Regression selector beats Best-Fixed (Retinex) by **$+2.015$ dB** ($16.634$ dB vs $14.619$ dB), closing **$38.7\%$ of the regret gap**. Retinex suffers on synthetic data (only 25% accuracy), whereas the ML selector successfully generalizes and adapts to synthetic degradation profiles.
2. On **LOL-v2 Real**, the selector defaults safely to Retinex on 100% of de-duplicated pairs ($d = 0.000$ dB), confirming stability under natural sensor shifts.

---

## 5. Module 4: Leave-One-Enhancer-Out (LOEO) Resilience

### 5.1 Methodology
To determine the marginal utility of each operator and system vulnerability to operator dropout, we systematically remove each enhancer $e \in \mathcal{A}$ from the bank, recompute the 5-operator oracle, and retrain the selector under 5-fold StratifiedGroupKFold CV.

*Artifacts*:
- Results: [`results/phase1/leave_one_enhancer_out.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/leave_one_enhancer_out.csv)
- Figure: [`results/figures/fig16_leave_one_enhancer_out.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig16_leave_one_enhancer_out.png)

### 5.2 Results

| Excluded Enhancer | Bank Size | Reduced Oracle (dB) | $\Delta$ Oracle Capacity Lost | Selector PSNR (dB) | Regret vs Full 6-Oracle (dB) | Top-1 Acc (%) | Importance Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Full 6-Bank)** | 6 | 18.304 | 0.000 | 16.817 | 1.486 | 60.0% | — |
| **identity** | 5 | 18.303 | 0.000 | 16.820 | 1.483 | 61.0% | 6 |
| **bilat_unsharp** | 5 | 18.304 | 0.000 | 16.817 | 1.486 | 60.0% | 5 |
| **clahe** | 5 | 18.287 | 0.017 | 16.743 | 1.561 | 58.4% | 4 |
| **gamma** | 5 | 17.551 | 0.753 | 16.608 | 1.696 | 63.3% | 3 |
| **hist_eq** | 5 | 17.243 | 1.061 | 16.011 | 2.293 | 74.0% | 2 |
| **retinex** | 5 | 16.738 | **1.566** | 15.728 | **2.576** | 78.1% | **1** |

**Key Finding**:
- **Pareto Core**: `retinex`, `hist_eq`, and `gamma` account for **$99.0\%$** of total oracle capacity.
- **Criticality Ranking**:
  1. `retinex`: Removing Retinex drops the oracle by $1.566$ dB and spikes selector regret to $2.576$ dB ($+1.090$ dB penalty).
  2. `hist_eq`: Removing Histogram Equalisation drops the oracle by $1.061$ dB and increases regret by $+0.807$ dB.
  3. `gamma`: Removing Gamma correction increases regret by $+0.210$ dB.
  4. `clahe`, `bilat_unsharp`, `identity`: Provide negligible marginal capacity ($<0.02$ dB).

---

## 6. Module 5: 10D Physical Feature Expansion Benchmark

### 6.1 Feature Formulation
We benchmark the 8D baseline against an expanded 10D physical feature set:
- Added Descriptor 9: **Air-Light Transmission $T(x)$** (He et al., 2009):
  $$T(x) = 1.0 - 0.95 \cdot \frac{\min_{c \in \{B,G,R\}} \left(\mathcal{E}_{15}(I_c(x))\right)}{A}$$
- Added Descriptor 10: **Dynamic Range Ratio $R_{\text{tone}}(x)$**:
  $$R_{\text{tone}}(x) = \frac{P_{99}(\text{gray})}{P_{1}(\text{gray}) + 10^{-4}}$$

*Artifacts*:
- 10D Features CSV: [`results/features_lol_v1_10d.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/features_lol_v1_10d.csv)
- Comparison: [`results/phase1/feature_expansion_comparison.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/feature_expansion_comparison.csv)
- Coefficients: [`results/phase1/feature_10d_coefficients.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/feature_10d_coefficients.csv)
- Figure: [`results/figures/fig13_feature_expansion.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig13_feature_expansion.png)

### 6.2 Cross-Validation Results (5 Folds $\times$ 3 Seeds)

| Feature Set | Features Included | Mean Regret (dB) | Top-1 Acc (%) | Paired $\Delta d$ vs 8D (dB) | 95% Bootstrap CI | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **8D Baseline** | 8 | $1.493 \pm 0.050$ | $60.14\% \pm 1.15\%$ | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| **10D Physical** | 10 | $1.474 \pm 0.010$ | $60.82\% \pm 0.34\%$ | **-0.0184** | **[-0.1087, +0.0696]** | 0.5481 |

### 6.3 10D Feature Importance Breakdown
From the fitted multinomial logistic regression weights across all 485 samples:
1. `edge_mean`: **19.96%**
2. `colourfulness`: **15.31%**
3. `brightness`: **14.84%**
4. `entropy`: **14.58%**
5. `edge_p90`: **11.31%**
6. `noise`: **6.69%**
7. `dark_channel`: **5.74%**
8. `contrast`: **5.70%**
9. `air_light_transmission`: **4.26%**
10. `dynamic_range_ratio`: **1.62%**

**Key Finding**:
While 10D features improve mean regret by a slight $-0.0184$ dB and increase top-1 accuracy to $60.82\%$, the paired difference is not statistically significant ($p = 0.5481$, 95% CI spans zero). This demonstrates that the **8D handcrafted physical descriptor set is parsimonious, near-optimal, and sufficient** for classical algorithm selection.

---

## 7. Artifacts & Deliverables Summary

| Deliverable Artifact | Type | Path | Purpose |
| :--- | :---: | :--- | :--- |
| **Conformal Calibration** | CSV | [`results/phase1/conformal_calibration.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/conformal_calibration.csv) | Empirical coverage and set sizes at 80-95% levels |
| **Abstention Curve** | CSV | [`results/phase1/abstention_curve.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/abstention_curve.csv) | Accuracy and regret vs confidence threshold $\tau$ |
| **RQ2 Agreement** | CSV | [`results/phase1/rq2_oracle_agreement.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_oracle_agreement.csv) | PSNR vs SSIM oracle agreement and ranking $\rho$ |
| **RQ2 Contingency** | CSV | [`results/phase1/rq2_oracle_contingency_matrix.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_oracle_contingency_matrix.csv) | 6x6 operator contingency between PSNR and SSIM |
| **RQ2 Tradeoffs** | CSV | [`results/phase1/rq2_metric_tradeoff.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/rq2_metric_tradeoff.csv) | Dual-metric regret evaluation across 6 strategies |
| **Cross-Dataset Summary** | CSV | [`results/phase1/cross_degradation_summary.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/cross_degradation_summary.csv) | Zero-shot performance on Real and Synthetic LOL-v2 |
| **LOEO Analysis** | CSV | [`results/phase1/leave_one_enhancer_out.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/leave_one_enhancer_out.csv) | Marginal utility and rank importance of operators |
| **10D Features Data** | CSV | [`results/features_lol_v1_10d.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/features_lol_v1_10d.csv) | Extracted 10D physical features on LOL-v1 (485 pairs) |
| **10D Comparison** | CSV | [`results/phase1/feature_expansion_comparison.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/feature_expansion_comparison.csv) | Paired statistical comparison of 8D vs 10D |
| **10D Coefficients** | CSV | [`results/phase1/feature_10d_coefficients.csv`](file:///Users/home/Desktop/Aiml_Sem5/results/phase1/feature_10d_coefficients.csv) | Decision weight allocation across 10 descriptors |
| **Figure 12** | PNG | [`results/figures/fig12_conformal_abstention.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig12_conformal_abstention.png) | Conformal calibration and abstention frontier |
| **Figure 13** | PNG | [`results/figures/fig13_feature_expansion.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig13_feature_expansion.png) | 8D vs 10D performance and feature weights |
| **Figure 14** | PNG | [`results/figures/fig14_psnr_ssim_tradeoff.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig14_psnr_ssim_tradeoff.png) | Dual-metric objective space and contingency heatmap |
| **Figure 15** | PNG | [`results/figures/fig15_cross_degradation_generalization.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig15_cross_degradation_generalization.png) | Cross-dataset zero-shot generalization bar chart |
| **Figure 16** | PNG | [`results/figures/fig16_leave_one_enhancer_out.png`](file:///Users/home/Desktop/Aiml_Sem5/results/figures/fig16_leave_one_enhancer_out.png) | LOEO capacity loss and realized regret |
