# Project Triathlon: Standard Operating Procedure (SOP)
## Phase 0 Experimental Protocol & Data Integrity Rules

**Team Triathlon**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (Gati Shakti Vishwavidyalaya, Vadodara)  
**Supervisor**: Prof. Kishor P. Upla (SVNIT Surat)  
**Milestone**: Phase 0 Benchmark Execution Protocol  
**Version**: 1.0 (November 2026)  

---

## 1. Scope & Objective
This SOP establishes mandatory protocols for data ingestion, feature extraction, cross-validation, and performance reporting to ensure 100% scientific reproducibility and prevent all forms of data leakage.

---

## 2. Preprocessing & Input Standardization Contract
1. **Long-Side Dimension**: All images must be resized such that their longest dimension is exactly **512 pixels** using `cv2.INTER_AREA`.
2. **Aspect Ratio**: The original aspect ratio must be strictly preserved.
3. **Format**: All inputs and intermediate caches must use lossless PNG or raw NumPy matrices.
4. **Color Format**: OpenCV reads in BGR format; Matplotlib requires RGB format. Conversions must be explicitly applied prior to display or plotting.
5. **Data Range**: All PSNR and SSIM calculations must specify `max_val = 255.0` (or `data_range = 255`) for 8-bit uint8 images.

---

## 3. Data Leakage Prevention Rules (The Three Zeroes)
1. **Zero Scene Leakage**:
   - The LOL dataset contains multiple exposure variants of the same physical scene.
   - Standard random train/test splits cause near-duplicate scene leakage.
   - **Rule**: Data must be split strictly using `GroupKFold` grouped by physical scene ID (via perceptual hash or scene clustering).
2. **Zero Preprocessing Leakage**:
   - Data scalers (`StandardScaler`) must **never** be fitted on the entire dataset.
   - **Rule**: The scaler must be fitted strictly inside the training fold within an encapsulated scikit-learn pipeline (`make_pipeline(StandardScaler(), Model())`).
3. **Zero Label Leakage**:
   - Features must be extracted **strictly on the degraded input image**, without access to reference images or candidate enhancement outputs.

---

## 4. Oracle Ground Truth & SSIM Tie-Breaking Protocol
1. Compute PSNR for all 6 methods against clean reference $y$.
2. Candidate methods within $\delta = 0.05\text{ dB}$ of the highest PSNR enter the tie-break candidate set $\mathcal{A}_{\text{top}}$.
3. The final Oracle label is selected as:
   $$a^*(x) = \arg\max_{a \in \mathcal{A}_{\text{top}}} \text{SSIM}(a(x), y)$$
4. Decision Regret is strictly non-negative:
   $$r(x, \pi) \triangleq \max\left(0.0, s_{a^*}(x) - s_{\pi(x)}(x)\right)$$

---

## 5. Statistical Reliability Protocol
1. All models must be evaluated across at least 3 random seeds ($s \in \{42, 123, 999\}$) over 5 folds, yielding $N=15$ independent runs.
2. Reports must include Mean $\pm$ Standard Deviation for both Accuracy (%) and Mean Regret (dB).
3. Statistical significance must be reported using paired two-tailed $t$-tests and Wilcoxon signed-rank tests with $95\%$ confidence intervals.
4. Model hyperparameters and method bank parameters must be frozen prior to evaluation on the official held-out `eval15` test set.

---

## 6. Execution Checklist
- [x] Dataset SHA-256 verified against `419fddc4...`
- [x] Feature cache generated at `results/features.csv`
- [x] Score cache generated at `results/scores.csv`
- [x] 15-run GroupKFold benchmark completed at `results/metrics_table.csv`
- [x] Statistical tests saved at `results/statistical_tests.csv`
- [x] Publication figures generated in `results/figures/` (Fig 1 through Fig 5)
- [x] Unit test suite passing (`make test`)

