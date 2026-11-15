# Project Triathlon: Pre-Registered Experimental Protocol (Phase 0)

*Document ID: PROTOCOL-TRI-2026-P0*  
*Date: 15 November 2026*  
*Authors: Team Triathlon (Logi Vasan, Suyash Srivastav, Ananya Kasaudhan)*  
*Department of Artificial Intelligence & Machine Learning, Gati Shakti Vishwavidyalaya, Vadodara*  
*Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)*  
*Status: Pre-Registered & Frozen*

---

## 1. Research Question & Hypothesis
Can a lightweight classifier observing only eight cheap handcrafted image statistics of a degraded low-light image predict the optimal image enhancement operator from a fixed portfolio, outperforming any static single-filter baseline under a non-negative decision regret criterion?

## 2. Fixed Enhancement Bank (Appendix A)
The operator portfolio consists of exactly six fixed classical operators with zero per-instance hyperparameter tuning:
1. **Identity**: Passthrough operator ($I_{\text{out}} = I_{\text{in}}$).
2. **Gamma Correction**: Power-law intensity remapping with $\gamma = 0.5$ via a 256-bin lookup table.
3. **CLAHE**: Contrast-Limited Adaptive Histogram Equalization applied to the L-channel of the LAB color space ($\text{clipLimit}=2.0$, $\text{tileGridSize}=(8, 8)$).
4. **Histogram Equalization**: Global cumulative density function equalization on the L-channel of the LAB color space.
5. **Retinex (Single-Scale)**: Single-Scale Retinex with Gaussian blur scale $\sigma = 80.0$, computed via logarithmic difference and min-max normalized to $[0, 255]$.
6. **Bilateral Filter + Unsharp Mask**: Two-stage edge-preserving denoising and sharpening ($d=9, \sigma_{\text{color}}=75.0, \sigma_{\text{space}}=75.0$, unsharp $\sigma=1.5, \text{amount}=0.8$).

*All methods strictly execute a single float-to-uint8 round-and-clip to $[0, 255]$.*

## 3. Feature Space (Appendix A)
Exactly eight physical degradation statistics extracted strictly from the degraded input image:
1. `brightness`: Mean pixel luminance $\mu_I$.
2. `contrast`: Standard deviation of pixel intensities $\sigma_I$.
3. `entropy`: 256-bin Shannon entropy of the intensity distribution.
4. `noise`: Immerkaer noise variance estimator computed across the flattest 10% of $16 \times 16$ patches.
5. `dark_channel`: Dark Channel Prior mean ($15 \times 15$ erosion across RGB channels).
6. `edge_mean`: Mean Sobel gradient magnitude.
7. `edge_p90`: 90th percentile Sobel gradient magnitude.
8. `colourfulness`: Hasler–Süsstrunk opponent color space dispersion metric.

*Strict anti-leakage rule: No feature extractor may inspect ground-truth reference images, and any feature column containing forbidden substrings (`psnr`, `ssim`, `label`, `oracle`, `uid`, `name`, `path`, `regret`) is rejected at runtime.*

## 4. Ground-Truth Oracle & Regret Formulation
- **PSNR**: Peak Signal-to-Noise Ratio computed on 8-bit RGB color arrays, capped at 100 dB.
- **SSIM**: Structural Similarity Index (RGB, Gaussian window $\sigma = 1.5$, data range 255.0).
- **Oracle Selection**: Method yielding the maximum PSNR against the clean reference image. If two or more methods are within $0.05\text{ dB}$ PSNR of the maximum, SSIM breaks the tie.
- **Decision Regret**: Non-negative penalty in decibels:
  $$\text{Regret}(x) = \text{PSNR}_{\text{oracle}}(x) - \text{PSNR}_{\text{selected}}(x) \ge 0$$

## 5. Scene Grouping & Cross-Validation Protocol
- **Scene Grouping**: 64-bit difference hash (dHash) computed on a $9 \times 8$ grayscale thumbnail of the clean reference image. Images with Hamming distance $\le 10$ form edges in a scene adjacency graph; connected components define physical scene clusters.
- **Cross-Validation Scheme**: 5-fold `StratifiedGroupKFold` evaluated across three independent random seeds (seeds 0, 1, 2 = 15 total folds).
- **Leakage Prevention**: Code asserts disjoint scene groups between training and validation folds:
  $$\text{Groups}_{\text{train}} \cap \text{Groups}_{\text{val}} = \emptyset$$
- **Pre-processing Encapsulation**: All scaling (`StandardScaler`) is encapsulated inside `make_pipeline` fitted exclusively on training folds.

## 6. Uncertainty & Hypothesis Testing
- Primary metric: Mean Decision Regret (dB).
- Secondary metrics: Top-1 Accuracy (%), Top-2 Accuracy (%), 90th-percentile Regret (dB), Maximum Regret (dB), Headroom Closed (%).
- Uncertainty estimation: Cluster bootstrap over scene groups with $B = 2,000$ resamples (95% percentile confidence intervals). Paired model comparisons use identical bootstrap resample indices.

## 7. Sign-off & Pre-Registration Lock
- Lead Architect / Features / Models: Logi Vasan
- Methods / Scoring / Labels: Suyash Srivastav
- Data Acquisition / Scene Grouping: Ananya Kasaudhan
- Project Supervisor: Prof. Kishor P. Upla (SVNIT Surat / GSV Vadodara)
