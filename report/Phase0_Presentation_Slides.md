# Fidelity-Driven Algorithm Selection for Low-Light Image Enhancement
## Handcrafted Physical Statistics for Per-Image Enhancer Optimization

**Team Triathlon**: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan (Gati Shakti Vishwavidyalaya, Vadodara)  
**Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE* (SVNIT Surat)  
**Milestone**: 5th Semester B.Tech Capstone Project (Phase 0 Defense)  
**Date**: November 2026  

---

## Slide 1: The Core Dilemma in Image Restoration

- **The Problem**: Low-light imaging suffers from irreversible photon shot noise, contrast loss, and color degradation ($g(x) = \mathcal{H}[f(x)] + n(x)$). Inversion is mathematically ill-posed.
- **The Classical Fallacy**: Vision systems deploy a single fixed enhancement filter (e.g., Retinex or CLAHE) across all images.
- **The Empirical Reality**: **No single enhancement method dominates across all real-world scenes**.
  - A filter optimal for uniform gloom amplifies noise catastrophically on grainy night scenes.
  - A smoothing filter destroys fine textures on dark but crisp scenes.
- **The Dilemma**: Full-reference quality metrics (PSNR, SSIM) require the ground truth clean image, which is strictly unavailable at inference time.

---

## Slide 2: Why Existing No-Reference (Blind) Metrics Fail

- **The Intuitive Attempt**: Why not run candidate filters and rank them using blind image quality assessment (NR-IQA) metrics like BRISQUE or NIQE?
- **Catastrophic Failure**:
  - Lyu et al. (*BlindPSNR*, IEEE SPL 2026) benchmarked 7 popular NR-IQA metrics for ranking low-light enhancers:
  - **Result: 0% Top-1 Selection Accuracy**.
- **Root Cause**: NR-IQA metrics evaluate perceptual "naturalness" in the abstract. A blurred, texture-free image often scores high on blind naturalness while destroying true scene structure.
- **Key Insight**: We must predict the winning algorithm from the **degradation physics of the input image itself**, prior to running any filter.

---

## Slide 3: Formulating Algorithm Selection & Decision Regret

- **John Rice's Framework (1976)**:
  $$\pi: \mathcal{X} \xrightarrow{\phi(x)} \mathcal{A} \quad \text{where } \mathcal{A} = \{a_1, \dots, a_6\}$$
- **The Metric: Non-Negative Decibel Regret**:
  - Accuracy alone treats near-ties identically to catastrophic errors.
  - We evaluate the true cost of decision-making:
    $$r(x, \pi) \triangleq s_{a^*}(x) - s_{\pi(x)}(x) \ge 0\text{ dB}$$
  - Losing $0.02\text{ dB}$ on a near-tie is harmless; losing $12\text{ dB}$ by amplifying noise into grain is disastrous.
  - Goal: Minimize mean regret $\bar{R}(\pi)$ and cap maximum worst-case regret $R_{\max}(\pi)$.

---

## Slide 4: The Six-Method Parameter-Locked Portfolio ($\mathcal{A}$)

All operators operate with locked parameters on luminance (LAB color space) to prevent parameter tuning confounds:

1. **Identity ($a_1$)**: Invariant copy ($\hat{y} = x$). Prevents degradation on adequately exposed scenes.
2. **Gamma Correction ($a_2$)**: Power-law LUT ($s = 255 \cdot (r/255)^{0.5}$) lifting deep shadows.
3. **CLAHE ($a_3$)**: Local adaptive histogram equalization (`clipLimit=2.0`, `8x8` tiles) on luminance.
4. **Histogram Equalization ($a_4$)**: Global cumulative distribution mapping on luminance.
5. **Single-Scale Retinex ($a_5$)**: Logarithmic illumination surround subtraction ($\sigma = 80.0$).
6. **Bilateral + Unsharp ($a_6$)**: Two-stage denoising ($d=9, \sigma_c=75, \sigma_s=75$) + sharpening ($\sigma=1.5, \alpha=0.8$).

---

## Slide 5: The Handcrafted Physical Feature Space ($\phi(x)$)

Computed exclusively on degraded input $x$ in **$<2\text{ ms}$ on CPU** without candidate execution:

| Feature | Mathematical Definition | Physical Degradation Captured |
|:---|:---|:---|
| **Brightness ($\mu$)** | $\frac{1}{N} \sum_i x_i$ | Global exposure deficit / gloom level |
| **Contrast ($\sigma$)** | $\sqrt{\frac{1}{N} \sum_i (x_i - \mu)^2}$ | Tonal dynamic compression |
| **Shannon Entropy ($H$)** | $-\sum_j p(j) \log_2 p(j)$ | Information density across 256 bins |
| **Noise ($\sigma_n$)** | $1.4826 \cdot \text{median}(\|L\|)$ in flattest decile | Sensor noise level in smooth regions |
| **Dark Channel ($D$)** | Mean of local $15 \times 15$ eroded $\min_c I_c$ | Depth of dark shadows and haze |
| **Edge Statistics** | Sobel gradient mean & 90th percentile | Crispness vs. blur of structural boundaries |
| **Colourfulness ($C$)** | Hasler-Süsstrunk opponent chromatic std + mean | Degree of chromatic wash-out |

---

## Slide 6: Zero-Leakage Experimental Protocol

- **Dataset**: LOL v1 Benchmark (485 real training pairs, 15 official held-out test pairs).
- **The Five Traps Avoided**:
  1. *No Scene Leakage*: Naive random K-fold leaks near-duplicate scenes. Data is split strictly by physical scene ID using `GroupKFold`.
  2. *No Preprocessing Leakage*: Feature scalers (`StandardScaler`) are fit strictly inside training folds.
  3. *No Label Leakage*: Features are extracted strictly on $x$ without reference or filter outputs.
  4. *Multi-Seed Reliability*: 5 folds $\times$ 3 random seeds = 15 runs per model.
  5. *Fixed Hyperparameters*: Model grids frozen prior to official test set evaluation.

---

## Slide 7: Primary Benchmark Results (15 Folds, LOL v1)

| Model / Selection Strategy | Accuracy Mean (%) | Accuracy Std (%) | Mean Regret (dB) | Regret Std (dB) | Max Regret (dB) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Choice (Chance)** | 16.77 | $\pm 2.98$ | 6.966 | $\pm 0.424$ | 18.441 |
| **Best Fixed Method (Retinex)** | 45.57 | $\pm 5.31$ | 2.734 | $\pm 0.267$ | 18.834 |
| **Hand-Written Heuristic Rule** | 12.16 | $\pm 3.02$ | 6.926 | $\pm 0.327$ | 17.186 |
| **Multinomial Logistic Regression** | 61.65 | $\pm 2.20$ | 1.384 | $\pm 0.183$ | 14.756 |
| **Random Forest Selector (Proposed)** | **67.42** | $\mathbf{\pm 2.25}$ | **1.166** | $\mathbf{\pm 0.154}$ | **16.035** |
| **XGBoost Classifier** | 65.91 | $\pm 4.99$ | 1.295 | $\pm 0.220$ | 16.016 |
| **Theoretical Oracle Upper Bound** | **100.00** | $\pm 0.00$ | **0.000** | $\pm 0.000$ | **0.000** |

---

## Slide 8: Key Empirical Achievements

1. **Top-1 Selection Accuracy**: **$67.42\%$** — beating the best fixed single method by **$+21.85\%$**.
2. **Fidelity Dividend**: Mean decision regret slashed from $2.734\text{ dB}$ down to **$1.166\text{ dB}$** (saving **$1.568\text{ dB}$** per image).
3. **Oracle Gap Recovery**: Captures **$57.35\%$** of the theoretical headroom between static enhancement and the Oracle.
4. **Statistical Significance**: Paired $t$-test against Retinex confirms **$p < 10^{-12}$** (extremely statistically significant).
5. **Failure of Manual Heuristics**: Hand-crafted rules fail at $12.16\%$ accuracy, proving humans cannot intuit high-dimensional degradation interactions.

---

## Slide 9: Generalization to Held-Out Test Set (`eval15`)

Performance on the official unseen 15 test pairs:

- **Accuracy**: $66.67\%$ (maintained cleanly from training).
- **Mean Regret**: $1.283\text{ dB}$ (vs. $2.777\text{ dB}$ for the fixed baseline).
- **Worst-Case Risk Capped**:
  - Best Fixed Baseline Max Regret: **$11.448\text{ dB}$**
  - Random Forest Selector Max Regret: **$5.328\text{ dB}$**
  - **Halves catastrophic failure risk** on completely unseen real-world scenes!

---

## Slide 10: Feature Importance & Ablation Insights

- **Gini Feature Importance**:
  - **Contrast ($\sigma$, 23.1%)** and **Dark Channel Prior ($D$, 20.1%)** dominate decisions.
  - **Shannon Entropy ($H$, 15.8%)** and **Brightness ($\mu$, 13.4%)** follow.
  - **Noise & Edge Statistics (18.6% combined)** serve as critical boundary guardians, preventing contrast boosters from triggering on noisy backgrounds.
- **Ablation Study**:
  - Using only Brightness & Contrast drops accuracy to $58.76\%$ ($1.54\text{ dB}$ regret).
  - All 8 physical features are necessary to reach peak performance.

---

## Slide 11: System Artifacts & Engineering Deliverables

- **Modular Python Package**: `src/` modularized with `methods`, `features`, `labels`, `dataset`, `train`, `evaluate`.
- **Standalone CLI Inference Tool**:
  ```bash
  python3 src/demo.py --input test.png --output_dir results/demo_output
  ```
  *(Features extracted, decision made, and filter executed in $<100\text{ ms}$).*
- **Automated Makefile**: `make test`, `make demo`, `make benchmark`, `make clean`.
- **Complete Test Suite**: Unit tests verifying operators, metrics, and models.
- **Formal IEEE Research Paper**: Full publication draft in LaTeX (`report/IEEE_Paper_Image_Enhancement_Selection.tex`).

---

## Slide 12: Multi-Semester Research Roadmap

- **Phase 0 (Current / 5th Sem - Nov 2026)**:
  - *Goal*: Can cheap physical features predict the PSNR winner?
  - *Result*: **Validated** (+21.85% accuracy, 57% regret reduction).
- **Phase 1 (6th Sem - Spring 2027)**:
  - *Goal*: Downstream task evaluation. Does the PSNR-optimal selector maximize object detection mAP (YOLOv10 / RT-DETR) on nighttime security feeds?
- **Phase 2 (7th Sem - Autumn 2027)**:
  - *Goal*: Cross-degradation transferability (composite haze, low-light, motion blur).
- **Phase 3 (8th Sem - Spring 2028)**:
  - *Goal*: Hybrid portfolios incorporating lightweight neural enhancers (Zero-DCE, SCI) alongside classical filters.

---

### Questions & Discussion
**Thank you!**  
*Team Triathlon — Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (Gati Shakti Vishwavidyalaya, Vadodara)*  
*Supervisor: Prof. Kishor P. Upla (SVNIT Surat)*

