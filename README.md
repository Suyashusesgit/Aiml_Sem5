# Image Enhancement and Per-Image Selection (Phase 0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark: LOL v1](https://img.shields.io/badge/Benchmark-LOL%20v1-orange.svg)](https://daooshee.github.io/BMVC2018website/)
[![Evaluation: IEEE](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](report/IEEE_Paper_Image_Enhancement_Selection.pdf)

> **5th Semester Capstone Project in Artificial Intelligence and Machine Learning (AIML)**  
> **Institution**: Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
> **Department**: Department of Electronics Engineering  
> **Team Triathlon**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan  
> **Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE*  
> **Date**: November 2026

---

## 📌 Executive Summary

Digital image enhancement is an ill-posed inverse problem: under-exposure, photon shot noise, and sensor limitations irreversibly destroy scene information. While dozens of restoration algorithms exist, **no single method is optimal for every image**. Static single-method deployment forces heavy quality sacrifices on out-of-distribution frames. Furthermore, recent research (*BlindPSNR*, Lyu et al., 2026) demonstrated that standard no-reference quality assessment metrics (NR-IQA, such as BRISQUE and NIQE) **fail catastrophically at ranking enhancement candidates, achieving 0% top-1 accuracy**.

This project solves this dilemma by framing per-image enhancement selection as an **algorithm selection problem** (Rice, 1976). We extract **seven computationally cheap physical degradation statistics** exclusively from the degraded input in milliseconds, completely bypassing the need to execute candidate filters at inference time. Supervised classifiers trained against an offline full-reference oracle select the optimal enhancer from a six-method bank.

### Key Benchmark Achievements:
- **Top-1 Selection Accuracy**: **$67.42\% \pm 2.25\%$** (outperforming the best fixed method by **$+21.85\%$**).
- **Mean Decision Regret**: Slashed from $2.734\text{ dB}$ (fixed baseline) to **$1.166 \pm 0.154\text{ dB}$** (a net saving of **$1.568\text{ dB}$** per image).
- **Oracle Gap Captured**: Recovers **$57.35\%$** of the theoretical gain between the static baseline and the oracle.
- **Held-Out Test Set (`eval15`)**: Maintains **$66.67\%$ accuracy** and caps worst-case regret at **$5.328\text{ dB}$** (vs. $11.448\text{ dB}$ for the fixed baseline).

---

## 🗂 Repository Structure

```
Aiml_Sem5/
├── Makefile                 # Quick-access CLI shortcuts (install, test, benchmark, demo, clean)
├── configs/
│   └── base.yaml            # Hyperparameters, paths, seeds, and operator configs
├── data/
│   ├── lol_dataset.zip      # Official LOL v1 benchmark archive (SHA-256 verified)
│   └── lol_dataset/
│       ├── our485/          # 485 real training pairs (low/ and high/)
│       └── eval15/          # 15 official held-out test pairs
├── docs/
│   └── REFERENCE_RESOURCES.md  # Comprehensive literature, textbooks, links & team matrix
├── models/
│   └── rf_selector.joblib   # Persisted Random Forest selector pipeline
├── notebooks/
│   └── exploration_and_benchmarks.ipynb  # Interactive visualization notebook
├── report/
│   ├── IEEE_Paper_Image_Enhancement_Selection.tex  # Full IEEE publication LaTeX source
│   ├── IEEE_Paper_Image_Enhancement_Selection.md   # IEEE Research Paper (Markdown)
│   ├── IEEEtran.cls                                # IEEE Journal document class
│   └── Monday_Update_Prof_Upla_Nov9.md             # Weekly supervisor progress memo
├── results/
│   ├── scores.csv           # Cached PSNR/SSIM scores across all 6 methods
│   ├── features.csv         # Cached 8-dimensional physical features
│   ├── metrics_table.csv    # Primary 15-run GroupKFold benchmark results
│   ├── ablation_table.csv   # 5-tier feature ablation study
│   ├── eval15_test_results.csv # Performance on official unseen test split
│   ├── worst_10_errors.csv  # Detailed error analysis of top-10 regret cases
│   └── figures/
│       ├── fig1_oracle_distribution.png   # Frequency of winning methods
│       ├── fig2_confusion_matrix.png     # Normalized classification confusion matrix
│       ├── fig3_regret_cdf.png            # Empirical cumulative regret distribution
│       └── fig4_feature_importances.png  # Gini feature importance ranking
├── src/
│   ├── __init__.py          # Package initialization and exports
│   ├── methods.py           # The Six-Method Enhancement Bank
│   ├── features.py          # The Seven Physical Degradation Descriptors
│   ├── labels.py            # MSE, PSNR, SSIM, Oracle and Regret computation
│   ├── dataset.py           # Preprocessing contract and scene grouping
│   ├── train.py             # 5-Fold GroupKFold CV pipeline (zero leakage)
│   ├── evaluate.py          # Metrics, figures, ablations, error analysis
│   └── demo.py              # Standalone single-image CLI inference tool
├── tests/
│   ├── __init__.py          # Test suite package
│   └── test_pipeline.py     # Unit tests for methods, features, metrics, and models
├── main.py                  # End-to-end execution script
├── requirements.txt         # Pinned exact library versions
├── LICENSE                  # MIT License
└── README.md                # Project documentation and reproduction guide
```

---

## 🔬 The Six-Method Enhancement Bank

Every method runs with parameter locking to isolate method selection from parameter tuning:

| Method | Core Operation | Parameters | Target Degradation | Typical Failure Mode |
|:---|:---|:---|:---|:---|
| **Identity** | Invariant passthrough | None | Already well-exposed frames | Leaves degradation untreated |
| **Gamma** | Power-law LUT | $\gamma = 0.5$ | Global underexposure | Amplifies background sensor noise |
| **CLAHE** | Local histogram eq. on LAB luminance | `clipLimit=2.0`, `grid=(8,8)` | Low local tonal contrast | Boundary halos, grain amplification |
| **Hist Eq** | Global CDF on LAB luminance | None | Narrow histogram compression | Posterization, harsh over-contrast |
| **Retinex** | Single-Scale SSR illumination subtraction | $\sigma = 80.0$, min-max norm | Non-uniform illumination fields | Color desaturation, edge halos |
| **Bilat+Unsharp** | Edge-preserving denoising + sharpening | $d=9, \sigma_c=75, \sigma_s=75, \alpha=0.8$ | Soft and grainy scenes | Gradient overshoot, haloing |

---

## 📐 The Handcrafted Physical Feature Space

All features are computed strictly on the degraded input image in under $2\text{ ms}$:

1. **Brightness ($\mu$)**: Global intensity mean $\mu = \frac{1}{N} \sum_i x_i$ (exposure level).
2. **Contrast ($\sigma$)**: Standard deviation $\sigma = \sqrt{\frac{1}{N} \sum_i (x_i - \mu)^2}$ (dynamic compression).
3. **Entropy ($H$)**: Shannon entropy $-\sum_j p(j) \log_2 p(j)$ across 256 intensity bins.
4. **Noise ($\sigma_n$)**: Median absolute deviation of Laplacian responses in flat regions ($1.4826 \cdot \text{median}(|L|)$ over the flattest decile).
5. **Dark Channel Prior ($D$)**: Mean of local $15 \times 15$ eroded channel minimum $\min_{c \in \{R,G,B\}} x_c$.
6. **Edge Statistics**: Mean and 90th percentile of Sobel gradient magnitude $|\nabla I|$.
7. **Colourfulness ($C$)**: Hasler-Süsstrunk opponent color chromatic metric $\sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3 \sqrt{\mu_{rg}^2 + \mu_{yb}^2}$.

---

## 📊 Primary Experimental Results

Evaluated via **5-Fold Grouped Cross-Validation (`GroupKFold`)** by physical scene across **3 random seeds** ($5 \times 3 = 15$ runs) on the LOL benchmark (485 training pairs):

| Model / Selection Strategy | Top-1 Accuracy Mean (%) | Accuracy Std (%) | Mean Regret (dB) | Regret Std (dB) | Max Regret (dB) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Choice (Chance)** | 16.77 | $\pm 2.98$ | 6.966 | $\pm 0.424$ | 18.441 |
| **Best Fixed Method (Retinex)** | 45.57 | $\pm 5.31$ | 2.734 | $\pm 0.267$ | 18.834 |
| **Hand-Written Heuristic Rule** | 12.16 | $\pm 3.02$ | 6.926 | $\pm 0.327$ | 17.186 |
| **Multinomial Logistic Regression** | 61.65 | $\pm 2.20$ | 1.384 | $\pm 0.183$ | 14.756 |
| **Random Forest Selector (Proposed)**| **67.42** | $\mathbf{\pm 2.25}$ | **1.166** | $\mathbf{\pm 0.154}$ | **16.035** |
| **XGBoost Classifier** | 65.91 | $\pm 4.99$ | 1.295 | $\pm 0.220$ | 16.016 |
| **Theoretical Oracle Upper Bound** | **100.00** | $\pm 0.00$ | **0.000** | $\pm 0.000$ | **0.000** |

### Evaluation on Official Held-Out Test Set (`eval15`):
- **Best Fixed Baseline**: $46.67\%$ accuracy, $2.777\text{ dB}$ mean regret, $11.448\text{ dB}$ max regret.
- **Random Forest Selector**: **$66.67\%$ accuracy**, **$1.283\text{ dB}$ mean regret**, **$5.328\text{ dB}$ max regret** (halving the maximum risk on unseen test scenes).

---

## 📈 Visualizations & Figures

| Figure | Description | File |
|:---|:---|:---|
| **Fig 1** | Distribution of Oracle Enhancement Methods across the 485 training pairs | [`results/figures/fig1_oracle_distribution.png`](results/figures/fig1_oracle_distribution.png) |
| **Fig 2** | Normalized Confusion Matrix for the Random Forest Selector | [`results/figures/fig2_confusion_matrix.png`](results/figures/fig2_confusion_matrix.png) |
| **Fig 3** | Empirical CDF of Decision Regret (dB) comparing all models | [`results/figures/fig3_regret_cdf.png`](results/figures/fig3_regret_cdf.png) |
| **Fig 4** | Gini Feature Importance Ranking (Mean Decrease in Impurity) | [`results/figures/fig4_feature_importances.png`](results/figures/fig4_feature_importances.png) |
| **Fig 5** | Qualitative Restoration Comparison Strip (Multi-Scene Restoration vs. Retinex) | [`results/figures/fig5_qualitative_strip.png`](results/figures/fig5_qualitative_strip.png) |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and install the pinned dependencies:
```bash
git clone https://github.com/your-username/Aiml_Sem5.git
cd Aiml_Sem5
pip install -r requirements.txt
```

### 2. Dataset Verification
The dataset archive is stored in `data/lol_dataset.zip`. Verify its SHA-256 checksum:
```bash
shasum -a 256 data/lol_dataset.zip
# Expected: 419fddc48958cd0f5599939ee0248852a37ceb8bb738c9b9525e95b25a89de9a
```

### 3. Run the End-to-End Pipeline
Execute the full benchmark pipeline (feature extraction, 15-run grouped CV, ablations, eval15 test evaluation, figure generation):
```bash
python3 main.py
```

### 4. Interactive Web Application
Launch the real-time web dashboard with interactive before/after viewer and feature gauges:
```bash
python3 app.py
# Open http://localhost:8501 in your browser
```

### 5. Single-Image Inference Demo
Run CLI inference and visual comparison on any sample image:
```bash
python3 src/demo.py --input data/lol_dataset/eval15/low/1.png --output_dir results/demo_output
```

### 6. Interactive Exploration
Launch the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/exploration_and_benchmarks.ipynb
```

---

## 📄 IEEE Paper & Capstone Deliverables

- **IEEE Research Paper (LaTeX)**: [`report/IEEE_Paper_Image_Enhancement_Selection.tex`](report/IEEE_Paper_Image_Enhancement_Selection.tex)
- **IEEE Research Paper (Markdown)**: [`report/IEEE_Paper_Image_Enhancement_Selection.md`](report/IEEE_Paper_Image_Enhancement_Selection.md)
- **Defense Presentation Slides (12 Slides)**: [`report/Phase0_Presentation_Slides.md`](report/Phase0_Presentation_Slides.md)
- **Phase 0 Capstone Proposal**: [`report/Phase0_Proposal.md`](report/Phase0_Proposal.md)
- **Standard Operating Procedure (SOP)**: [`report/Phase0_SOP.md`](report/Phase0_SOP.md)
- **Supervisor Status Memo (Day-5 Gate)**: [`report/Monday_Update_Prof_Upla_Nov9.md`](report/Monday_Update_Prof_Upla_Nov9.md)

---

## 🗺 Multi-Semester Research Roadmap

- **Phase 0 (Current -- Nov 2026)**: Can cheap physical features predict the PSNR winner?  
  *Outcome*: **Validated** (+21.85% accuracy over fixed method, 57.3% regret reduction).
- **Phase 1 (Semester 6)**: Downstream object detection evaluation. Does the PSNR-optimal choice maximize downstream detection mAP with YOLOv10 / RT-DETR on night frames?
- **Phase 2 (Semester 7)**: Cross-degradation transferability. Testing selectors trained across composite degradations (low-light, haze, blur).
- **Phase 3 (Semester 7)**: Hybrid portfolios incorporating lightweight deep neural enhancers (Zero-DCE, SCI) alongside classical filters.

---

## 👥 Team & Acknowledgments

- **Students**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (5th Semester B.Tech, AIML / ECE, SVNIT Surat).
- **Supervisor**: Prof. Kishor P. Upla, Department of Electronics Engineering, SVNIT Surat.
- **Reference**: Prepared according to the handbook *"Image Enhancement and Per-Image Selection: from pixels and histograms to classical learning, regret, and the research frontier"* (SVNIT, Nov 2026).
