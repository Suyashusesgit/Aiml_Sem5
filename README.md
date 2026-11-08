# Image Enhancement and Per-Image Selection (Phase 0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark: LOL v1](https://img.shields.io/badge/Benchmark-LOL%20v1-orange.svg)](https://daooshee.github.io/BMVC2018website/)
[![Evaluation: IEEE](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](report/IEEE_Paper_Image_Enhancement_Selection.pdf)

> **5th Semester Capstone Project in Artificial Intelligence and Machine Learning (AIML)**  
> **Students**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (**Department of Electronics and Communication Engineering, Gati Shakti Vishwavidyalaya, Vadodara 390004, Gujarat**)  
> **Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE* (**Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat**)  
> **Date**: November 2026

---

## 📌 Executive Summary

Digital image enhancement is an ill-posed inverse problem: under-exposure, photon shot noise, and sensor limitations irreversibly destroy scene information. While dozens of restoration algorithms exist, **no single method is optimal for every image**. Static single-method deployment forces heavy quality sacrifices on out-of-distribution frames. Furthermore, recent research (*BlindPSNR*, Lyu et al., 2026) demonstrated that standard no-reference quality assessment metrics (NR-IQA, such as BRISQUE and NIQE) **fail catastrophically at ranking enhancement candidates, achieving 0% top-1 accuracy**.

This project solves this dilemma by framing per-image enhancement selection as an **algorithm selection problem** (Rice, 1976). We extract **an 8-dimensional feature vector spanning seven physical degradation categories** exclusively from the degraded input in under $1\text{ ms}$, completely bypassing the need to execute candidate filters at inference time. Supervised classifiers trained against an offline full-reference oracle select the optimal enhancer from a six-method bank.

### Key Benchmark Achievements (15-Run GroupKFold by Scene):
- **Top-1 Selection Accuracy**: **$68.18\% \pm 3.49\%$** (outperforming the best fixed method by **$+22.61\%$**, $p = 9.44 \times 10^{-16}$).
- **Mean Decision Regret**: Slashed from $2.734\text{ dB}$ (fixed baseline) to **$1.160 \pm 0.236\text{ dB}$** (a net saving of **$1.574\text{ dB}$** per image, $p = 1.90 \times 10^{-11}$).
- **Mean SSIM**: Reaches **$0.6949$** (SSIM Regret **$0.0408$**).
- **Oracle Gap Captured**: Recovers **$57.57\%$** of the theoretical gain between the static baseline and the full-reference oracle.
- **Held-Out Test Set (`eval15`)**: Maintains **$66.67\%$ accuracy**, **$1.283\text{ dB}$ mean regret**, and caps worst-case regret at **$5.328\text{ dB}$** (vs. $11.448\text{ dB}$ for the fixed baseline).
- **Downstream Utility**: Boosts ORB keypoints by $+113\%$, SIFT RANSAC inliers from $12.7 \to 213.7$ matches, and Canny boundary F1-score from $11.34\% \to 66.90\%$.

---

## 🗂 Repository Structure

```
Aiml_Sem5/
├── Makefile                 # Quick-access CLI shortcuts (install, test, benchmark, demo, clean, app)
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
│   ├── IEEE_Paper_Overleaf_Package.zip             # Self-contained Overleaf camera-ready zip
│   ├── Project_Triathlon_Executive_Presentation.pptx # 12-slide widescreen PowerPoint deck
│   ├── Executive_Summary_Prof_Upla.docx            # 1-page high-density executive briefing brief
│   ├── Project_Triathlon_Step_by_Step_Walkthrough.docx # Comprehensive 8-chapter project guide
│   ├── Monday_Update_Prof_Upla_Nov9.md             # Weekly supervisor progress memo
│   ├── Phase0_Presentation_Slides.md               # Markdown slide deck content
│   ├── Phase0_Proposal.md                          # Phase 0 capstone proposal
│   └── Phase0_SOP.md                               # Standard operating procedure
├── results/
│   ├── scores.csv           # Cached PSNR/SSIM scores across all 6 methods
│   ├── features.csv         # Cached 8-dimensional physical features
│   ├── metrics_table.csv    # Primary 15-run GroupKFold benchmark results
│   ├── ablation_table.csv   # 5-tier feature ablation study
│   ├── eval15_test_results.csv # Performance on official unseen test split
│   ├── worst_10_errors.csv  # Detailed error analysis of top-10 regret cases
│   ├── downstream_geometric_metrics.csv # SIFT RANSAC & Canny boundary benchmark
│   ├── noise_robustness_sweep.csv       # Continuous sensor noise sweep metrics
│   └── figures/
│       ├── fig1_oracle_distribution.png       # Frequency of winning methods
│       ├── fig2_confusion_matrix.png         # Normalized classification confusion matrix
│       ├── fig3_regret_cdf.png                # Empirical cumulative regret distribution
│       ├── fig4_feature_importances.png      # Gini feature importance ranking
│       ├── fig5_qualitative_strip.png        # Qualitative restoration comparison strip
│       ├── fig6_downstream_utility.png       # Downstream ORB keypoint density gain
│       ├── fig7_downstream_geometric_eval.png # SIFT RANSAC inliers & Canny boundary F1
│       └── fig8_robustness_noise_sweep.png   # Dynamic selector adaptation under noise
├── scripts/
│   ├── generate_presentation.py       # Programmatic PPTX presentation builder
│   ├── generate_handout_docx.py       # 1-page executive briefing Word generator
│   └── generate_walkthrough_docx.py   # Comprehensive step-by-step Word generator
├── src/
│   ├── __init__.py          # Package initialization and exports
│   ├── methods.py           # The Six-Method Enhancement Bank
│   ├── features.py          # The 8-dimensional Physical Degradation Descriptors
│   ├── labels.py            # MSE, PSNR, SSIM, Oracle and Regret computation
│   ├── dataset.py           # Preprocessing contract and scene grouping
│   ├── train.py             # 5-Fold GroupKFold CV pipeline (zero leakage)
│   ├── evaluate.py          # Metrics, figures, ablations, error analysis
│   ├── downstream_detection.py      # Downstream ORB keypoint utility evaluation
│   ├── downstream_semantic_eval.py  # SIFT RANSAC and Canny boundary evaluation
│   ├── robustness_sweep.py          # Gaussian sensor noise sensitivity sweep
│   └── demo.py              # Standalone single-image CLI inference tool
├── tests/
│   ├── __init__.py          # Test suite package
│   └── test_pipeline.py     # Unit tests for methods, features, metrics, and models
├── app.py                   # Real-time web application dashboard (HTTP/JSON/HTML)
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

All features form an **8-dimensional vector** computed strictly on the degraded input image in under $1\text{ ms}$:

1. **Mean Luminance ($\mu$)**: Global intensity mean $\mu = \frac{1}{N} \sum_i x_i$ (exposure level).
2. **RMS Contrast ($\sigma$)**: Standard deviation $\sigma = \sqrt{\frac{1}{N} \sum_i (x_i - \mu)^2}$ (dynamic compression).
3. **Shannon Entropy ($H$)**: $-\sum_j p(j) \log_2 p(j)$ across 256 intensity bins.
4. **Flat-Region Noise ($\sigma_n$)**: Median absolute deviation of Laplacian responses in flat regions ($1.4826 \cdot \text{median}(|L|)$ over the flattest decile).
5. **Dark Channel Prior ($D$)**: Mean of local $15 \times 15$ eroded channel minimum $\min_{c \in \{R,G,B\}} x_c$.
6. **Edge Mean**: Spatial mean of Sobel gradient magnitude $|\nabla I|$.
7. **Edge p90**: 90th percentile of Sobel gradient magnitude (salient boundary energy).
8. **Colourfulness ($C$)**: Hasler-Süsstrunk opponent color chromatic metric $\sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3 \sqrt{\mu_{rg}^2 + \mu_{yb}^2}$.

---

## 📊 Primary Experimental Results

Evaluated via **5-Fold Grouped Cross-Validation (`GroupKFold`)** by physical scene across **3 random seeds** ($5 \times 3 = 15$ runs) on the LOL benchmark (485 training pairs):

| Model / Selection Strategy | Top-1 Accuracy Mean (%) | Accuracy Std (%) | Mean Regret (dB) | Regret Std (dB) | Max Regret (dB) | Mean SSIM | SSIM Regret |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Choice (Chance)** | 17.66 | $\pm 4.77$ | 6.916 | $\pm 0.511$ | 18.085 | 0.4392 | 0.2859 |
| **Best Fixed Method (Retinex)** | 45.57 | $\pm 3.41$ | 2.734 | $\pm 0.118$ | 17.815 | 0.7287 | 0.0313 |
| **Hand-Written Heuristic Rule** | 12.16 | $\pm 3.29$ | 6.926 | $\pm 0.360$ | 17.064 | 0.5751 | 0.1603 |
| **Multinomial Logistic Regression** | 61.86 | $\pm 2.53$ | 1.421 | $\pm 0.213$ | 15.032 | 0.6877 | 0.0486 |
| **Random Forest Selector (Proposed)**| **68.18** | $\mathbf{\pm 3.49}$ | **1.160** | $\mathbf{\pm 0.236}$ | **15.181** | **0.6949** | **0.0408** |
| **XGBoost Classifier** | 67.70 | $\pm 4.90$ | 1.249 | $\pm 0.322$ | 14.673 | 0.6926 | 0.0428 |
| **Theoretical Oracle Upper Bound** | **100.00** | $\pm 0.00$ | **0.000** | $\pm 0.000$ | **0.000** | **0.7297** | **0.0000** |

### Evaluation on Official Held-Out Test Set (`eval15`):
- **Best Fixed Baseline**: $46.67\%$ accuracy, $2.777\text{ dB}$ mean regret, $11.448\text{ dB}$ max regret, Mean SSIM $0.7212$.
- **Random Forest Selector**: **$66.67\%$ accuracy**, **$1.283\text{ dB}$ mean regret**, **$5.328\text{ dB}$ max regret** (halving the maximum risk on unseen test scenes), Mean SSIM **$0.6965$** (SSIM regret **$0.0623$**).

---

## 📈 Visualizations & Figures

| Figure | Description | File |
|:---|:---|:---|
| **Fig 1** | Distribution of Oracle Enhancement Methods across the 485 training pairs | [`results/figures/fig1_oracle_distribution.png`](results/figures/fig1_oracle_distribution.png) |
| **Fig 2** | Normalized Confusion Matrix for the Random Forest Selector | [`results/figures/fig2_confusion_matrix.png`](results/figures/fig2_confusion_matrix.png) |
| **Fig 3** | Empirical CDF of Decision Regret (dB) comparing all models | [`results/figures/fig3_regret_cdf.png`](results/figures/fig3_regret_cdf.png) |
| **Fig 4** | Gini Feature Importance Ranking (Mean Decrease in Impurity) | [`results/figures/fig4_feature_importances.png`](results/figures/fig4_feature_importances.png) |
| **Fig 5** | Qualitative Restoration Comparison Strip (Multi-Scene Restoration vs. Retinex) | [`results/figures/fig5_qualitative_strip.png`](results/figures/fig5_qualitative_strip.png) |
| **Fig 6** | Downstream Keypoint Utility Benchmark (ORB Detection Density Gain) | [`results/figures/fig6_downstream_utility.png`](results/figures/fig6_downstream_utility.png) |
| **Fig 7** | Downstream Semantic & Geometric Benchmark (SIFT RANSAC & Canny Boundary F1) | [`results/figures/fig7_downstream_geometric_eval.png`](results/figures/fig7_downstream_geometric_eval.png) |
| **Fig 8** | Sensor Noise Robustness Sweep ($\sigma \in [0, 50]$ showing dynamic policy shift) | [`results/figures/fig8_robustness_noise_sweep.png`](results/figures/fig8_robustness_noise_sweep.png) |

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/Aiml_Sem5.git
cd Aiml_Sem5
make install
```

### 2. Dataset Verification
The dataset archive is stored in `data/lol_dataset.zip`. Verify its SHA-256 checksum:
```bash
shasum -a 256 data/lol_dataset.zip
# Expected: 419fddc48958cd0f5599939ee0248852a37ceb8bb738c9b9525e95b25a89de9a
```

### 3. Run the End-to-End Benchmark Pipeline
Execute the full pipeline (feature extraction, 15-run grouped CV, ablations, eval15 test evaluation, figure generation):
```bash
make benchmark
# Or directly: python3 main.py
```

### 4. Interactive Web Application
Launch the real-time web dashboard with interactive before/after split comparison slider, ORB keypoints overlay, and feature gauges:
```bash
make app
# Open http://localhost:8501 in your browser
```

### 5. Downstream Semantic & Geometric Benchmarks
Run downstream feature correspondence (SIFT + RANSAC inliers) and structural edge preservation (Canny boundary F1):
```bash
make downstream_semantic
```

### 6. Sensor Noise Robustness Sweep
Evaluate dynamic policy adaptation under varying Gaussian sensor noise:
```bash
make robustness
```

### 7. Executive Presentation & Walkthrough Word Generators
Generate the 12-slide widescreen PowerPoint presentation deck and the supervisor executive summary handout:
```bash
make presentation
make docx
```

### 8. Camera-Ready Overleaf Zip Archive
Build the camera-ready LaTeX package with all figures:
```bash
make package
```

### 9. Run Test Suite
```bash
make test
```

---

## 📄 IEEE Paper & Capstone Deliverables

- **IEEE Research Paper (LaTeX)**: [`report/IEEE_Paper_Image_Enhancement_Selection.tex`](report/IEEE_Paper_Image_Enhancement_Selection.tex)
- **IEEE Research Paper (Markdown)**: [`report/IEEE_Paper_Image_Enhancement_Selection.md`](report/IEEE_Paper_Image_Enhancement_Selection.md)
- **Overleaf Camera-Ready Archive**: [`report/IEEE_Paper_Overleaf_Package.zip`](report/IEEE_Paper_Overleaf_Package.zip)
- **Executive Presentation Deck (PowerPoint, 12 Slides)**: [`report/Project_Triathlon_Executive_Presentation.pptx`](report/Project_Triathlon_Executive_Presentation.pptx)
- **Executive Summary Handout (1-Page Brief)**: [`report/Executive_Summary_Prof_Upla.docx`](report/Executive_Summary_Prof_Upla.docx)
- **Comprehensive Project Walkthrough (Word, 8 Chapters)**: [`report/Project_Triathlon_Step_by_Step_Walkthrough.docx`](report/Project_Triathlon_Step_by_Step_Walkthrough.docx)
- **Supervisor Status Memo (Day-5 Gate)**: [`report/Monday_Update_Prof_Upla_Nov9.md`](report/Monday_Update_Prof_Upla_Nov9.md)
- **Standard Operating Procedure (SOP)**: [`report/Phase0_SOP.md`](report/Phase0_SOP.md)
- **Phase 0 Capstone Proposal**: [`report/Phase0_Proposal.md`](report/Phase0_Proposal.md)

---

## 🗺 Multi-Semester Research Roadmap

- **Phase 0 (Current -- Nov 2026)**: Can cheap physical features predict the PSNR winner?  
  *Outcome*: **Validated** (+22.61% accuracy over fixed method, 57.57% regret reduction, $p = 9.44 \times 10^{-16}$).
- **Phase 1 (Semester 6)**: Downstream object detection evaluation. Does the PSNR-optimal choice maximize downstream detection mAP with YOLOv10 / RT-DETR on night frames?
- **Phase 2 (Semester 7)**: Cross-degradation transferability. Testing selectors trained across composite degradations (low-light, haze, blur).
- **Phase 3 (Semester 7)**: Hybrid portfolios incorporating lightweight deep neural enhancers (Zero-DCE, SCI) alongside classical filters.

---

## 👥 Team & Acknowledgments

- **Students**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (5th Semester B.Tech, Department of Electronics and Communication Engineering (ECE), Gati Shakti Vishwavidyalaya, Vadodara 390004, Gujarat, India).
- **Supervisor**: Prof. Kishor P. Upla (*Senior Member, IEEE*), Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat, India.
- **Reference**: Prepared according to the handbook *"Image Enhancement and Per-Image Selection: from pixels and histograms to classical learning, regret, and the research frontier"* (SVNIT, Nov 2026).

