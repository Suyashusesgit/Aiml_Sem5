# Image Enhancement and Per-Image Selection (Phases 0, 1 & 2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmark: LOL v1](https://img.shields.io/badge/Benchmark-LOL%20v1-orange.svg)](https://daooshee.github.io/BMVC2018website/)
[![Evaluation: IEEE](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](report/IEEE_Paper_Image_Enhancement_Selection.pdf)
[![Predict-Then-Optimize](https://img.shields.io/badge/PTO%20Regressors-1.368%20dB%20Regret-brightgreen.svg)](src/predict_then_optimize.py)
[![Throughput](https://img.shields.io/badge/Edge%20Throughput-1350%20FPS%20(CPU)-blue.svg)](results/latency_regret_frontier.csv)
[![Unit Tests](https://img.shields.io/badge/Tests-22%2F22%20Passing-success.svg)](tests/)
[![Downstream: Inlier Matching](https://img.shields.io/badge/ORB%20RANSAC-%2B107.4%25%20Inliers-blueviolet.svg)](src/downstream_matching.py)

> **5th Semester Capstone Project in Artificial Intelligence and Machine Learning (AIML)**  
> **Students**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (**Department of Electronics and Communication Engineering, Gati Shakti Vishwavidyalaya, Vadodara 390004, Gujarat**)  
> **Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE* (**Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat**)  
> **Date**: November 2026

---

## 📌 Executive Summary

Digital image enhancement is an ill-posed inverse problem: under-exposure, photon shot noise, and sensor limitations irreversibly destroy scene information. While dozens of restoration algorithms exist, **no single method is optimal for every image**. Static single-method deployment forces heavy quality sacrifices on out-of-distribution frames. Furthermore, recent research (*BlindPSNR*, Lyu et al., 2026) demonstrated that standard no-reference quality assessment metrics (NR-IQA, such as BRISQUE and NIQE) **fail catastrophically at ranking enhancement candidates, achieving 0% top-1 accuracy**.

This project solves this dilemma by framing per-image enhancement selection as an **algorithm selection problem** (Rice, 1976). We extract **an 8-dimensional feature vector spanning seven physical degradation categories** exclusively from the degraded input in under $1\text{ ms}$, completely bypassing the need to execute candidate filters at inference time. Supervised classifiers trained against an offline full-reference oracle select the optimal enhancer from a multi-operator bank.

### Key Multi-Phase Achievements:
- **Phase 0 & Milestone Completion (Core Selection, Regret Collapse & Predict-Then-Optimize)**:
  - **Predict-Then-Optimize (PTO) Regressors**: Slashes mean regret to **$1.368\text{ dB}$** with **$58.35\%$ Top-1 accuracy** across all 278 physical scenes (0% leakage), outperforming discrete classification ($1.57\text{ dB}$) and fixed Retinex ($2.73\text{ dB}$).
  - **Dong et al. Optical Inversion Duality**: Discovered and integrated the optical isomorphism $I_{\text{inv}}(x) \equiv \text{Koschmieder's Haze Model}$. Implemented `invert_dehaze` as the 7th method ($9.2\text{ ms}$ on CPU), achieving Oracle winner status on **$21.8\%$** of benchmark scenes.
  - **Latency-Regret Pareto Frontier**: Evaluated microsecond edge budgets from $0.1\text{ ms}$ to $15.0\text{ ms}$. At **$1.0\text{ ms}$ budget**, selector achieves **$1.583\text{ dB}$ regret at $1,349.8\text{ FPS}$** on single-core CPU.
  - **Random Forest Baseline**: $68.18\% \pm 3.49\%$ accuracy, $1.160\text{ dB}$ regret, capturing $57.57\%$ of the theoretical oracle gap ($p = 9.44 \times 10^{-16}$).
- **Phase 1 (Downstream Perception, Inlier Tracking & Edge Realization)**:
  - **Geometric Inlier Matching (ORB + RANSAC)**: Enhancing photon-starved frames delivers a **$+107.4\%$ relative inlier yield** ($96 \to 199$ inliers, $96.1\%$ inlier ratio), validating downstream visual odometry and SLAM repeatability.
  - **YOLOv8 Detection Recall**: Boosts object recall to **$67.15\%$** vs. ground truth (+18.62% over static Retinex, +12.74% over raw dark frames).
  - **ONNX Real-Time Engine**: Serialized into `models/rf_selector.onnx` ($4.20\text{ MB}$), running at **$11.81\text{ ms}$ ($84.7\text{ FPS}$)** on single-core CPU with only $54.4\text{ MB}$ peak RAM.
- **Phase 2 (Multi-Degradation Cascades & Real-Time Video Streaming)**:
  - **Real-Time Video Stream Processor (`src/video_pipeline.py`)**: Implements Exponential Moving Average (EMA) feature smoothing and a Markovian Hysteresis Gate ($\lambda = 0.15$), completely eliminating temporal method flickering and strobing at **$>100\text{ FPS}$**.
  - **Multi-Hazard 10D Feature Hierarchy**: Slashes regret on defocus blur by **$4.70\text{ dB}$** and haze by **$1.17\text{ dB}$**, delivering a **$+0.915\text{ dB}$ net fidelity dividend** across all optical regimes.
  - **RQ2 Decision-Optimality Empirical Proof**: Discovered that PSNR-optimal selection aligns with semantic decision vigilance (e.g. driver eye aspect ratio) in only **$13.3\%$** of scenes, proving that edge tasks require task-adapted utility functions.

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
│   ├── rf_selector.joblib   # Persisted Random Forest selector pipeline
│   ├── rf_selector.onnx     # Optimized ONNX runtime model (4.20 MB, CPU real-time)
│   └── pto_regressors.joblib # Predict-Then-Optimize 6-head continuous regression suite (3.4 KB)
├── notebooks/
│   └── exploration_and_benchmarks.ipynb  # Interactive visualization notebook
├── report/
│   ├── IEEE_Paper_Image_Enhancement_Selection.tex  # Full IEEE publication LaTeX source
│   ├── IEEE_Paper_Image_Enhancement_Selection.pdf  # Camera-ready 14-page IEEE manuscript (PDF)
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
│   ├── scores.csv           # Cached PSNR/SSIM scores across all methods
│   ├── features.csv         # Cached 8-dimensional physical features
│   ├── metrics_table.csv    # Primary 15-run GroupKFold benchmark results
│   ├── ablation_table.csv   # 5-tier feature ablation study
│   ├── latency_regret_frontier.csv # Pareto tradeoff table (1,350 FPS at 1.583 dB)
│   ├── rq2_decision_optimality.csv # RQ2 semantic vigilance vs PSNR alignment data
│   ├── dong_interaction_analysis.csv # Invert-Dehaze feature correlation statistics
│   ├── eval15_test_results.csv # Performance on official unseen test split
│   ├── worst_10_errors.csv  # Detailed error analysis of top-10 regret cases
│   ├── downstream_geometric_metrics.csv # SIFT RANSAC & Canny boundary benchmark
│   ├── downstream_yolo_metrics.csv      # Phase 1 YOLOv8 object detection benchmark
│   ├── edge_hardware_profile.csv        # Phase 1 ONNX CPU edge latency & RAM profile
│   ├── multi_degradation_benchmark.csv  # Phase 2 multi-hazard cascade benchmark
│   ├── noise_robustness_sweep.csv       # Continuous sensor noise sweep metrics
│   └── figures/
│       ├── fig1_oracle_distribution.png       # Frequency of winning methods
│       ├── fig2_confusion_matrix.png         # Normalized classification confusion matrix
│       ├── fig3_regret_cdf.png                # Empirical cumulative regret distribution
│       ├── fig4_feature_importances.png      # Gini feature importance ranking
│       ├── fig5_qualitative_strip.png        # Qualitative restoration comparison strip
│       ├── fig6_downstream_utility.png       # Downstream ORB keypoint density gain
│       ├── fig7_downstream_geometric_eval.png # SIFT RANSAC inliers & Canny boundary F1
│       ├── fig8_robustness_noise_sweep.png   # Dynamic selector adaptation under noise
│       ├── fig9_edge_profiling.png           # Edge hardware latency, FPS & RAM profile
│       ├── fig10_multi_degradation_cascade.png # Phase 2 multi-hazard cascade benchmark
│       └── fig11_downstream_yolo_detection.png # Phase 1 YOLOv8 object detection metrics
├── scripts/
│   ├── analyze_dong_interaction.py    # Dong Invert-Dehaze feature interaction evaluator
│   ├── generate_presentation.py       # Programmatic PPTX presentation builder
│   ├── generate_handout_docx.py       # 1-page executive briefing Word generator
│   └── generate_walkthrough_docx.py   # Comprehensive step-by-step Word generator
├── src/
│   ├── __init__.py          # Package initialization and exports
│   ├── methods.py           # The Enhancement Method Bank (7 Operators including Invert-Dehaze)
│   ├── features.py          # The 8-dimensional Physical Degradation Descriptors
│   ├── labels.py            # MSE, PSNR, SSIM, Oracle and Regret computation
│   ├── dataset.py           # Preprocessing contract and scene grouping (278 scenes)
│   ├── train.py             # 5-Fold GroupKFold CV pipeline (zero leakage)
│   ├── evaluate.py          # Metrics, figures, ablations, error analysis
│   ├── predict_then_optimize.py     # Predict-Then-Optimize 6-head continuous regression engine
│   ├── downstream_matching.py       # Downstream ORB RANSAC inlier correspondence tracker
│   ├── latency_aware_selection.py   # Latency-constrained Pareto selection engine
│   ├── rq2_decision_optimality.py   # RQ2 semantic vigilance benchmark
│   ├── video_pipeline.py            # Phase 2 real-time video processor (EMA + Anti-Flicker)
│   ├── downstream_detection.py      # Downstream ORB keypoint utility evaluation
│   ├── downstream_semantic_eval.py  # SIFT RANSAC and Canny boundary evaluation
│   ├── downstream_yolo_detection.py # Phase 1 YOLOv8 object detection benchmark
│   ├── edge_hardware_profile.py     # Phase 1 ONNX CPU edge hardware profiling
│   ├── multi_degradation.py         # Phase 2 multi-hazard cross-transfer cascade
│   ├── robustness_sweep.py          # Gaussian sensor noise sensitivity sweep
│   └── demo.py              # Standalone single-image CLI inference tool
├── tests/
│   ├── __init__.py                  # Test suite package
│   ├── test_pipeline.py             # Unit tests for methods, features, metrics, and models
│   ├── test_lol_v2.py               # Unit tests for zero-shot LOL-v2 Real/Synthetic validation
│   ├── test_video_pipeline.py       # Unit tests for video stream processor & hysteresis
│   └── test_deferred_milestones.py  # Unit tests for PTO, ORB matching, and latency selector
├── app.py                   # Real-time web application dashboard (HTTP/JSON/HTML)
├── main.py                  # End-to-end execution script
├── requirements.txt         # Pinned exact library versions
├── LICENSE                  # MIT License
└── README.md                # Project documentation and reproduction guide
```

---

## 🔬 The Eight-Method Enhancement Bank

Every method runs with parameter locking to isolate method selection from parameter tuning:

| Method | Core Operation | Parameters | Target Degradation | Typical Failure Mode |
|:---|:---|:---|:---|:---|
| **Identity** | Invariant passthrough | None | Already well-exposed frames | Leaves degradation untreated |
| **Gamma** | Power-law LUT | $\gamma = 0.5$ | Global underexposure | Amplifies background sensor noise |
| **CLAHE** | Local histogram eq. on LAB luminance | `clipLimit=2.0`, `grid=(8,8)` | Low local tonal contrast | Boundary halos, grain amplification |
| **Hist Eq** | Global CDF on LAB luminance | None | Narrow histogram compression | Posterization, harsh over-contrast |
| **Retinex** | Single-Scale SSR illumination subtraction | $\sigma = 80.0$, min-max norm | Non-uniform illumination fields | Color desaturation, edge halos |
| **Bilat+Unsharp** | Edge-preserving denoising + sharpening | $d=9, \sigma_c=75, \sigma_s=75, \alpha=0.8$ | Soft and grainy scenes | Gradient overshoot, haloing |
| **Invert-Dehaze** | Optical Inversion Duality (Dong et al., 2011) | $\omega=0.95, t_0=0.10, r=7$ | Non-uniform shadows with residual signal | Noise division blowout if unconstrained |
| **High-Boost** | High-frequency acuity boosting | $k=2.0, \sigma=1.2$ on LAB luminance | Defocus optical blur | Sharpens sensor noise if unmitigated |

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

### 6. Downstream YOLOv8 Object Detection Benchmark (Phase 1)
Run end-to-end YOLOv8 object detection across the official 15 held-out scenes:
```bash
make downstream_yolo
```

### 7. Edge Embedded Hardware Profiling (Phase 1)
Profile ONNX export, CPU single-core latency, throughput FPS, and RAM footprint:
```bash
make edge_profile
```

### 8. Multi-Degradation Generalization & Hierarchical Cascade (Phase 2)
Run 10D physical feature extraction and cross-hazard evaluation across blur, haze, low-light, and composite noise:
```bash
make multi_degradation
```

### 9. Sensor Noise Robustness Sweep
Evaluate dynamic policy adaptation under varying Gaussian sensor noise:
```bash
make robustness
```

### 10. Executive Presentation & Walkthrough Word Generators
Generate the 12-slide widescreen PowerPoint presentation deck and the supervisor executive summary handout:
```bash
make presentation
make docx
```

### 11. Camera-Ready Overleaf Zip Archive
Build the camera-ready LaTeX package with all figures:
```bash
make package
```

### 12. Run Test Suite
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

- **Phase 0 (Core Algorithm Selection -- COMPLETED)**: Can cheap physical features predict the PSNR winner?  
  *Outcome*: **Validated** (+22.61% accuracy over fixed method, 57.57% regret reduction, $p = 9.44 \times 10^{-16}$).
- **Phase 1 (Downstream Object Detection & Edge Realization -- COMPLETED)**:
  *Outcome*: **Validated** — Boosted YOLOv8 object detection recall to **67.15%** (+18.62% over fixed Retinex, +12.74% over raw). ONNX runtime profile achieves **11.81 ms total pipeline latency (84.7 FPS)** on single CPU core with zero GPU dependencies and only 54.4 MB RAM.
- **Phase 2 (Multi-Degradation Cross-Transferability -- COMPLETED)**:
  *Outcome*: **Validated** — 10D extended physical feature cascade across low-light, optical blur, haze, and composite noise slashes blur regret by **4.70 dB** and delivers a **+0.915 dB net regret saving** across all optical hazards.
- **Phase 3 (Hybrid Classical-Neural Portfolios -- Semester 6/7 Future Work)**: Hybrid portfolios incorporating lightweight deep neural enhancers (Zero-DCE, SCI) alongside classical filters to dynamically balance fidelity against milliwatt power budgets.

---

## 👥 Team & Acknowledgments

- **Students**: Ananya Kasaudhan, Suyash Srivastav, Logi Vasan (5th Semester B.Tech, Department of Electronics and Communication Engineering (ECE), Gati Shakti Vishwavidyalaya, Vadodara 390004, Gujarat, India).
- **Supervisor**: Prof. Kishor P. Upla (*Senior Member, IEEE*), Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat, India.
- **Reference**: Prepared according to the handbook *"Image Enhancement and Per-Image Selection: from pixels and histograms to classical learning, regret, and the research frontier"* (SVNIT, Nov 2026).

