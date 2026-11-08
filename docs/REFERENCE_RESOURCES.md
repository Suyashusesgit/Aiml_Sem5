# Project Triathlon — Reference Resources

**Team Triathlon**: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan  
**Institution**: Department of Electronics and Communication Engineering, Gati Shakti Vishwavidyalaya, Vadodara 390004, Gujarat, India  
**Course**: AIML 5th Semester Capstone Project  
**Supervisor**: Prof. Kishor P. Upla, *Senior Member, IEEE*  
**Supervisor Affiliation**: Department of Electronics Engineering, Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat 395007, Gujarat, India  
**Date**: November 2026  

---

## 📖 Primary Reference (Start Here)

### `Image_Enhancement_Handbook.pdf`
56 pages, 24 chapters in 5 parts. Built specifically for this project:
- **Part 1**: Pixels, histograms, colour spaces.
- **Part 2**: Six classical enhancement methods (gamma, histogram equalization, CLAHE, Retinex, bilateral + unsharp, identity).
- **Part 3**: PSNR, SSIM, NR-IQA, oracle/regret analysis.
- **Part 4**: Seven handcrafted features, leakage prevention, GroupKFold protocol.
- **Part 5**: Phase 0 build guide, glossary, 34 references.

> *Rule*: Read Parts 1–4 before writing code. Part 5 is the implementation checklist.

---

## 📚 Textbooks & Core Reading

| Book | Key Chapters | Rationale |
|:---|:---|:---|
| **Digital Image Processing** (Gonzalez & Woods, 4th ed.) | Ch. 2, 3, 5, 6 | Standard reference for pixels, histograms, spatial filters, restoration, and colour models. |
| **Computer Vision: Algorithms and Applications** (Szeliski, 2nd ed.) | Ch. 3 (Computational Photography) | Modern explanations of filtering, edge extraction, and image processing pipelines. |
| **An Introduction to Statistical Learning** (James et al.) | Ch. 2, 4, 5, 8 | Logistic regression, cross-validation, random forests, boosting — at the right mathematical depth. |
| **Hands-On Machine Learning** (Géron, 3rd ed.) | Ch. 2, 3, 6, 7 | Practical scikit-learn workflow, classification metrics, trees, and ensembles. |

*Highest-value chapters*: Gonzalez Ch. 3; ISLR Ch. 4, 5, 8.

---

## 📄 Key Research Papers (Read in Order)

| Paper | Reference / Link | Why It Matters for Project Triathlon |
|:---|:---|:---|
| **BlindPSNR: A No-Reference Fidelity Predictor for Low-Light Image Enhancement** | [arXiv:2607.27628](https://arxiv.org/abs/2607.27628) (Lyu et al., 2026) | Closest existing work. A CNN predicts no-ref PSNR for parameter selection. Our project represents the **computationally cheap physical feature end** of the cost curve. |
| **What Does LOL Measure? Hidden Label Shifts in Low-Light Enhancement Benchmarks** | [arXiv:2610.38121](https://arxiv.org/abs/2610.38121) | Critical for data protocol. Documents label shifts, scene groupings, and overlap in LOL-v1 vs. LOL-v2. |
| **Image Quality Assessment: From Error Visibility to Structural Similarity** | Wang et al., 2004 (*IEEE TIP*) | The seminal SSIM paper. Explains what SSIM measures and justifies its role as our tie-break within $0.05\text{ dB}$ PSNR. |
| **Mean Squared Error: Love It or Leave It?** | Wang & Bovik, 2009 (*IEEE SPM*) | Why PSNR is necessary but incomplete; establishes the rationale for "beyond PSNR" regret framing. |
| **Contrast Limited Adaptive Histogram Equalization (CLAHE)** | Zuiderveld, 1994 (*Graphics Gems IV*) | The foundational CLAHE operator: clip-limit and tile-grid intuition to prevent noise amplification. |
| **Properties and Performance of a Center/Surround Retinex** | Jobson et al., 1997 (*IEEE TIP*) | Single-Scale Retinex baseline formulation with Gaussian surround subtraction in logarithmic space. |
| **Bilateral Filtering for Gray and Color Images** | Tomasi & Manduchi, 1998 (*ICCV*) | Edge-preserving smoothing filter: range and spatial Gaussian kernel balance. |
| **Single Image Haze Removal Using Dark Channel Prior** | He et al., 2009 (*IEEE TPAMI*) | Dark channel feature definition: provides the minimum-channel patch erosion statistic. |
| **Deep Retinex Decomposition for Low-Light Enhancement** | Chen et al., 2018 (*BMVC*) | Official LOL dataset publication and benchmark archive. |

---

## 🎥 Video Lectures

- **First Principles of Computer Vision** — Prof. Shree Nayar (Columbia University). YouTube. (Filtering, histograms, edges, colour spaces).
- **NPTEL: Digital Image Processing** — Prof. P.K. Biswas (IIT Kharagpur). (Histogram processing, spatial filtering, restoration).
- **StatQuest with Josh Starmer** — YouTube. (Decision Trees, Random Forests, XGBoost, Cross-Validation, Confusion Matrices, ROC/AUC).
- **PyImageSearch** — Practical OpenCV guides (Gamma correction, histogram equalization, CLAHE, bilateral filtering).
- **Andrew Ng Machine Learning Specialization** — Coursera (Logistic regression, $L_2$ regularization, model evaluation).

---

## 🔗 Documentation & API References

- **OpenCV Python Tutorials**: [docs.opencv.org](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- **CLAHE & HistEq**: [OpenCV Histogram Equalization](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)
- **Bilateral Filtering**: [OpenCV Smoothing](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)
- **scikit-image Metrics**: [scikit-image PSNR/SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
- **scikit-learn GroupKFold**: [scikit-learn GroupKFold Guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- **scikit-learn Cross-Validation & Common Pitfalls**: [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) | [Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- **XGBoost Documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)

---

## 🗄 Datasets & Protocol

| Dataset | Link | Project Role |
|:---|:---|:---|
| **LOL-v1** (485 train / 15 test) | [Official BMVC Page](https://daooshee.github.io/BMVC2018website/) | **Primary**: Train split partitioned strictly by scene via `GroupKFold`. |
| **LOL-v2 Synthetic** | [Hugging Face](https://huggingface.co/datasets/okhater/lolv2-synthetic) | **Clean External Check**: Cross-dataset verification. |
| **LOL-v2 Real** | [Hugging Face](https://huggingface.co/datasets/okhater/lolv2-real) | **Stress Test Only**: Domain shift stress testing. |

---

## ⚠️ Five Traps to Avoid

1. **BGR vs. RGB**: OpenCV loads images in BGR format; Matplotlib expects RGB. Always convert (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`) before plotting.
2. **PSNR Data Range**: Ensure `max_val = 255.0` (or `data_range=255`) for 8-bit uint8 images.
3. **Features on Wrong Image**: Features $\phi(x)$ must be computed **strictly on the degraded input**, never on the enhanced output or clean ground truth.
4. **Random Train/Test Split**: Always use `GroupKFold` grouped by physical scene ID; naive random splits cause near-duplicate scene leakage.
5. **Tuning on Test Set**: Fix the method bank parameters, feature definitions, and model hyperparameters prior to evaluating on the held-out `eval15` test set.

---

## 👥 Team Ownership & Responsibilities

| Role | Owner | Core Responsibilities |
|:---|:---|:---|
| **Data & Splits** | **Ananya Kasaudhan (GSV)** | LOL dataset download, integrity verification, scene grouping splits, 50-image pilot study, LOL-v2 prep. |
| **Methods, Labels, PSNR** | **Suyash Srivastav (GSV)** | Six-method enhancement bank implementation, PSNR/SSIM cache computation, oracle labeling, SSIM tie-breaking. |
| **Features, Models, Demo** | **Logi Vasan (GSV)** | Handcrafted physical feature extraction, LogReg/RF/XGBoost modeling, GroupKFold validation, demo script (`src/demo.py`), end-to-end integration. |

---

## 📅 Deliverables Tracker (Target: 20 Nov 2026)

- [x] **Phase 0 Research Paper**: Publication-ready IEEE draft (`report/IEEE_Paper_Image_Enhancement_Selection.tex`, `.md`).
- [x] **Baseline Regret Table (Day-5 Gate: 9 Nov)**: 15-run GroupKFold benchmark completed (`results/metrics_table.csv`).
- [x] **Trained Models + Feature Pipeline**: Full modular codebase in `src/`.
- [x] **Working Demo Script**: Interactive CLI tool (`src/demo.py`).
- [x] **Interactive Web App**: Running at `http://localhost:8501` (`app.py`).
- [x] **One-Page Monday Update for Prof. Upla**: Day-5 gate progress report (`report/Monday_Update_Prof_Upla_Nov9.md`).

