# Weekly Project Status Memo (Day-5 Gate: 9 Nov 2026)

**To**: Prof. Kishor P. Upla, *Senior Member, IEEE* (Supervisor, SVNIT Surat)  
**From**: Team Triathlon — Suyash Srivastav, Ananya Kasaudhan, Logi Vasan (Gati Shakti Vishwavidyalaya, Vadodara)  
**Course**: 5th Semester B.Tech Capstone Project, ECE  
**Date**: Monday, November 9, 2026  
**Project**: Fidelity-Driven Algorithm Selection for Low-Light Image Enhancement via Handcrafted Physical Statistics  

---

### 1. Executive Summary & Multi-Phase Milestone Completion
We have completed all targets set for the **Day-5 Gate (9 Nov 2026)** per the Phase 0 Kickstart Plan, and substantially advanced into **Phase 1** and **Phase 2**:
1. **Methods & Preprocessing Contract (Phase 0)**: Six enhancement operators locked and validated on LOL v1.
2. **Feature Extraction Pipeline (Phase 0)**: 7 physical degradation categories (8 scalar descriptors) extracting in $<2\text{ ms}$ exclusively on degraded inputs.
3. **Rigorous Validation (Phase 0)**: 5-fold `GroupKFold` cross-validation by physical scene across 3 random seeds (15 folds per model) with encapsulated scaler pipelines to prevent data leakage.
4. **Primary Benchmark (Phase 0)**: Random Forest selector outperforms the best fixed baseline by **$+22.61\%$ accuracy** ($p = 9.44 \times 10^{-16}$) and cuts mean regret by **$57.35\%$** (saving $1.574\text{ dB}$, $p = 1.90 \times 10^{-11}$).
5. **Downstream YOLOv8 Object Detection Benchmark (Phase 1)**: Demonstrated that while static Retinex degrades per-scene detection recall to $48.53\%$ (-5.88% vs raw), our selector achieves **$67.15\%$ recall** (+18.62% over Retinex, 37 boxes vs 30 boxes).
6. **Edge Hardware Profiling & ONNX Acceleration (Phase 1)**: Serialized `models/rf_selector.onnx` (4.20 MB), running at **$11.81\text{ ms}$ total pipeline latency ($84.7\text{ FPS}$)** on a single-core CPU with zero GPU dependencies and $54.4\text{ MB}$ peak RAM.
7. **Multi-Degradation Cross-Transferability & Hierarchical Cascading (Phase 2)**: Extended to 10D physical features and 8 operators across low-light, blur, haze, and composite noise, slashing blur regret by **$4.70\text{ dB}$** and yielding a **$+0.915\text{ dB}$ net regret saving** across all optical hazards.

---

### 2. Primary 15-Run Cross-Validation Results

| Model / Strategy | Top-1 Accuracy Mean (%) | Accuracy Std (%) | Mean Regret (dB) | Regret Std (dB) | Max Regret (dB) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Random Choice (Chance)** | 17.66 | $\pm 4.77$ | 6.916 | $\pm 0.511$ | 18.085 |
| **Best Fixed Single Method (Retinex)** | 45.57 | $\pm 3.41$ | 2.734 | $\pm 0.118$ | 17.815 |
| **Hand-Written Heuristic Rule** | 12.16 | $\pm 3.29$ | 6.926 | $\pm 0.360$ | 17.064 |
| **Multinomial Logistic Regression** | 61.86 | $\pm 2.53$ | 1.421 | $\pm 0.213$ | 15.032 |
| **Random Forest Selector (Proposed)** | **68.18** | $\mathbf{\pm 3.49}$ | **1.160** | $\mathbf{\pm 0.236}$ | **15.181** |
| **XGBoost Classifier** | 67.70 | $\pm 4.90$ | 1.249 | $\pm 0.322$ | 14.673 |
| **Theoretical Oracle Upper Bound** | **100.00** | $\pm 0.00$ | **0.000** | $\pm 0.000$ | **0.000** |

*Key Takeaway*: The proposed selector recovers **$57.4\%$** of the theoretical headroom between the fixed baseline and the Oracle, saving an average of **$1.574\text{ dB}$** in image fidelity per frame.

---

### 3. Downstream Perception & Multi-Hazard Benchmarks

- **YOLOv8 Object Detection (`eval15`)**:
  - Raw Low-Light: 26 boxes (54.41% recall)
  - Static Retinex: 30 boxes (48.53% recall — noise/halo drop)
  - **Proposed RF Selector: 37 boxes (67.15% recall, +18.62% over Retinex)**
- **Edge Embedded Profiling (Apple Silicon Single-Core CPU)**:
  - Total latency: **11.81 ms** (84.7 FPS) | Peak RAM: **54.4 MB** | Model: **4.20 MB ONNX**
  - Compared to Zero-DCE (42.8 ms CPU), RetinexNet (48.0 ms GPU), Restormer (320.0 ms GPU).
- **Multi-Hazard Cross-Transfer Cascade**:
  - Defocus blur regret: cut from $15.10\text{ dB} \to 10.41\text{ dB}$ (**saves 4.70 dB**).
  - Atmospheric haze regret: cut from $3.57\text{ dB} \to 2.40\text{ dB}$ (**saves 1.17 dB**).
  - Overall net regret saving: **+0.915 dB**.

---

### 4. Deliverables Status & Team Ownership

| Deliverable | Owner | Status | Artifact Location |
|:---|:---|:---:|:---|
| **Data & Scene Splits** | Ananya (GSV) | Completed | `data/lol_dataset/our485`, `eval15` |
| **Methods & Oracle Cache** | Suyash (GSV) | Completed | `src/methods.py`, `results/scores.csv` |
| **Features, CV & Demo** | Logi (GSV) | Completed | `src/features.py`, `src/train.py`, `src/demo.py` |
| **YOLOv8 Detection & Edge Profiling** | Team | Completed | `src/downstream_yolo_detection.py`, `src/edge_hardware_profile.py` |
| **Multi-Hazard Cascade Engine** | Team | Completed | `src/multi_degradation.py`, `models/rf_selector.onnx` |
| **Interactive Web App** | Team | Running | `http://localhost:8501` (`app.py`) |
| **IEEE Transactions Paper (11 Figs, 9 Tabs)** | Team | Camera-Ready | `report/IEEE_Paper_Image_Enhancement_Selection.tex` |
| **Overleaf Archive** | Team | Ready | `report/IEEE_Paper_Overleaf_Package.zip` |
| **15-Slide Presentation Deck** | Team | Ready | `report/Project_Triathlon_Executive_Presentation.pptx` |
| **Comprehensive Word Walkthrough** | Team | Ready | `report/Project_Triathlon_Step_by_Step_Walkthrough.docx` |

---

### 5. Next Steps for Supervisor Review & Capstone Defense
1. **Overleaf PDF Verification**: Finalize supervisor review on the compiled camera-ready PDF.
2. **Interactive Live Demonstration**: Present real-time Streamlit app (`app.py`) with before/after slider and YOLOv8 detection overlay during the project evaluation.
3. **Conference Submission**: Prepare IEEE Transactions / Conference submission package.
