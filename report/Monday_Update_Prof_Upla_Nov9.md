# Weekly Project Status Memo (Day-5 Gate: 9 Nov 2026)

**To**: Prof. Kishor P. Upla, *Senior Member, IEEE* (Supervisor, SVNIT Surat)  
**From**: Team Triathlon — Suyash Srivastav, Ananya Kasaudhan, Logi Vasan (Gati Shakti Vishwavidyalaya, Vadodara)  
**Course**: 5th Semester B.Tech Capstone Project, ECE  
**Date**: Monday, November 9, 2026  
**Project**: Fidelity-Driven Algorithm Selection for Low-Light Image Enhancement via Handcrafted Physical Statistics  

---

### 1. Executive Summary & Day-5 Milestone Completion
We have completed all targets set for the **Day-5 Gate (9 Nov 2026)** per the Phase 0 Kickstart Plan:
1. **Methods & Preprocessing Contract**: Six enhancement operators (Identity, Gamma $\gamma=0.5$, CLAHE, Global Histogram Equalization, Retinex $\sigma=80$, Bilateral Filter + Unsharp Mask) locked and validated on LOL v1.
2. **Feature Extraction Pipeline**: 7 physical degradation descriptors ($\mu, \sigma, H, \sigma_n, D, |\nabla I|, C$) extracting in $<2\text{ ms}$ exclusively on degraded inputs.
3. **Rigorous Validation**: 5-fold `GroupKFold` cross-validation by physical scene across 3 random seeds (15 folds per model) with encapsulated scaler pipelines to prevent data leakage.
4. **Primary Benchmark**: Random Forest selector outperforms the best fixed baseline by **$+22.61\%$ accuracy** ($p = 9.44 \times 10^{-16}$) and cuts mean regret by **$57.35\%$** (saving $1.574\text{ dB}$, $p = 1.90 \times 10^{-11}$).
5. **Downstream CV Utility**: Demonstrates structural keypoint recovery jumping from $6.10\%$ on raw dark images up to $49.94\%$ with our selector.

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

### 3. Official Held-Out Test Set Verification (`eval15`)

On the unseen 15 official test scenes:
- **Best Fixed Method**: $46.67\%$ accuracy, $2.777\text{ dB}$ mean regret, $11.448\text{ dB}$ max regret.
- **Random Forest Selector**: **$66.67\%$ accuracy**, **$1.283\text{ dB}$ mean regret**, **$5.328\text{ dB}$ max regret**.
- *Risk Capping*: The maximum worst-case fidelity error is halved from $11.45\text{ dB}$ to $5.33\text{ dB}$.

---

### 4. Deliverables Status & Team Ownership

| Deliverable | Owner | Status | Artifact Location |
|:---|:---|:---:|:---|
| **Data & Scene Splits** | Ananya (GSV) | Completed | `data/lol_dataset/our485`, `eval15` |
| **Methods & Oracle Cache** | Suyash (GSV) | Completed | `src/methods.py`, `results/scores.csv` |
| **Features, CV & Demo** | Logi (GSV) | Completed | `src/features.py`, `src/train.py`, `src/demo.py` |
| **Interactive Web App** | Team | Running | `http://localhost:8501` (`app.py`) |
| **IEEE Conference Paper** | Team | Camera-Ready | `report/IEEE_Paper_Image_Enhancement_Selection.tex` |
| **Overleaf Archive** | Team | Ready | `report/IEEE_Paper_Overleaf_Package.zip` |

---

### 5. Next Steps for Week 2 (10–16 Nov 2026)
1. **LOL-v2 Synthetic Verification**: Test selector transferability on synthetic LOL-v2 pairs (Ananya).
2. **Ablation Deep-Dive**: Quantify impact of omitting Dark Channel Prior vs. Noise estimators in the report (Suyash).
3. **Stand-Alone Demo CLI Validation**: Validate zero-reference inference on consumer camera night images (Logi).

