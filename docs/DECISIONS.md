# Decisions Log & Gate Audit

*Project: Project Triathlon*  
*Date: 15 November 2026*  

---

## 1. Reference Implementation & Canonical Code Selection

### Search for Pre-Existing Reference
A thorough search of the workspace and commit history was conducted for `triathlon-phase0/`, `reference/`, `PROTOCOL.md`, and `src/selectors.py`. These files were not present in the current repository working directory. In accordance with Section 1 / Section 4 ("If you cannot find it, rebuild the pipeline from Appendix A"), we adopt the team's modular `src/` codebase and align every module strictly with Appendix A.

### Gate Audit (G1–G6)

| Gate | Criterion | Status | Technical Details & Remediation |
| :---: | :--- | :---: | :--- |
| **G1** | Pytest / Unittest passes | **PASS** | 22 unit tests passing (`Ran 22 tests in 20.0s, OK`). |
| **G2** | Scene-grouped CV with non-overlapping assertion | **PASS (Remediated)** | Previously `GroupKFold` relied on scikit-learn's disjoint group invariant. Added explicit assertion in fold creation: `assert len(set(groups[tr_idx]).intersection(set(groups[va_idx]))) == 0`. |
| **G3** | Scalers & fitted preprocessing strictly inside training folds | **PASS** | `StandardScaler()` is strictly encapsulated inside `sklearn.pipeline.make_pipeline` per fold. |
| **G4** | Features use degraded input only (no forbidden columns) | **PASS (Remediated)** | Feature extraction signature takes only degraded BGR images. Pipeline asserts that no feature column name contains `psnr`, `ssim`, `label`, `oracle`, `uid`, `name`, `path`, or `regret`. |
| **G5** | Deterministic reproducibility across runs | **PASS** | Fixed random seeds (`seed_everything`, fixed CV seeds). |
| **G6** | Bank and parameters match Appendix A | **PASS** | Frozen 6-method portfolio (`identity`, `gamma 0.5`, `clahe 2.0 8x8`, `hist_eq`, `retinex 80`, `bilat_unsharp 9,75,75,1.5,0.8`). |

---

## 2. Extensions Segregation & Protected Assets Restoration

In accordance with user requirements, all components related to classical CV (ORB/RANSAC geometric matching, video temporal smoother, web application, latency benchmarks, and submission deliverables) are maintained under strict classical ML constraints (ORB Inlier Tracking completely removed).

Any non-duplicate files previously placed in `_archive/` have been restored to structured extension and submission locations:

### Directory Mapping for Protected Assets

| Protected Asset Category | Permanent Location | Files Included | Purpose & Execution Mode |
| :--- | :--- | :--- | :--- |
| **Downstream Machine Vision** | `extensions/downstream/` | `downstream_matching.py`, `downstream_detection.py`, `downstream_semantic_eval.py`, `README.md` | Pure classical vision: ORB/RANSAC geometric inlier recovery and keypoint repeatability on CPU. |
| **Video Temporal Smoother** | `extensions/video/` | `video_pipeline.py`, `generate_video_demo.py`, `video_demo_anti_flicker.mp4`, `README.md` | Real-time temporal feature smoothing (EMA) and Markovian hysteresis gating to prevent inter-frame visual strobing (>100 FPS on CPU). |
| **Latency Benchmarking** | `extensions/latency/` | `edge_hardware_profile.py`, `latency_aware_selection.py`, `edge_hardware_profile.csv`, `latency_regret_frontier.csv`, `README.md` | CPU latency profiling across 500 runs and Pareto frontier optimization under computational constraints. |
| **Interactive Web App** | `apps/web/` | `app.py`, `README.md` | Zero-dependency standalone Python `http.server` interactive web application for live drag-and-drop enhancement and visual comparison. |
| **Submission Deliverables** | `paper/submission/` | `IEEE_Paper_To_Submit_Anonymized.pdf`, `Supplementary_Visual_Atlas.pdf`, `IEEE_Paper_CameraReady_With_Authors.pdf`, `Project_Triathlon_Basic_vs_Extra_Comparison.pdf`, `IEEE_LaTeX_Source_Package.zip`, `README_SUBMISSION_GUIDE.md` | Complete camera-ready and anonymized submission PDFs, supplementary visual atlas, and Overleaf source bundle for IEEE Transactions. |
| **Executive Deliverables** | `docs/deliverables/` | `Project_Triathlon_Master_SOP.pdf`, `Project_Triathlon_Executive_Presentation.pptx`, `Project_Triathlon_Comprehensive_Deliverables_Report.pdf`, `Project_Triathlon_Step_by_Step_Walkthrough.docx`, `Executive_Summary_Prof_Upla.docx`, `Suyash/`, `Ananya/` | Project presentations, walkthrough documents, student completion reports, and master SOPs. |

### Dependency Isolation Contract
- **Dependencies (`streamlit`)**: Maintained in `extensions/requirements-extensions.txt`.
- **Core Library Independence**: `src/` has **ZERO** deep learning frameworks (`torch`, `tensorflow`). Core training, evaluation, cross-validation, and paper asset generation run purely on lightweight CPU libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `opencv-python`).
- **Rule on Duplicates**: Only exact SHA-256 duplicate copies remain archived in `_archive/report/` (`IEEEtran.cls`, duplicate copies of submission PDFs). A primary copy of every file is live in `paper/`, `paper/submission/`, or `docs/deliverables/`.

---

## 4. Reference Image dHash Scene Grouping Audit (Hamming <= 10)
- **Methodology**: 64-bit dHash computed on $9 \times 8$ grayscale thumbnails of clean normal-light reference images. Adjacent pixel difference bits: `diff = (img[:, 1:] > img[:, :-1]).flatten()`.
- **Graph & Connected Components**: Pairs with Hamming distance $\le 10$ form edges. Connected components identify physical scene clusters.
- **Results on LOL-v1 (`our485`)**:
  - Total pairs: 485
  - Total physical scene clusters: 228
  - Multi-image clusters: 173 scenes (largest cluster contains 6 multi-exposure instances of the identical physical scene).
  - Single-image clusters: 55 scenes.
- **Contact Sheet Inspection (`results/groups_lol_v1_train.png`)**:
  - Examination of the generated contact sheet confirms that images grouped within the same cluster represent identical spatial viewpoint scenes captured under differing camera exposures and subtle viewpoint jitter.
  - No synthetic scene leakage crosses cluster boundaries.
  - Per Rule 2 & Section 6, this threshold ($\le 10$) is locked without post-hoc tuning.

---

## 5. Experiment E5 Control (a) Score Permutation Analysis & Null Distribution Audit
- **Issue**: In `results/decision/verdict.json`, `all_controls_pass` was evaluated as `False` because Control (a) failed its automated single-seed check (`status: FAIL`).
- **Investigation**:
  - In Control (a), score rows were permuted across images using a single seed (seed 42).
  - Under permuted labels, feature-label correspondence was completely decoupled. Logistic Regression fell back on its prior/intercept, predicting primarily the majority class (`retinex`, 45.6%) and second-majority class (`hist_eq`, 35.3%).
  - When evaluated against true unpermuted image ground truth, LR's mixture of `retinex` and `hist_eq` achieved 2.33 dB regret versus 2.74 dB for static `retinex` (an apparent +0.507 dB advantage over Best-Fixed on that single realization).
  - The script's strict test `ca_pass = (lr_a_lo <= 0.0) and (rf_a_lo <= 0.0)` tested the lower bound of a cluster bootstrap on that single permutation; because `lr_a_lo = +0.232 > 0.0`, it triggered a failure.
- **100-Permutation Monte Carlo Null Distribution**:
  - To test whether this was an artifact of a single seed or genuine signal, a 100-permutation null distribution was computed:
    - **Chance Levels**: Uniform 1/6 chance = 16.67%; Majority class rate (`retinex`) = 45.57%.
    - **Top-1 Accuracy on Permuted Labels**: LR mean = 43.70% (std = 1.46%, 95% range: [40.92%, 46.39%]); RF mean = 40.53% (std = 2.44%, 95% range: [36.18%, 44.85%]). Both center squarely on the majority baseline.
    - **Advantage over Best-Fixed (dB)**:
      - LR: mean = +0.1568 dB (std = 0.2343 dB, 95% empirical range: [-0.2801, +0.5773] dB). Zero is well within the null distribution ($z = 1.49$, empirical $p \approx 0.07$).
      - RF: mean = -0.1304 dB (std = 0.2314 dB, 95% empirical range: [-0.5864, +0.2889] dB). Zero is well within the null distribution.
- **Resolution & Protocol Adherence**:
  - In strict compliance with Non-Negotiable Rule 2, `docs/DECISION_RULE.md` was **NOT modified**.
  - The pre-committed verdict remains **`TIE` $\rightarrow$ Deploy Multinomial Logistic Regression** because Conditions 1, 2, and 3 failed independently on their own, and under a tie Occam's razor selects LR.
  - This deviation is recorded here in accordance with instructions.

---

## 6. Experiment E4 External Validity & Sensitivity Analysis (Both Directions)

### Forward Direction: LOL-v1 Train (485) $\rightarrow$ LOL-v2 Real Test (100)
- **Nearest-Neighbor Hamming Distance Histogram (LOL-v2 Real Test vs. LOL-v1 our485)**:
  - `[0, 2)`: 88 images (exact or 1-bit difference from training images)
  - `[2, 4)`: 3 images
  - `[4, 12)`: 0 images
  - `[12, 14)`: 4 images
  - `[14, 16)`: 3 images
  - `[20, 22)`: 2 images
  - Min dist: 0, Max dist: 21, Median: 0.0.
- **Why $d_{\text{real\_test}} = 0.0000$ dB**:
  - Across all threshold settings (Exact $d=0$, Thr 4, Thr 6, Thr 10), exactly 91 images are excluded, leaving exactly 9 images ($N_{\text{after}} = 9$, not an empty set).
  - On these 9 images, both Logistic Regression and Random Forest select **100% `retinex`** (`['retinex', ..., 'retinex']`), yielding identical predictions on 100% of the images.
  - Because their selections are identical, both models achieve identical mean regret (2.839 dB), giving $d_{\text{real\_test}} = \text{RF} - \text{LR} = 0.0000$ dB and 95% CI $[0.0000, 0.0000]$ dB.

### Reverse Direction: LOL-v2 Real Train (689) $\rightarrow$ LOL-v1 our485 (485)
- **Nearest-Neighbor Hamming Distance Histogram (LOL-v1 our485 vs. LOL-v2 Real Train)**:
  - `[0, 2)`: 399 images
  - `[2, 12)`: 0 images
  - `[12, 22)`: 86 images
- **Sensitivity Analysis Across Exclusion Thresholds**:
  - **No Exclusion (Unfiltered)**: $N_{\text{before}} = 485$, Excluded = 0, $N_{\text{after}} = 485$ | LR: 1.506 dB, RF: 0.585 dB | $d_{\text{reverse}} = \mathbf{-0.9208\text{ dB}}$ (or $-0.5669$ dB on original E4 split) [95% CI: $-1.1371, -0.7097$ dB] | Same picks: 66.0%. This setting suffered massive train-test contamination (399 of the 485 test images were present in the training set).
  - **Exact / Thr 4 / Thr 6 / Thr 10 (De-duplicated)**: $N_{\text{before}} = 485$, Excluded = 399, $N_{\text{after}} = 86$ clean images | LR: 3.163 dB, RF: 3.301 dB | $d = \mathbf{+0.1380\text{ dB}}$ [95% CI: $-0.2202, +0.5999$ dB] | Same picks: 76.7%.
  - When train-test contamination is removed, Logistic Regression outperforms Random Forest ($d > 0$), and the difference is statistically indistinguishable from zero (95% CI spans zero). RF not demonstrably better in-distribution.

---

## 7. Protected Assets Audit & Relocation Verification

| Asset Keyword | Permanent Active Location | Archived / Duplicate Location | Status & Notes |
| :--- | :--- | :--- | :--- |
| **ORB Inlier Tracking** | None | Removed | Completely excised in accordance with user requirements (strictly classical ML & CV). |
| **orb / ransac** | `extensions/downstream/downstream_matching.py` | None | Fully active on CPU; +72.9% inlier gain verified in 68 ms. |
| **flicker** | `extensions/video/video_demo_anti_flicker.mp4`, `extensions/video/generate_video_demo.py`, `extensions/video/video_pipeline.py` | None | Temporal hysteresis gating active (0 switches vs 8 raw switches across 60 frames). |
| **app.py** | `apps/web/app.py` (canonical app) & `app.py` (root forwarder) | None | Verified via socket HTTP test on port 8599 (HTTP 200 OK). |
| **JOURNAL_SUBMISSION** | Consolidated into single canonical paper (`paper/main.tex` and `paper/main.pdf`) | Redundant split variants removed | Consolidated into single master document for phased development. |

---

## 8. Single-Paper Phased Development Consolidation

* **Directive**: As instructed, all auxiliary/split paper documents (`docs/deliverables/IEEE_Paper_Anonymized.tex`, `docs/deliverables/IEEE_Paper_Image_Enhancement_Selection.tex`, `docs/deliverables/Project_Triathlon_Basic_vs_Extra_Comparison.tex`, `paper/submission/`, `_archive/report/`, and auxiliary zip bundles) have been removed.
* **Canonical Architecture**: The project now maintains **EXACTLY ONE** master research paper:
  - Source: [`paper/main.tex`](file:///Users/home/Desktop/Aiml_Sem5/paper/main.tex)
  - Compiled PDF: [`paper/main.pdf`](file:///Users/home/Desktop/Aiml_Sem5/paper/main.pdf) (20 pages, 11.95 MB)
* **Phased Expansion**: As subsequent project phases (Phase 2 Sem 7, Phase 3 Sem 8) are executed, all experimental additions, figures, and analyses are directly integrated into `paper/main.tex` and compiled to `paper/main.pdf`.



---

## 5. UI Audit & Live Demo Remediation (17 Nov 2026)
- **Branch**:  (tagged ).
- **Defects Fixed**: DEF-01 through DEF-10 resolved and verified by automated unit test suite .
- **Default Sample Policy**: Deterministic selection of held-out eval15 Scene 547 based on median brightness (15.67 / 255).
- **Presentation Boost Contract**: Visual gamma 0.4 display boost is isolated to display rendering; ML feature extraction and inference operate strictly on raw uncompressed pixels.
- **Scientific Facts Invariant**: apps/web/facts.py dynamically supplies all metrics from results/<run> and models/selector.json. Zero hardcoded literal numbers in app markup.


---

## 5. UI Audit & Live Demo Remediation (17 Nov 2026)
- **Branch**: `ui-audit` (tagged `pre-ui`).
- **Defects Fixed**: DEF-01 through DEF-10 resolved and verified by automated unit test suite `tests/test_ui_audit.py`.
- **Default Sample Policy**: Deterministic selection of held-out eval15 Scene 547 based on median brightness (15.67 / 255).
- **Presentation Boost Contract**: Visual gamma 0.4 display boost is isolated to display rendering; ML feature extraction and inference operate strictly on raw uncompressed pixels.
- **Scientific Facts Invariant**: apps/web/facts.py dynamically supplies all metrics from results/<run> and models/selector.json. Zero hardcoded literal numbers in app markup.
