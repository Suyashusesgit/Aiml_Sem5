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

In accordance with user requirements, all components related to YOLOv8, ORB/RANSAC geometric matching, the video temporal smoother, the web application, latency benchmarks, and the submission deliverables are **STRICTLY PROTECTED**.

Any non-duplicate files previously placed in `_archive/` have been restored to structured extension and submission locations:

### Directory Mapping for Protected Assets

| Protected Asset Category | Permanent Location | Files Included | Purpose & Execution Mode |
| :--- | :--- | :--- | :--- |
| **Downstream Machine Vision** | `extensions/downstream/` | `yolov8n.pt`, `downstream_yolo_detection.py`, `downstream_matching.py`, `downstream_detection.py`, `downstream_semantic_eval.py`, `README.md` | YOLOv8 object detection on CPU (`device="cpu"`) and ORB/RANSAC geometric inlier recovery. |
| **Video Temporal Smoother** | `extensions/video/` | `video_pipeline.py`, `generate_video_demo.py`, `video_demo_anti_flicker.mp4`, `README.md` | Real-time temporal feature smoothing (EMA) and Markovian hysteresis gating to prevent inter-frame visual strobing (>100 FPS on CPU). |
| **Latency Benchmarking** | `extensions/latency/` | `edge_hardware_profile.py`, `latency_aware_selection.py`, `edge_hardware_profile.csv`, `latency_regret_frontier.csv`, `README.md` | CPU latency profiling across 500 runs and Pareto frontier optimization under computational constraints. |
| **Interactive Web App** | `apps/web/` | `app.py`, `README.md` | Zero-dependency standalone Python `http.server` interactive web application for live drag-and-drop enhancement and visual comparison. |
| **Submission Deliverables** | `paper/submission/` | `IEEE_Paper_To_Submit_Anonymized.pdf`, `Supplementary_Visual_Atlas.pdf`, `IEEE_Paper_CameraReady_With_Authors.pdf`, `Project_Triathlon_Basic_vs_Extra_Comparison.pdf`, `IEEE_LaTeX_Source_Package.zip`, `README_SUBMISSION_GUIDE.md` | Complete camera-ready and anonymized submission PDFs, supplementary visual atlas, and Overleaf source bundle for IEEE Transactions. |
| **Executive Deliverables** | `docs/deliverables/` | `Project_Triathlon_Master_SOP.pdf`, `Project_Triathlon_Executive_Presentation.pptx`, `Project_Triathlon_Comprehensive_Deliverables_Report.pdf`, `Project_Triathlon_Step_by_Step_Walkthrough.docx`, `Executive_Summary_Prof_Upla.docx`, `Suyash/`, `Ananya/` | Project presentations, walkthrough documents, student completion reports, and master SOPs. |

### Dependency Isolation Contract
- **Heavy Dependencies (`ultralytics`, `torch`, `torchvision`, `streamlit`)**: Segregated strictly to `extensions/requirements-extensions.txt`.
- **Core Library Independence**: `src/` has **ZERO** imports of `torch` or `ultralytics`. Core training, evaluation, cross-validation, and paper asset generation run purely on lightweight CPU libraries (`numpy`, `scipy`, `pandas`, `scikit-learn`, `opencv-python`).
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
