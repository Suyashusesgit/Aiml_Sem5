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

## 2. Extensions Segregation
Modules not part of the frozen Phase 0 core selector protocol are cleanly segregated into `extensions/`:
- `extensions/app.py`: Interactive browser web application.
- `extensions/video_pipeline.py`: Video temporal anti-flicker smoothing.
- `extensions/downstream_detection.py` & `downstream_yolo_detection.py`: YOLOv8 object detection benchmark.
- `extensions/downstream_matching.py`: ORB/AKAZE keypoint feature matching.
- `extensions/edge_hardware_profile.py` & `latency_aware_selection.py`: ONNX export and CPU latency profiling.
- `extensions/multi_degradation.py`: Multi-hazard regime cascading.

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
