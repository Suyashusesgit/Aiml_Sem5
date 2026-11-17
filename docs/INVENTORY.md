# Repository Inventory & Baseline Audit

*Date: 15 November 2026*  
*Project: Project Triathlon (Per-Image Low-Light Enhancement Selection)*  
*Department of Artificial Intelligence & Machine Learning, Gati Shakti Vishwavidyalaya, Vadodara*  
*Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)*  
*Initial Snapshot Git Commit: `feb9b09` (Tag: `pre-cleanup`)*

---

## 1. Top-Level Directory and File Inventory

| Path | Type | Disk Size | Primary Purpose & Contents |
| :--- | :---: | :---: | :--- |
| `.gitignore` | File | 655 B | Specifies intentionally untracked files for Git version control. |
| `.vscode/` | Dir | 8.0 KB | VS Code project settings and debugging task configurations (`settings.json`, `tasks.json`). |
| `JOURNAL_SUBMISSION/` | Dir | 48 MB | Bundled IEEE journal submission archives and PDFs (redundant duplicates of `report/`). |
| `LICENSE` | File | 1.1 KB | MIT open-source license specification. |
| `Makefile` | File | 3.8 KB | Build targets for testing, linting, evaluation, and documentation generation. |
| `README.md` | File | 24 KB | Top-level project overview, pipeline description, setup instructions, and benchmark summaries. |
| `app.py` | File | 66 KB | Self-contained interactive HTTP web server (Python `http.server`) for browser demonstrations. |
| `configs/` | Dir | 4.0 KB | Experiment and model configuration YAML files (`base.yaml`). |
| `data/` | Dir | 1.6 GB | **READ-ONLY** dataset binaries (`lol_dataset/` our485/eval15, `LOL-v2/` Real/Synthetic, CSV scene manifests). |
| `docs/` | Dir | 24 KB | Project governance, SOPs, architectural notes, and experimental logs. |
| `main.py` | File | 5.1 KB | Legacy monolithic experiment driver for feature extraction, CV, figures, and ablations. |
| `models/` | Dir | 14 MB | Serialized production model artifacts (`rf_selector.joblib`, `rf_selector.onnx`, `pto_regressors.joblib`, LOL-v2 checkpoints). |
| `notebooks/` | Dir | 16 KB | Jupyter analysis notebook (`exploration_and_benchmarks.ipynb`). |
| `pyproject.toml` | File | 397 B | PEP 518/621 build system metadata and tool configurations. |
| `pyrefly.toml` | File | 369 B | Pyrefly static type-checker configuration. |
| `pyrightconfig.json` | File | 875 B | Microsoft Pyright static type analysis configuration. |
| `report/` | Dir | 90 MB | IEEE Transactions LaTeX paper (`IEEE_Paper_Image_Enhancement_Selection.tex`), figures, generated PDFs, and slides. |
| `requirements.txt` | File | 146 B | Core runtime dependencies (NumPy, OpenCV, scikit-learn, XGBoost, Matplotlib, Pandas, Joblib). |
| `results/` | Dir | 43 MB | Generated CSV result tables, score/feature caches, ROC/CDF plots, figures, and model checkpoints. |
| `scripts/` | Dir | 200 KB | Offline auxiliary scripts (benchmarking, atlas generation, video synthesis, DOCX/PPTX generators). |
| `src/` | Dir | 344 KB | Core project library modules (`methods.py`, `features.py`, `labels.py`, `dataset.py`, `train.py`, etc.). |
| `tests/` | Dir | 64 KB | Unit test suite (22 tests across 4 modules). |
| `ORB/Classical Vision.pt` | File | 6.5 MB | Pretrained PyTorch weights for ORB/Classical Vision-nano (downstream utility evaluation only). |

---

## 2. Python Packages & Primary Entry Points

### A. Environment Dependencies (`requirements.txt`)
- `numpy`
- `opencv-python` (or `opencv-python-headless`)
- `pandas`
- `scikit-learn`
- `xgboost`
- `joblib`
- `matplotlib`
- `pyyaml`
- `onnxruntime`

### B. Existing Entry Points
- `main.py`: End-to-end benchmark execution on LOL-v1 (extract, CV, figures, ablations, eval15).
- `src/demo.py`: Single-image inference CLI (`--image`, `--reference`, `--output`, `--model_path`).
- `app.py`: Standalone web application on `http://localhost:8501`.
- `scripts/benchmark_lol_v2.py`: Zero-shot evaluation on LOL-v2 Real & Synthetic.
- `scripts/benchmark_8enhancers_10features.py`: 8-enhancer / 10-descriptor comparison benchmark.
- `scripts/train_test_lol_v2.py`: In-distribution LOL-v2 model training and benchmarking.
- `scripts/generate_presentation.py`: 16:9 PowerPoint slide generator (`python-pptx`).
- `scripts/generate_walkthrough_docx.py`: DOCX summary generator (`python-docx`).
- `scripts/generate_video_demo.py`: Video synthesis script demonstrating anti-flicker temporal smoothing.

---

## 3. Notebooks, Results & Caches

### A. Notebooks
- `notebooks/exploration_and_benchmarks.ipynb`: Exploratory data analysis, score distributions, and interactive plotting.

### B. Cached Data & Results (`results/`)
- `results/scores.csv`: 485 rows x PSNR/SSIM for all candidate methods on LOL-v1 our485.
- `results/features.csv`: 485 rows x 8 base features + 2 interaction features on LOL-v1 our485.
- `results/scene_splits.csv` & `official_scene_splits.csv`: Scene grouping assignments.
- `results/baseline_regret_table.csv`: Regret summaries for baselines and primary classifiers.
- `results/metrics_table.csv`: GroupKFold summary metrics.
- `results/scores_lol_v2_real.csv` & `scores_lol_v2_synthetic.csv`: LOL-v2 score caches.
- `results/features_lol_v2_real.csv` & `features_lol_v2_synthetic.csv`: LOL-v2 feature caches.
- `results/models/`: Serialized models (`logistic_regression.pkl`, `random_forest.pkl`, `hist_gradient_boosting.pkl`).
- `results/figures/`: Publication figures (`fig1` through `fig11`).

---

## 4. Paper & Submission Folders
- `report/IEEE_Paper_Image_Enhancement_Selection.tex`: Primary manuscript (IEEE Transactions on Image Processing format, 958 lines).
- `report/IEEE_Paper_Image_Enhancement_Selection.pdf`: Compiled camera-ready PDF.
- `report/IEEE_Paper_Anonymized.pdf`: Anonymized review PDF.
- `report/Project_Triathlon_Executive_Presentation.pptx`: 16:9 presentation slide deck.
- `report/Project_Triathlon_Supplementary_Visual_Atlas.pdf`: Visual comparison atlas.
- `JOURNAL_SUBMISSION/`: Mirror package for journal submission (contains exact duplicates of `report/` PDFs and zips).

---

## 5. Extensions & Secondary Modules
- **Web Dashboard**: `app.py` (lightweight standalone web server).
- **Temporal Video Smoothing**: `src/video_pipeline.py` & `scripts/generate_video_demo.py` (exponential moving average temporal filtering to prevent selection flickering across video streams).
- **Downstream Object Detection**: `src/downstream_matching.py`, `src/downstream_detection.py`, and `ORB/Classical Vision.pt` (ORB/Classical Vision downstream detection utility comparison).
- **Downstream Feature Matching**: `src/downstream_matching.py` (ORB/AKAZE keypoint and descriptor inlier matching).
- **Edge Latency Profiling**: `src/edge_hardware_profile.py` & `src/latency_aware_selection.py` (ONNX export and CPU execution budget profiling).
- **Predict-Then-Optimize (PTO)**: `src/predict_then_optimize.py` (Ridge regression predicting continuous per-method PSNR).
- **Multi-Degradation Cascading**: `src/multi_degradation.py` (Phase 2 hierarchical regime diagnosis).

---

## 6. Test Suite
- `tests/test_pipeline.py`: Tests enhancement method shapes, bounds, feature extraction, and end-to-end model inference.
- `tests/test_deferred_milestones.py`: Tests PTO regressors, ORB matching, and latency-aware selection.
- `tests/test_lol_v2.py`: Tests LOL-v2 pair loading, manifest creation, and score scoring.
- `tests/test_video_pipeline.py`: Tests temporal anti-flicker queue, exponential smoothing, and video inference.
- **Execution Status**: All 22 tests passing (`Ran 22 tests in 19.999s OK`).

---

## 7. Duplicate & Redundant Files Analysis

### A. Exact Duplicates by SHA-256 Hash
The following 5 pairs of files are bitwise identical:

| SHA-256 Prefix | Primary File (`report/`) | Redundant Duplicate (`JOURNAL_SUBMISSION/`) | Size | Action |
| :--- | :--- | :--- | :---: | :--- |
| `171f72ab7e36` | `report/IEEE_Transactions_Anonymized_Submission_Package.zip` | `JOURNAL_SUBMISSION/IEEE_LaTeX_Source_Package.zip` | 24 MB | Archive `JOURNAL_SUBMISSION/` duplicate |
| `8678701dc3bd` | `report/IEEE_Paper_Image_Enhancement_Selection.pdf` | `JOURNAL_SUBMISSION/IEEE_Paper_CameraReady_With_Authors.pdf` | 6.5 MB | Archive `JOURNAL_SUBMISSION/` duplicate |
| `df0e374058a7` | `report/Project_Triathlon_Supplementary_Visual_Atlas.pdf` | `JOURNAL_SUBMISSION/Supplementary_Visual_Atlas.pdf` | 11 MB | Archive `JOURNAL_SUBMISSION/` duplicate |
| `57f934be268c` | `report/IEEE_Paper_Anonymized.pdf` | `JOURNAL_SUBMISSION/IEEE_Paper_To_Submit_Anonymized.pdf` | 6.5 MB | Archive `JOURNAL_SUBMISSION/` duplicate |
| `c0c9c61a49ca` | `report/Project_Triathlon_Basic_vs_Extra_Comparison.pdf` | `JOURNAL_SUBMISSION/Project_Triathlon_Basic_vs_Extra_Comparison.pdf` | 42 KB | Archive `JOURNAL_SUBMISSION/` duplicate |

### B. Near-Duplicate Code Audit (difflib SequenceMatcher >= 0.90)
- Automated pairwise comparison across all 41 Python files identified **0 near-duplicate source modules**. All Python modules have distinct functions.

### C. Large Binary Artifacts to Archive
- `ORB/Classical Vision.pt` (6.5 MB): Pretrained PyTorch weights, outside classical CPU-only selector core $\rightarrow$ Archive to `_archive/ORB/Classical Vision.pt`.
- `JOURNAL_SUBMISSION/` (48 MB): Entire duplicate bundle directory $\rightarrow$ Archive to `_archive/JOURNAL_SUBMISSION/`.
- `results/video_demo_anti_flicker.mp4` (2.5 MB): Generated demo video $\rightarrow$ Archive to `_archive/results/video_demo_anti_flicker.mp4`.
