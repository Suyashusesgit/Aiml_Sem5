# Repository Structure & Layout

*Project: Project Triathlon*  
*Standard: Section 5 Production Architecture with Protected Extensions*  

```
.
├── .gitignore                          # Git untracked specifications
├── configs/                            # Configuration YAML files (base.yaml)
├── data/                               # [READ-ONLY] Dataset binaries (LOL-v1, LOL-v2)
├── docs/                               # Governance, SOPs, audits, and experimental logs
│   ├── deliverables/                   # Master SOPs, student completion reports, executive summaries
│   ├── CLAIMS_LEDGER.md                # Numeric claims audit against empirical evidence
│   ├── CLEANUP_REPORT.md               # Archived files manifest & SHA-256 accounting
│   ├── DECISIONS.md                    # Technical decisions and protected assets mapping
│   ├── DECISION_RULE.md                # Pre-committed locked decision protocol
│   ├── FINAL_REPORT.md                 # Executive research and engineering report (Phase 0)
│   ├── FUTURE_PHASES_ROADMAP.md        # Engineering roadmap for Phase 2 (Sem 7) & Phase 3 (Sem 8)
│   ├── PHASE1_REPORT.md                # Technical research report for Phase 1 (Sem 6)
│   ├── PROGRESS.md                     # Phase-by-phase execution progress log
│   └── STRUCTURE.md                    # Repository architecture map
├── apps/                               # User-facing production applications
│   └── web/                            # Interactive browser-based enhancement web app
│       ├── app.py                      # Lightweight http.server application
│       └── README.md                   # Web application instructions (CPU)
├── extensions/                         # Optional domain extensions (isolated from core)
│   ├── requirements-extensions.txt     # Heavy dependencies (ultralytics, torch, streamlit)
│   ├── downstream/                     # Downstream machine vision (YOLOv8 & ORB/RANSAC)
│   │   ├── yolov8n.pt                  # Pretrained YOLOv8 Nano weights (CPU)
│   │   ├── downstream_yolo_detection.py# YOLOv8 object detection benchmark
│   │   ├── downstream_matching.py      # ORB feature matching with RANSAC
│   │   ├── downstream_detection.py     # Keypoint recovery benchmark
│   │   └── README.md                   # Downstream instructions (CPU)
│   ├── video/                          # Video temporal feature smoothing
│   │   ├── video_pipeline.py           # EMA smoothing & Markovian hysteresis gate
│   │   ├── generate_video_demo.py      # Video benchmark generator
│   │   ├── video_demo_anti_flicker.mp4 # Rendered demonstration video
│   │   └── README.md                   # Video pipeline instructions (CPU)
│   └── latency/                        # Latency benchmarks & edge profiling
│       ├── edge_hardware_profile.py    # 500-run hardware profiling
│       ├── latency_aware_selection.py  # Latency-constrained Pareto frontier
│       ├── edge_hardware_profile.csv   # CPU latency measurements
│       ├── latency_regret_frontier.csv # Regret-latency trade-off curve
│       └── README.md                   # Latency instructions (CPU)
├── experiments/                        # Decision experiments driver scripts (E0–E5)
│   └── run_experiment.py               # Master E0–E5 experiment runner & decision evaluator
├── models/                             # Serialized production model artifacts
│   ├── selector.joblib                 # Trained winning Multinomial Logistic Regression model
│   └── selector.json                   # Model metadata, SHA-256, and evidence provenance
├── notebooks/                          # Exploratory analysis and visual verification notebooks
├── paper/                              # Publication LaTeX manuscript, assets, and figures
│   ├── main.tex                        # Master IEEE Transactions LaTeX document
│   ├── IEEEtran.cls                    # IEEE Transactions document class
│   ├── generated/                      # Auto-generated LaTeX tables from empirical CSVs
│   │   ├── tab_primary_results.tex     # Primary benchmark table (E0)
│   │   └── tab_modelcmp.tex            # Empirical comparison table (E0–E5)
│   └── submission/                     # Complete camera-ready & anonymized submission bundle
│       ├── IEEE_Paper_To_Submit_Anonymized.pdf
│       ├── Supplementary_Visual_Atlas.pdf
│       ├── IEEE_Paper_CameraReady_With_Authors.pdf
│       ├── Project_Triathlon_Basic_vs_Extra_Comparison.pdf
│       ├── IEEE_LaTeX_Source_Package.zip
│       └── README_SUBMISSION_GUIDE.md
├── pytest.ini                          # Pytest testing configuration
├── requirements-tested.txt             # Pinned runtime dependencies
├── requirements.txt                    # Base runtime dependencies
├── results/                            # [GENERATED] Experiment outputs, CSVs, and plots
│   ├── decision/                       # Outputs for E0–E5 and verdict.json
│   ├── figures/                        # Publication figures (Figures 1–16)
│   ├── logs/                           # Background execution logs
│   └── phase1/                         # Phase 1 experimental outputs & benchmarks
├── scripts/                            # Maintenance, repo check, and reporting utilities
│   ├── check_repo.py                   # Automated repository health checker
│   ├── make_decision_report.py         # Empirical report and table generator
│   ├── run_phase1.py                   # Master Phase 1 experimental runner
│   ├── run_phase1_conformal.py         # Split-conformal calibration runner
│   ├── run_phase1_cross_degradation.py # Zero-shot cross-dataset runner
│   ├── run_phase1_feature_expansion.py # 10D physical features runner
│   ├── run_phase1_leave_one_out.py     # Leave-One-Enhancer-Out runner
│   ├── run_phase1_rq2_oracle_study.py  # RQ2 dual-metric runner
│   └── train_deploy_model.py           # Model training and serialization script
├── src/                                # Canonical library package (methods, features, etc.)
│   ├── __init__.py
│   ├── cli.py                          # Unified CLI entry point
│   ├── conformal.py                    # Split-conformal predictor & risk-controlled abstention
│   ├── dataset.py                      # Dataset loader, dHash grouping, resizing
│   ├── demo.py                         # Single-image inference engine
│   ├── evaluate.py                     # Cluster bootstrap resampler & metrics
│   ├── features.py                     # Handcrafted physical degradation descriptors
│   ├── labels.py                       # PSNR/SSIM scoring & oracle tie-breaker
│   ├── methods.py                      # 6 canonical enhancement methods
│   ├── paper_assets.py                 # Automated LaTeX asset generator
│   ├── pipeline.py                     # Dataset preparation orchestrator
│   ├── plots.py                        # Visualization generation
│   ├── selectors.py                    # Classical ML selectors (LR, RF, PTO, Baselines)
│   └── train.py                        # 5-fold StratifiedGroupKFold fold generator
├── tests/                              # Automated unit test suite (28 tests)
└── _archive/                           # [GIT-IGNORED] Archived exact duplicates (Rule 4)
```

---

## Directory Descriptions
- `apps/`: User-facing interactive software interfaces, including the standalone web application (`apps/web/`).
- `configs/`: YAML configurations defining dataset paths, model parameters, and thresholds.
- `data/`: Read-only storage for LOL-v1 (`our485`, `eval15`) and LOL-v2 (`Real_captured`, `Synthetic`) image datasets.
- `docs/`: Comprehensive project documentation, protocols, inventories, and decision records.
- `extensions/`: Modular, non-core capabilities (YOLOv8 object detection, ORB geometric matching, video temporal smoothing, edge latency profiling) isolated with their own READMEs and heavy dependency file (`extensions/requirements-extensions.txt`).
- `experiments/`: Master runner (`run_experiment.py`) executing pre-registered decision experiments E0–E5.
- `models/`: Trained champion model (`selector.joblib`) and metadata manifest (`selector.json`).
- `paper/`: IEEE Transactions LaTeX source (`main.tex`), class files (`IEEEtran.cls`), generated tables, and journal submission bundle (`paper/submission/`).
- `results/`: Output directories for generated experiment metrics, decision logs, and publication plots.
- `scripts/`: Repository health verification (`check_repo.py`), decision report generator (`make_decision_report.py`), and model deployment (`train_deploy_model.py`).
- `src/`: Core importable Python library implementing methods, features, selectors, training, and evaluation. Completely free of heavy dependencies (zero `torch` or `ultralytics`).
- `tests/`: Automated unit test suite verifying pipeline components, bounds, and absence of leakage.
- `_archive/`: Preserved exact duplicates and superseded exploratory prototype scripts with cryptographic SHA-256 accounting.
