# Repository Structure & Layout

*Project: Project Triathlon*  
*Standard: Section 5 Production Architecture*  

```
.
├── .gitignore                     # Git untracked specifications
├── configs/                       # Configuration YAML files (base.yaml)
├── data/                          # [READ-ONLY] Dataset binaries (LOL-v1, LOL-v2)
├── docs/                          # Governance, SOPs, audits, and experimental logs
├── experiments/                   # Decision experiments thin driver scripts (E0–E5)
├── models/                        # Serialized production model artifacts
├── notebooks/                     # Exploratory analysis and visual verification notebooks
├── paper/                         # Publication LaTeX manuscript, assets, and figures
├── pytest.ini                     # Pytest testing configuration
├── requirements-tested.txt        # Pinned runtime dependencies
├── requirements.txt               # Base runtime dependencies
├── results/                       # [GENERATED] Experiment outputs, CSVs, and plots
├── scripts/                       # Maintenance, repo check, and reporting utilities
├── src/                           # Canonical library package (methods, features, etc.)
├── tests/                         # Automated unit test suite
└── _archive/                      # [GIT-IGNORED] Archived non-canonical artifacts (Rule 4)
```

## Directory Descriptions
- `configs/`: Experiment configuration files defining dataset paths, model parameters, and thresholds.
- `data/`: Read-only storage for LOL-v1 (`our485`, `eval15`) and LOL-v2 (`Real_captured`, `Synthetic`) image datasets.
- `docs/`: Comprehensive project documentation, protocols, inventories, and decision records.
- `experiments/`: Thin Python scripts executing pre-registered decision experiments E0–E5.
- `models/`: Trained selector joblib binaries and configuration manifests.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visual inspections.
- `paper/`: IEEE Transactions LaTeX source (`main.tex`), class files (`IEEEtran.cls`), and generated tables.
- `results/`: Output directories for generated experiment metrics, decision logs, and publication plots.
- `scripts/`: Repository health verification (`check_repo.py`) and decision report generator (`make_decision_report.py`).
- `src/`: Core importable Python library implementing methods, features, selectors, training, and evaluation.
- `tests/`: Automated test suite verifying pipeline components, bounds, and absence of leakage.
- `_archive/`: Preserved legacy files, duplicate packages, and external binary artifacts with SHA-256 accounting.
