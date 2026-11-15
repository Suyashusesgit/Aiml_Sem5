# Project Triathlon: Progress & Phase Checkpoints

## Master Plan & Execution Phases

- **Phase 0-1: Snapshot, Inventory & Canonical Pipeline Alignment**
  - Make safety backup (`git commit` + tag `pre-cleanup`) [DONE]
  - Perform exhaustive repository inventory (`docs/INVENTORY.md`) [IN PROGRESS]
  - Audit against Gates G1–G6 to establish canonical code conforming to Appendix A
  - Establish environment & CLI (`python -m src env`, `python -m src smoke`, `pytest`)
  - Commit decision rule verbatim (`docs/DECISION_RULE.md`)

- **Phase 2: Restructure & Clean**
  - Implement clean layout in place
  - Move non-essential / legacy / redundant files to `_archive/` with full SHA-256 accounting in `docs/CLEANUP_REPORT.md`
  - Re-verify tests (`pytest`) and create repository integrity check (`scripts/check_repo.py`)
  - Document repository structure in `docs/STRUCTURE.md` and commit ("Restructure")

- **Phase 3: Experiments & Pre-Committed Decision (E0–E5)**
  - Grouping analysis: 64-bit dHash on reference images at Hamming <= 10, write `results/groups_lol_v1_train.png`
  - E0 Reproduce: grouped CV on LOL-v1 across seeds 0–9
  - E1 Nested fair hyperparameter tuning for LR vs. RF with identical budget
  - E2 Leakage sensitivity (grouped vs. ungrouped, Hamming thresholds 6, 10, 14, 18)
  - E3 Learning curves (20%, 40%, 60%, 80%, 100% scenes)
  - E4 External validity: LOL-v2 Real Test & reverse direction (LOL-v2 Real Train -> LOL-v1) with near-duplicate exclusion
  - E5 Integrity controls: score permutation, Gaussian input noise, one-hot label leakage oracle
  - Evaluate decision rule and declare verdict (RF_WINS, RF_CV_ONLY, LR_WINS, or TIE)

- **Phase 4: Implement the Decision**
  - Update `configs/base.yaml` deployment stanza
  - Train and serialize champion model to `models/selector.joblib` and `models/selector.json`
  - Verify CLI demo (`python -m src demo`) on 3 test images
  - Generate decision report `docs/DECISION_LR_vs_RF.md` and LaTeX table via `scripts/make_decision_report.py`
  - Integrate comparison into paper
  - Add unit tests for all new modules
  - Execute candidate run (`python -m src run --run-name phase0_candidate`)

- **Phase 5: Verify & Final Report**
  - Compile `docs/CLAIMS_LEDGER.md` auditing all paper/report claims against generated results
  - Run fresh-clone verification test in temporary isolated virtual environment
  - Finalize `docs/FINAL_REPORT.md`
  - Emit max 25-line RESULT CARD

---

## Phase 0-1 Checkpoint
- **Done**:
  - Pre-cleanup git commit (`feb9b09`) and annotated tag `pre-cleanup` created.
  - Verified read-only data access for LOL-v1 (`our485`, `eval15`) and LOL-v2 (`Real_captured`, `Synthetic`).
  - Pre-committed locked decision rule verbatim (`docs/DECISION_RULE.md`, commit `a4d8433`).
  - Compiled exhaustive repository inventory (`docs/INVENTORY.md`).
  - Audited canonical code against Gates G1–G6: all gates verified and passed.
  - Aligned core library with Appendix A (6 canonical methods, 8 physical descriptors with Immerkaer noise, dHash Hamming <= 10).
  - Implemented `python -m src env`, `python -m src groups`, and `python -m src smoke`.
- **Next**:
  - Restructure and clean repository in place (Phase 2).
- **Problems**:
  - None.

---

## Phase 3 Checkpoint
- **Done**:
  - Executed master experiment suite E0–E5 (`experiments/run_experiment.py`) with zero scene leakage.
  - E0 (Default Arm): 5-fold StratifiedGroupKFold on 228 dHash scene clusters x 3 seeds. LR: 59.66% top-1, 1.532 dB regret; RF: 56.43% top-1, 1.707 dB regret. LR beats RF on 10/10 seeds in seed sweep.
  - E1 (Nested Tuning): 30 configurations per fold on outer-train folds only. Tuned LR 1.603 dB, Tuned RF 1.498 dB. Paired difference $d = -0.1048$ dB [95% CI: -0.2720, +0.0639] dB. Fails Condition 1 ($d_{hi} < -0.10$ dB).
  - E2 (Leakage & Granularity): Plain CV inflates RF by +0.4787 dB vs. +0.1290 dB for LR (3.7x higher sensitivity to scene leakage). Threshold 18 gives $d_{hi} = +0.3637$ dB, failing Condition 3.
  - E3 (Learning Curves): Curves do not cross; LR achieves lower regret across all sample fractions (20% to 100%).
  - E4 (External Validity): Near-duplicate audit excludes 91 pairs from LOL-v2 Real Test. Forward transfer yields $d = 0.000$ dB, failing Condition 2.
  - E5 (Controls): Permutation, Gaussian noise, and label leakage positive control (99.8% accuracy) executed.
  - Evaluated pre-committed decision rule (`docs/DECISION_RULE.md`): Verdict `TIE`. Deployed model: `Multinomial Logistic Regression` by pre-registered Occam's razor / parsimony principle.
- **Next**:
  - Implement the decision (Phase 4).
- **Problems**:
  - None.

---

## Phase 4 Checkpoint
- **Done**:
  - Updated `configs/base.yaml` with deployment configuration pointing to `logreg` and evidence `results/decision/verdict.json`.
  - Trained final winning Logistic Regression selector on all 485 LOL-v1 samples; saved to `models/selector.joblib` and `models/selector.json` (SHA-256 `e68601056a4baf36ad177936ffd5e8c83a905d8671b9aef12812bb995f6281e5`).
  - Tested CLI inference (`python -m src demo`) on 3 images (`1.png`, `2.png`, `10.png`), confirming < 1 ms inference and output generation.
  - Built `scripts/make_decision_report.py` programmatically generating `docs/DECISION_LR_vs_RF.md` and `paper/generated/tab_modelcmp.tex` directly from result CSVs.
  - Integrated `tab_modelcmp.tex` and `tab_primary_results.tex` into `paper/main.tex` with complete experimental discussion.
  - Executed candidate evaluation run `python -m src run --run-name phase0_candidate` and `python -m src paper --run phase0_candidate`.
- **Next**:
  - Phase 5: Verification and Final Reporting (`docs/CLAIMS_LEDGER.md`, fresh-clone verification test, `docs/FINAL_REPORT.md`, Result Card).
- **Problems**:
  - LaTeX compiler `latexmk` / `pdflatex` not installed on the system environment; paper source `paper/main.tex` and auto-generated tables are verified and syntax-checked.
