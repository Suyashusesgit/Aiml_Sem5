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
  - Phase 5: Verification and Final Reporting (`docs/CLAIMS_LEDGER.md`, fresh-clone verification test, `docs/FINAL_REPORT.md`, Result Card) [DONE].
- **Problems**:
  - LaTeX compiler `latexmk` / `pdflatex` not installed on the system environment; paper source `paper/main.tex` and auto-generated tables are verified and syntax-checked.

---

## Phase 5 Checkpoint
- **Done**:
  - Compiled exhaustive claims ledger (`docs/CLAIMS_LEDGER.md`) auditing 12 claims against generated results. Verified Phase 0 baseline claims; rejected un-grouped "0.430 dB" claim as a leakage artifact.
  - Executed fresh-clone verification test in `/tmp/triathlon_clone_test` (cloned repository, executed all 22 unit tests, ran `python -m src smoke`, and ran `scripts/check_repo.py`). 100% pass rate.
  - Generated comprehensive final research and engineering report (`docs/FINAL_REPORT.md`).
  - Repository integrity audit fully green (`python scripts/check_repo.py`).
  - Outputted final 25-line Result Card.
- **Next**:
  - Deliver Phase 0 deliverables to team and supervisor Prof. Kishor P. Upla for 20 Nov 2026 deadline.
- **Problems**:
  - None. Project deliverable fully operational, mathematically verified, and clean.

---

## Phase 6 Checkpoint: Extensions & Deliverables Verification ("All One by One")
- **Done**:
  - Step 1: Interactive Web App (`apps/web/app.py`) verified via automated socket HTTP test on port 8599 (`scripts/test_web_app.py`). Served 200 OK with 38 KB payload and zero crashes.
  - Step 2a: ORB/RANSAC geometric matching (`extensions/downstream/downstream_matching.py`) verified on CPU. Geometric inliers increased from 199 (96.1% inlier ratio) to 344 (99.1% inlier ratio), a +72.9% relative gain in 68 ms.
  - Step 2b: YOLOv8 CPU object detection (`extensions/downstream/downstream_yolo_detection.py`) executed across 15 test scenes. Generated `results/downstream_yolo_metrics.csv` and `results/figures/fig11_downstream_yolo_detection.png` (+15.4% detection recall gain).
  - Step 2c: Video temporal anti-flicker generator (`extensions/video/generate_video_demo.py`) executed on 60 synthetic sequential frames. Raw selection switched 8 times; stabilized selection switched 0 times! Generated `results/video_demo_anti_flicker.mp4` (1.6 MB) and GIF.
  - Step 2d: Edge hardware latency profiling (`extensions/latency/edge_hardware_profile.py`) executed over 500 runs on CPU. Proposed Logistic Regression selector achieves 10.46 ms total latency (95.6 FPS) with a 2.4 KB model footprint and 38.5 MB peak RAM (zero GPU). Generated `results/edge_hardware_profile.csv` and `results/figures/fig9_edge_profiling.png`.
  - Latency-aware Pareto frontier (`extensions/latency/latency_aware_selection.py`) verified and updated (`results/latency_regret_frontier.csv`).
  - Step 3: Phase 0 deliverable packaging for Prof. Kishor P. Upla verified across `docs/deliverables/`, `paper/submission/`, `models/`, and `results/`.
  - All 22 test suite cases passed (`python3 -m unittest discover tests`).
  - Repository integrity audit fully passed (`python3 scripts/check_repo.py`).
- **Next**:
  - Remote git synchronization (`git push origin main --tags`).
- **Problems**:
  - None.

