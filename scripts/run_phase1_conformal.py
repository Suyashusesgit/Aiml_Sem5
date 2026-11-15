"""
Phase 1 Module 1: Conformal Prediction & Risk-Controlled Abstention
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Performs:
1. Scene-grouped cross-conformal prediction (SplitConformalClassifier) across alpha in [0.05, 0.10, 0.15, 0.20].
2. Verification of finite-sample coverage guarantees and set efficiency.
3. Confidence-gated abstention sweep (tau in [0.0, 0.70]) with safe fallback to Retinex.
4. Generates results/phase1/conformal_calibration.csv, results/phase1/abstention_curve.csv,
   and results/figures/fig12_conformal_abstention.png.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import CANONICAL_METHODS
from src.features import FEATURE_NAMES, validate_feature_dataframe
from src.selectors import make_selector
from src.conformal import SplitConformalClassifier, evaluate_confidence_abstention
from src.train import make_folds


def align_probabilities(probs: np.ndarray, model_classes: np.ndarray, target_classes: List[str]) -> np.ndarray:
    aligned = np.zeros((probs.shape[0], len(target_classes)), dtype=np.float32)
    cls_to_target_idx = {c: i for i, c in enumerate(target_classes)}
    for src_idx, c in enumerate(model_classes):
        if c in cls_to_target_idx:
            aligned[:, cls_to_target_idx[c]] = probs[:, src_idx]
    return aligned


def run_conformal_benchmark():
    print("=" * 70)
    print("PHASE 1 · MODULE 1: CONFORMAL PREDICTION & RISK-CONTROLLED ABSTENTION")
    print("=" * 70)

    # 1. Load LOL-v1 data
    feat_path = "results/features_lol_v1_our485.csv"
    score_path = "results/scores_lol_v1_our485.csv"
    if not os.path.exists(feat_path) or not os.path.exists(score_path):
        raise FileNotFoundError("LOL-v1 features or scores CSV not found.")

    df_feat = pd.read_csv(feat_path)
    df_score = pd.read_csv(score_path)
    validate_feature_dataframe(df_feat, FEATURE_NAMES)

    # Merge on filename
    df = pd.merge(df_feat, df_score, on="filename")
    X = df[FEATURE_NAMES].values
    y = df["label"].values
    target_classes = list(CANONICAL_METHODS)

    # Determine scene groups using scene_splits_full.csv
    splits_path = "results/scene_splits_full.csv"
    if os.path.exists(splits_path):
        df_splits = pd.read_csv(splits_path)
        # map filename to scene_id
        fn_to_group = dict(zip(df_splits["filename"], df_splits["scene_id"]))
        groups = np.array([fn_to_group.get(fn, idx) for idx, fn in enumerate(df["filename"])])
    else:
        # Fallback to group by index
        groups = np.arange(len(df))

    unique_groups = np.unique(groups)
    print(f"Total samples: {len(df)} across {len(unique_groups)} distinct scene clusters.")

    # 2. 5-Fold Scene-Grouped Cross-Conformal Evaluation
    folds = make_folds(X, y, groups, n_splits=5, seed=42)

    alpha_levels = [0.05, 0.10, 0.15, 0.20]
    results_by_alpha = {a: {"target": 1.0 - a, "coverages": [], "set_sizes": [], "singletons": []} for a in alpha_levels}

    # Also collect out-of-fold probability predictions for the entire dataset
    oof_probs = np.zeros((len(df), len(target_classes)), dtype=np.float32)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Nested calibration split within training fold: 75% proper train, 25% calibration
        unique_tr_groups = np.unique(groups_train)
        rng = np.random.RandomState(42 + fold_idx)
        shuffled_g = rng.permutation(unique_tr_groups)
        n_prop = int(len(unique_tr_groups) * 0.75)
        prop_group_set = set(shuffled_g[:n_prop])

        prop_mask = np.array([g in prop_group_set for g in groups_train])
        cal_mask = ~prop_mask

        X_prop, y_prop = X_train[prop_mask], y_train[prop_mask]
        X_cal, y_cal = X_train[cal_mask], y_train[cal_mask]

        # Train multinomial logistic regression
        selector = make_selector("logreg", random_state=42)
        selector.fit(X_prop, y_prop)

        # Full-fold model for OOF probabilities
        full_fold_selector = make_selector("logreg", random_state=42)
        full_fold_selector.fit(X_train, y_train)
        raw_oof = full_fold_selector.predict_proba(X_test)
        oof_probs[test_idx] = align_probabilities(raw_oof, full_fold_selector.classes_, target_classes)

        # Conformal evaluation for each alpha
        for a in alpha_levels:
            cp = SplitConformalClassifier(selector, alpha=a)
            cp.calibrate(X_cal, y_cal)
            eval_dict = cp.evaluate_coverage(X_test, y_test)
            pred_sets = cp.predict_set(X_test)

            singleton_ratio = np.mean([1 if len(s) == 1 else 0 for s in pred_sets])
            results_by_alpha[a]["coverages"].append(eval_dict["empirical_coverage"])
            results_by_alpha[a]["set_sizes"].append(eval_dict["average_set_size"])
            results_by_alpha[a]["singletons"].append(singleton_ratio)

    # Summarize conformal calibration
    calib_records = []
    print("\n--- Conformal Prediction Calibration Results ---")
    print(f"{'Target Coverage':<18} | {'Empirical Coverage':<20} | {'Avg Set Size':<15} | {'Singleton Ratio':<15}")
    print("-" * 75)
    for a in alpha_levels:
        target_pct = (1.0 - a) * 100.0
        emp_mean = float(np.mean(results_by_alpha[a]["coverages"])) * 100.0
        emp_std = float(np.std(results_by_alpha[a]["coverages"])) * 100.0
        size_mean = float(np.mean(results_by_alpha[a]["set_sizes"]))
        size_std = float(np.std(results_by_alpha[a]["set_sizes"]))
        sing_mean = float(np.mean(results_by_alpha[a]["singletons"])) * 100.0

        print(f"{target_pct:>5.1f}%             | {emp_mean:>5.1f}% (+/- {emp_std:>4.1f}%)    | {size_mean:>4.2f} (+/- {size_std:>4.2f}) | {sing_mean:>5.1f}%")

        calib_records.append({
            "alpha": a,
            "target_coverage_pct": round(target_pct, 1),
            "empirical_coverage_pct": round(emp_mean, 2),
            "empirical_coverage_std": round(emp_std, 2),
            "average_set_size": round(size_mean, 3),
            "average_set_size_std": round(size_std, 3),
            "singleton_set_pct": round(sing_mean, 2),
        })

    df_calib = pd.DataFrame(calib_records)
    df_calib.to_csv("results/phase1/conformal_calibration.csv", index=False)
    print("Saved: results/phase1/conformal_calibration.csv")

    # 3. Confidence-Gated Abstention Sweep on Out-of-Fold Predictions
    # Create a mock wrapper that implements predict_proba returning oof_probs
    class OOFModelWrapper:
        def __init__(self, probs, classes):
            self.probs = probs
            self.classes_ = np.array(classes)
        def predict_proba(self, X):
            return self.probs

    oof_wrapper = OOFModelWrapper(oof_probs, target_classes)
    tau_thresholds = [0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    df_abstain = evaluate_confidence_abstention(
        model=oof_wrapper,
        X=X,
        y_oracle=y,
        df_scores=df,
        thresholds=tau_thresholds,
        default_fallback="retinex"
    )

    print("\n--- Confidence-Gated Abstention Sweep (Fallback: Retinex) ---")
    print(df_abstain.to_string(index=False))
    df_abstain.to_csv("results/phase1/abstention_curve.csv", index=False)
    print("Saved: results/phase1/abstention_curve.csv")

    # 4. Generate Figure 12: Conformal Coverage & Risk-Controlled Abstention
    os.makedirs("results/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Coverage vs Target & Set Size
    targets = [r["target_coverage_pct"] for r in calib_records]
    empirical = [r["empirical_coverage_pct"] for r in calib_records]
    sizes = [r["average_set_size"] for r in calib_records]

    ax1.plot([75, 100], [75, 100], "k--", alpha=0.6, label="Ideal Calibration (y=x)")
    ax1.plot(targets, empirical, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Empirical Coverage")
    ax1.set_xlabel("Target Coverage Guaranteed: $1 - \\alpha$ (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Coverage Rate (%)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlim(78, 97)
    ax1.set_ylim(78, 100)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(targets, sizes, "s-", color="#ff7f0e", linewidth=2.0, markersize=7, label="Avg Set Size |C(x)|")
    ax1_twin.set_ylabel("Average Prediction Set Size", fontsize=11, fontweight="bold", color="#ff7f0e")
    ax1_twin.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1_twin.set_ylim(1.0, 4.0)

    # Combine legends for ax1
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right", framealpha=0.9)
    ax1.set_title("(a) Split-Conformal Coverage & Set Efficiency", fontsize=12, fontweight="bold")

    # Panel B: Abstention vs Effective Regret
    taus = df_abstain["confidence_threshold_tau"].values
    regrets = df_abstain["overall_effective_regret_db"].values
    abstain_rates = df_abstain["abstain_rate_pct"].values

    color_reg = "#2ca02c"
    color_abs = "#d62728"

    ax2.plot(taus, regrets, "o-", color=color_reg, linewidth=2.5, markersize=8, label="Effective Regret (dB)")
    ax2.axhline(2.731, color="gray", linestyle="--", alpha=0.7, label="Static Retinex Baseline (2.731 dB)")
    ax2.set_xlabel("Confidence Abstention Threshold $\\tau$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Overall Regret vs Oracle (dB)", fontsize=11, fontweight="bold", color=color_reg)
    ax2.tick_params(axis="y", labelcolor=color_reg)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(taus, abstain_rates, "^--", color=color_abs, linewidth=2.0, markersize=7, label="Abstain Rate (%)")
    ax2_twin.set_ylabel("Abstention Rate to Safe Default (%)", fontsize=11, fontweight="bold", color=color_abs)
    ax2_twin.tick_params(axis="y", labelcolor=color_abs)
    ax2_twin.set_ylim(-2, 102)

    lines_3, labels_3 = ax2.get_legend_handles_labels()
    lines_4, labels_4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc="center left", framealpha=0.9)
    ax2.set_title("(b) Risk-Controlled Abstention Frontier", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig_path = "results/figures/fig12_conformal_abstention.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated Figure 12: {fig_path}")

    return df_calib, df_abstain


if __name__ == "__main__":
    run_conformal_benchmark()
