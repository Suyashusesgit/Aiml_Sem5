"""
Phase 1 Module 5: 10D Physical Feature Expansion Benchmark
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Compares 8D handcrafted baseline against expanded 10D physical feature set:
Adds:
- air_light_transmission (He et al. 2009 atmospheric transmission estimate)
- dynamic_range_ratio (P99 / (P1 + eps))

Evaluated under strict 5-fold StratifiedGroupKFold cross-validation across seeds [0, 1, 2].
Measures:
- Top-1 Accuracy (%)
- Mean Regret vs Oracle (dB)
- Paired delta d = Regret_10D - Regret_8D (with 95% bootstrap CI)
- Normalized feature weights / coefficient magnitudes
Generates:
- results/features_lol_v1_10d.csv
- results/phase1/feature_expansion_comparison.csv
- results/phase1/feature_10d_coefficients.csv
- results/figures/fig13_feature_expansion.png
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    FEATURE_NAMES,
    PHYSICAL_10_DESCRIPTORS,
    extract_features,
    validate_feature_dataframe
)
from src.selectors import make_selector
from src.train import make_folds
from src.evaluate import GroupResampler, paired_ci


def run_feature_expansion():
    print("=" * 70)
    print("PHASE 1 · MODULE 5: 10D PHYSICAL FEATURE EXPANSION BENCHMARK")
    print("=" * 70)

    # 1. Load scores and existing 8D features
    df_score = pd.read_csv("results/scores_lol_v1_our485.csv")
    feat_8d_path = "results/features_lol_v1_our485.csv"
    df_8d = pd.read_csv(feat_8d_path)
    validate_feature_dataframe(df_8d, FEATURE_NAMES)

    # 2. Extract or Load 10D Features
    feat_10d_path = "results/features_lol_v1_10d.csv"
    if os.path.exists(feat_10d_path):
        print(f"Loading existing 10D features from {feat_10d_path}...")
        df_10d = pd.read_csv(feat_10d_path)
    else:
        print("Extracting 10D physical features for all 485 LOL-v1 images...")
        rows_10d = []
        low_dir = "data/lol_dataset/our485/low"
        for fn in df_score["filename"]:
            img_path = os.path.join(low_dir, fn)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
            f_dict = extract_features(img, include_engineered=False, include_all_physical=True)
            f_dict["filename"] = fn
            rows_10d.append(f_dict)
        df_10d = pd.DataFrame(rows_10d)
        validate_feature_dataframe(df_10d, PHYSICAL_10_DESCRIPTORS)
        df_10d.to_csv(feat_10d_path, index=False)
        print(f"Saved: {feat_10d_path}")

    validate_feature_dataframe(df_10d, PHYSICAL_10_DESCRIPTORS)

    # Merge with scores
    df_merged_8d = pd.merge(df_8d, df_score, on="filename")
    df_merged_10d = pd.merge(df_10d, df_score, on="filename")
    n = len(df_score)

    splits_path = "results/scene_splits_full.csv"
    if os.path.exists(splits_path):
        df_splits = pd.read_csv(splits_path)
        fn_to_group = dict(zip(df_splits["filename"], df_splits["scene_id"]))
        groups = np.array([fn_to_group.get(fn, idx) for idx, fn in enumerate(df_score["filename"])])
    else:
        groups = np.arange(n)

    y_oracle = df_score["label"].values
    oracle_psnrs = df_score["oracle_psnr"].values

    X_8d = df_merged_8d[FEATURE_NAMES].values
    X_10d = df_merged_10d[PHYSICAL_10_DESCRIPTORS].values

    seeds = [0, 1, 2]
    all_runs_8d = []
    all_runs_10d = []

    print("\nEvaluating 8D vs 10D under 5-Fold StratifiedGroupKFold across 3 seeds...")

    for seed in seeds:
        folds = make_folds(X_8d, y_oracle, groups, n_splits=5, seed=seed)

        oof_8d = np.empty(n, dtype=object)
        oof_10d = np.empty(n, dtype=object)

        for train_idx, test_idx in folds:
            # 8D Selector
            sel_8d = make_selector("logreg", feature_names=FEATURE_NAMES, random_state=seed)
            sel_8d.fit(X_8d[train_idx], y_oracle[train_idx])
            oof_8d[test_idx] = sel_8d.predict(X_8d[test_idx])

            # 10D Selector
            sel_10d = make_selector("logreg", feature_names=PHYSICAL_10_DESCRIPTORS, random_state=seed)
            sel_10d.fit(X_10d[train_idx], y_oracle[train_idx])
            oof_10d[test_idx] = sel_10d.predict(X_10d[test_idx])

        # Compute regrets
        r_8d = np.array([oracle_psnrs[i] - df_score.iloc[i][f"psnr_{oof_8d[i]}"] for i in range(n)])
        r_10d = np.array([oracle_psnrs[i] - df_score.iloc[i][f"psnr_{oof_10d[i]}"] for i in range(n)])

        acc_8d = float(np.mean(oof_8d == y_oracle)) * 100.0
        acc_10d = float(np.mean(oof_10d == y_oracle)) * 100.0

        all_runs_8d.append({"seed": seed, "regrets": r_8d, "acc": acc_8d})
        all_runs_10d.append({"seed": seed, "regrets": r_10d, "acc": acc_10d})

    # Aggregate metrics
    mean_reg_8d = float(np.mean([np.mean(r["regrets"]) for r in all_runs_8d]))
    std_reg_8d = float(np.std([np.mean(r["regrets"]) for r in all_runs_8d]))
    mean_acc_8d = float(np.mean([r["acc"] for r in all_runs_8d]))
    std_acc_8d = float(np.std([r["acc"] for r in all_runs_8d]))

    mean_reg_10d = float(np.mean([np.mean(r["regrets"]) for r in all_runs_10d]))
    std_reg_10d = float(np.std([np.mean(r["regrets"]) for r in all_runs_10d]))
    mean_acc_10d = float(np.mean([r["acc"] for r in all_runs_10d]))
    std_acc_10d = float(np.std([r["acc"] for r in all_runs_10d]))

    # Paired differences (pooled over all 3 seeds)
    concat_r_8d = np.concatenate([r["regrets"] for r in all_runs_8d])
    concat_r_10d = np.concatenate([r["regrets"] for r in all_runs_10d])
    pooled_groups = np.tile(groups, len(seeds))
    resampler = GroupResampler(pooled_groups, n_bootstraps=1000, seed=42)
    d_point, d_lo, d_hi = paired_ci(concat_r_10d, concat_r_8d, resampler)
    d_mean = d_point
    t_stat, p_val = ttest_rel(concat_r_10d, concat_r_8d)

    print("\n--- Feature Expansion Comparison ---")
    print(f"8D Baseline:  Regret = {mean_reg_8d:.3f} +/- {std_reg_8d:.3f} dB | Top-1 = {mean_acc_8d:.2f}% +/- {std_acc_8d:.2f}%")
    print(f"10D Physical: Regret = {mean_reg_10d:.3f} +/- {std_reg_10d:.3f} dB | Top-1 = {mean_acc_10d:.2f}% +/- {std_acc_10d:.2f}%")
    print(f"Paired Delta d (10D - 8D): {d_mean:+.4f} dB [95% CI: {d_lo:+.4f}, {d_hi:+.4f}] (p = {p_val:.4f})")

    df_cmp = pd.DataFrame([
        {
            "feature_set": "8D Baseline",
            "features_included": ", ".join(FEATURE_NAMES),
            "n_features": len(FEATURE_NAMES),
            "mean_regret_db": round(mean_reg_8d, 3),
            "std_regret_db": round(std_reg_8d, 3),
            "top1_acc_pct": round(mean_acc_8d, 2),
            "std_acc_pct": round(std_acc_8d, 2),
            "d_vs_8d_db": 0.000,
            "d_ci_lo": 0.000,
            "d_ci_hi": 0.000,
            "p_value": 1.000,
        },
        {
            "feature_set": "10D Physical",
            "features_included": ", ".join(PHYSICAL_10_DESCRIPTORS),
            "n_features": len(PHYSICAL_10_DESCRIPTORS),
            "mean_regret_db": round(mean_reg_10d, 3),
            "std_regret_db": round(std_reg_10d, 3),
            "top1_acc_pct": round(mean_acc_10d, 2),
            "std_acc_pct": round(std_acc_10d, 2),
            "d_vs_8d_db": round(d_mean, 4),
            "d_ci_lo": round(d_lo, 4),
            "d_ci_hi": round(d_hi, 4),
            "p_value": round(p_val, 4),
        }
    ])
    os.makedirs("results/phase1", exist_ok=True)
    df_cmp.to_csv("results/phase1/feature_expansion_comparison.csv", index=False)
    print("Saved: results/phase1/feature_expansion_comparison.csv")

    # 3. Fit 10D model on all 485 samples to analyze feature weights
    final_sel_10d = make_selector("logreg", feature_names=PHYSICAL_10_DESCRIPTORS, random_state=42)
    final_sel_10d.fit(X_10d, y_oracle)
    lr_estimator = final_sel_10d.pipeline.steps[-1][1]
    # Mean absolute coefficient across all classes
    mean_abs_coef = np.mean(np.abs(lr_estimator.coef_), axis=0)
    rel_importance_pct = (mean_abs_coef / np.sum(mean_abs_coef)) * 100.0

    df_coef = pd.DataFrame({
        "feature": PHYSICAL_10_DESCRIPTORS,
        "mean_abs_coefficient": np.round(mean_abs_coef, 4),
        "relative_weight_pct": np.round(rel_importance_pct, 2)
    }).sort_values(by="relative_weight_pct", ascending=False)

    df_coef.to_csv("results/phase1/feature_10d_coefficients.csv", index=False)
    print("Saved: results/phase1/feature_10d_coefficients.csv")

    # 4. Generate Figure 13: Feature Expansion & Importance
    os.makedirs("results/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: 8D vs 10D Performance
    sets = ["8D Baseline", "10D Physical"]
    regs = [mean_reg_8d, mean_reg_10d]
    reg_errs = [std_reg_8d, std_reg_10d]
    accs = [mean_acc_8d, mean_acc_10d]
    acc_errs = [std_acc_8d, std_acc_10d]

    x = np.arange(len(sets))
    w = 0.35

    ax1.bar(x - w/2, regs, w, yerr=reg_errs, capsize=5, label="Regret (dB) $\\downarrow$", color="#1f77b4", edgecolor="black")
    ax1.set_ylabel("Mean Regret (dB) $\\downarrow$", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.set_ylim(0, 2.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sets, fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + w/2, accs, w, yerr=acc_errs, capsize=5, label="Top-1 Acc (%) $\\uparrow$", color="#2ca02c", edgecolor="black")
    ax1_twin.set_ylabel("Top-1 Accuracy (%) $\\uparrow$", fontsize=11, fontweight="bold", color="#2ca02c")
    ax1_twin.set_ylim(40, 75)
    ax1_twin.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title("(a) 8D vs. 10D Regret and Accuracy", fontsize=12, fontweight="bold")

    # Panel B: 10D Feature Importance Weights
    features_sorted = [f.replace("_", "\\_") for f in df_coef["feature"]][::-1]
    weights_sorted = df_coef["relative_weight_pct"].values[::-1]
    bar_colors = ["#ff7f0e" if ("transmission" in f or "dynamic" in f) else "#1f77b4" for f in df_coef["feature"][::-1]]

    bars = ax2.barh(features_sorted, weights_sorted, color=bar_colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Relative Decision Weight (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) 10D Feature Decision Weights (Orange = Expanded)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
    for b in bars:
        width = b.get_width()
        ax2.text(width + 0.3, b.get_y() + b.get_height()/2.0, f"{width:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path = "results/figures/fig13_feature_expansion.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated Figure 13: {fig_path}")

    return df_cmp, df_coef


if __name__ == "__main__":
    run_feature_expansion()
