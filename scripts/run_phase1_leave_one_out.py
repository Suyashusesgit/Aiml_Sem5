"""
Phase 1 Module 4: Leave-One-Enhancer-Out (LOEO) Generalization Analysis
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Investigates algorithm bank resilience and marginal utility of each classical operator:
For each operator e in {identity, gamma, clahe, hist_eq, retinex, bilat_unsharp}:
1. Remove e from candidate bank (5-operator reduced bank).
2. Recompute oracle upper bound on LOL-v1 our485.
3. Train and evaluate Multinomial Logistic Regression under 5-fold StratifiedGroupKFold.
4. Measure drop in Oracle PSNR, Selector Regret vs Reduced Oracle, Selector Regret vs Full Oracle,
   and Top-1 Accuracy.
5. Saves results/phase1/leave_one_enhancer_out.csv and results/figures/fig16_leave_one_enhancer_out.png.
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
from src.train import make_folds


def run_loeo_analysis():
    print("=" * 70)
    print("PHASE 1 · MODULE 4: LEAVE-ONE-ENHANCER-OUT (LOEO) GENERALIZATION")
    print("=" * 70)

    # 1. Load LOL-v1 data
    df_feat = pd.read_csv("results/features_lol_v1_our485.csv")
    df_score = pd.read_csv("results/scores_lol_v1_our485.csv")
    validate_feature_dataframe(df_feat, FEATURE_NAMES)
    df = pd.merge(df_feat, df_score, on="filename")

    X = df[FEATURE_NAMES].values
    n = len(df)
    methods = list(CANONICAL_METHODS)

    splits_path = "results/scene_splits_full.csv"
    if os.path.exists(splits_path):
        df_splits = pd.read_csv(splits_path)
        fn_to_group = dict(zip(df_splits["filename"], df_splits["scene_id"]))
        groups = np.array([fn_to_group.get(fn, idx) for idx, fn in enumerate(df["filename"])])
    else:
        groups = np.arange(n)

    # Full Bank Oracle
    psnr_matrix_full = np.zeros((n, len(methods)))
    for j, m in enumerate(methods):
        psnr_matrix_full[:, j] = df[f"psnr_{m}"].values

    full_oracle_psnr = np.max(psnr_matrix_full, axis=1)
    mean_full_oracle = float(np.mean(full_oracle_psnr))

    # Evaluate Full Bank 6-operator LR baseline
    y_full = np.array([methods[idx] for idx in np.argmax(psnr_matrix_full, axis=1)])
    folds_full = make_folds(X, y_full, groups, n_splits=5, seed=42)

    oof_preds_full = np.empty(n, dtype=object)
    for tr, te in folds_full:
        sel = make_selector("logreg", random_state=42)
        sel.fit(X[tr], y_full[tr])
        oof_preds_full[te] = sel.predict(X[te])

    full_sel_psnr = np.array([df.iloc[i][f"psnr_{oof_preds_full[i]}"] for i in range(n)])
    full_sel_mean = float(np.mean(full_sel_psnr))
    full_sel_regret = float(np.mean(np.maximum(0.0, full_oracle_psnr - full_sel_psnr)))
    full_top1 = float(np.mean(oof_preds_full == y_full)) * 100.0

    print(f"Full Bank (6 Operators): Oracle = {mean_full_oracle:.3f} dB | Selector = {full_sel_mean:.3f} dB | Regret = {full_sel_regret:.3f} dB | Top-1 = {full_top1:.1f}%")

    records = [{
        "excluded_enhancer": "None (Full 6-Bank)",
        "bank_size": 6,
        "oracle_psnr_db": round(mean_full_oracle, 3),
        "delta_oracle_db": 0.000,
        "selector_psnr_db": round(full_sel_mean, 3),
        "regret_vs_reduced_oracle_db": round(full_sel_regret, 3),
        "regret_vs_full_oracle_db": round(full_sel_regret, 3),
        "top1_acc_pct": round(full_top1, 2),
        "marginal_importance_rank": 0,
    }]

    print("\n--- LOEO Sensitivity Across Excluded Enhancers ---")
    print(f"{'Excluded Enhancer':<20} | {'Oracle (dB)':<11} | {'Delta Oracle':<12} | {'Sel PSNR':<10} | {'Regret (vs Full)':<16} | {'Top-1 (%)':<9}")
    print("-" * 90)

    for ex_method in methods:
        reduced_methods = [m for m in methods if m != ex_method]
        psnr_matrix_red = np.zeros((n, len(reduced_methods)))
        for j, m in enumerate(reduced_methods):
            psnr_matrix_red[:, j] = df[f"psnr_{m}"].values

        oracle_psnr_red = np.max(psnr_matrix_red, axis=1)
        mean_oracle_red = float(np.mean(oracle_psnr_red))
        delta_oracle = mean_full_oracle - mean_oracle_red

        y_red = np.array([reduced_methods[idx] for idx in np.argmax(psnr_matrix_red, axis=1)])
        folds_red = make_folds(X, y_red, groups, n_splits=5, seed=42)

        oof_preds_red = np.empty(n, dtype=object)
        for tr, te in folds_red:
            sel = make_selector("logreg", random_state=42)
            sel.fit(X[tr], y_red[tr])
            oof_preds_red[te] = sel.predict(X[te])

        sel_psnr_red = np.array([df.iloc[i][f"psnr_{oof_preds_red[i]}"] for i in range(n)])
        mean_sel_red = float(np.mean(sel_psnr_red))
        regret_vs_red = float(np.mean(np.maximum(0.0, oracle_psnr_red - sel_psnr_red)))
        regret_vs_full = float(np.mean(np.maximum(0.0, full_oracle_psnr - sel_psnr_red)))
        top1_red = float(np.mean(oof_preds_red == y_red)) * 100.0

        print(f"{ex_method:<20} | {mean_oracle_red:>10.3f} | {delta_oracle:>11.3f} | {mean_sel_red:>9.3f} | {regret_vs_full:>15.3f} | {top1_red:>8.1f}%")

        records.append({
            "excluded_enhancer": ex_method,
            "bank_size": 5,
            "oracle_psnr_db": round(mean_oracle_red, 3),
            "delta_oracle_db": round(delta_oracle, 3),
            "selector_psnr_db": round(mean_sel_red, 3),
            "regret_vs_reduced_oracle_db": round(regret_vs_red, 3),
            "regret_vs_full_oracle_db": round(regret_vs_full, 3),
            "top1_acc_pct": round(top1_red, 2),
            "marginal_importance_rank": 0,  # Updated below
        })

    df_loeo = pd.DataFrame(records)
    # Rank by regret_vs_full_oracle_db descending (higher regret when excluded -> more important)
    ex_rows = df_loeo[df_loeo["bank_size"] == 5].sort_values(by="regret_vs_full_oracle_db", ascending=False)
    ranks = {row["excluded_enhancer"]: r + 1 for r, (_, row) in enumerate(ex_rows.iterrows())}
    df_loeo["marginal_importance_rank"] = df_loeo["excluded_enhancer"].map(lambda x: ranks.get(x, 0))

    os.makedirs("results/phase1", exist_ok=True)
    df_loeo.to_csv("results/phase1/leave_one_enhancer_out.csv", index=False)
    print("\nSaved: results/phase1/leave_one_enhancer_out.csv")

    # 5. Generate Figure 16: LOEO Impact Chart
    os.makedirs("results/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = df_loeo[df_loeo["bank_size"] == 5].sort_values(by="delta_oracle_db", ascending=True)
    enhancers = [m.replace("_", "\\_") for m in plot_df["excluded_enhancer"]]

    # Panel A: Oracle Capacity Loss
    bars1 = ax1.barh(enhancers, plot_df["delta_oracle_db"], color="#d62728", edgecolor="black", alpha=0.85)
    ax1.set_xlabel("Oracle PSNR Capacity Loss (dB) $\\Delta$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Theoretical Bank Capacity Loss without Enhancer", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)
    for b in bars1:
        w = b.get_width()
        ax1.text(w + 0.02, b.get_y() + b.get_height() / 2.0, f"+{w:.2f} dB", va="center", fontsize=9, fontweight="bold")

    # Panel B: Selector Regret vs Full Oracle
    plot_df_reg = df_loeo[df_loeo["bank_size"] == 5].sort_values(by="regret_vs_full_oracle_db", ascending=True)
    enhancers_reg = [m.replace("_", "\\_") for m in plot_df_reg["excluded_enhancer"]]
    bars2 = ax2.barh(enhancers_reg, plot_df_reg["regret_vs_full_oracle_db"], color="#1f77b4", edgecolor="black", alpha=0.85)
    ax2.axvline(full_sel_regret, color="gray", linestyle="--", linewidth=1.5, label=f"Full Bank Regret ({full_sel_regret:.2f} dB)")
    ax2.set_xlabel("Effective Regret vs Full 6-Bank Oracle (dB) $\\downarrow$", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Realized ML Selector Regret under Operator Removal", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)
    for b in bars2:
        w = b.get_width()
        ax2.text(w + 0.03, b.get_y() + b.get_height() / 2.0, f"{w:.2f} dB", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path = "results/figures/fig16_leave_one_enhancer_out.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated Figure 16: {fig_path}")

    return df_loeo


if __name__ == "__main__":
    run_loeo_analysis()
