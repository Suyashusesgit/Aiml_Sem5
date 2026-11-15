"""
Phase 1 Module 2: RQ2 Study on PSNR vs. SSIM Oracle Agreement & Metric Tradeoffs
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Investigates:
1. Oracle label agreement rate between PSNR-optimal and SSIM-optimal enhancers across 485 LOL-v1 scenes.
2. 6x6 contingency matrix between PSNR oracle and SSIM oracle.
3. Cross-metric regret analysis: PSNR loss when optimizing SSIM, and SSIM loss when optimizing PSNR.
4. Performance of deployed ML selector, Best-Fixed (Retinex), and Random baseline on both metrics.
5. Average Spearman rank correlation (rho) and Kendall's tau between PSNR and SSIM rankings.
6. Generates results/phase1/rq2_oracle_agreement.csv, results/phase1/rq2_metric_tradeoff.csv,
   and results/figures/fig14_psnr_ssim_tradeoff.png.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import CANONICAL_METHODS
from src.features import FEATURE_NAMES, validate_feature_dataframe
from src.selectors import make_selector
from src.train import make_folds


def run_rq2_study():
    print("=" * 70)
    print("PHASE 1 · MODULE 2: RQ2 PSNR VS. SSIM ORACLE AGREEMENT & TRADEOFFS")
    print("=" * 70)

    score_path = "results/scores_lol_v1_our485.csv"
    feat_path = "results/features_lol_v1_our485.csv"
    if not os.path.exists(score_path) or not os.path.exists(feat_path):
        raise FileNotFoundError("Missing scores or features file.")

    df_score = pd.read_csv(score_path)
    df_feat = pd.read_csv(feat_path)
    df = pd.merge(df_feat, df_score, on="filename")

    methods = list(CANONICAL_METHODS)
    n = len(df)

    psnr_matrix = np.zeros((n, len(methods)), dtype=np.float32)
    ssim_matrix = np.zeros((n, len(methods)), dtype=np.float32)

    for j, m in enumerate(methods):
        psnr_matrix[:, j] = df[f"psnr_{m}"].values
        ssim_matrix[:, j] = df[f"ssim_{m}"].values

    # Determine PSNR oracle and SSIM oracle
    psnr_oracle_idx = np.argmax(psnr_matrix, axis=1)
    ssim_oracle_idx = np.argmax(ssim_matrix, axis=1)

    psnr_oracle_labels = np.array([methods[i] for i in psnr_oracle_idx])
    ssim_oracle_labels = np.array([methods[i] for i in ssim_oracle_idx])

    oracle_max_psnr = np.max(psnr_matrix, axis=1)
    oracle_max_ssim = np.max(ssim_matrix, axis=1)

    # 1. Oracle Agreement
    exact_agreement_mask = (psnr_oracle_idx == ssim_oracle_idx)
    exact_agreement_rate = float(np.mean(exact_agreement_mask)) * 100.0
    print(f"\nExact Oracle Agreement (PSNR vs. SSIM): {exact_agreement_rate:.2f}% ({np.sum(exact_agreement_mask)}/{n} pairs)")

    # 2. Rank Correlations
    spearman_rhos = []
    kendall_taus = []
    for i in range(n):
        rho, _ = spearmanr(psnr_matrix[i], ssim_matrix[i])
        tau, _ = kendalltau(psnr_matrix[i], ssim_matrix[i])
        if not np.isnan(rho):
            spearman_rhos.append(rho)
        if not np.isnan(tau):
            kendall_taus.append(tau)

    mean_rho = float(np.mean(spearman_rhos))
    mean_tau = float(np.mean(kendall_taus))
    print(f"Mean Method Ranking Spearman Rho: {mean_rho:.4f}")
    print(f"Mean Method Ranking Kendall Tau:  {mean_tau:.4f}")

    # 3. 6x6 Contingency Matrix
    contingency = pd.crosstab(
        pd.Series(psnr_oracle_labels, name="PSNR_Oracle"),
        pd.Series(ssim_oracle_labels, name="SSIM_Oracle")
    ).reindex(index=methods, columns=methods, fill_value=0)

    print("\n--- Contingency Matrix: PSNR Oracle (rows) vs. SSIM Oracle (cols) ---")
    print(contingency)

    # Save contingency matrix
    os.makedirs("results/phase1", exist_ok=True)
    contingency.to_csv("results/phase1/rq2_oracle_contingency_matrix.csv")

    # 4. Out-of-fold selector predictions
    splits_path = "results/scene_splits_full.csv"
    if os.path.exists(splits_path):
        df_splits = pd.read_csv(splits_path)
        fn_to_group = dict(zip(df_splits["filename"], df_splits["scene_id"]))
        groups = np.array([fn_to_group.get(fn, idx) for idx, fn in enumerate(df["filename"])])
    else:
        groups = np.arange(len(df))

    X = df[FEATURE_NAMES].values
    y_psnr = psnr_oracle_labels
    folds = make_folds(X, y_psnr, groups, n_splits=5, seed=42)

    oof_preds_psnr = np.empty(n, dtype=object)
    for train_idx, test_idx in folds:
        selector = make_selector("logreg", random_state=42)
        selector.fit(X[train_idx], y_psnr[train_idx])
        oof_preds_psnr[test_idx] = selector.predict(X[test_idx])

    # 5. Dual-Metric Benchmark Table
    # For each strategy, evaluate Mean PSNR, PSNR Regret (dB), Mean SSIM, SSIM Regret
    strategies = {
        "PSNR Oracle": psnr_oracle_labels,
        "SSIM Oracle": ssim_oracle_labels,
        "Deployed Selector (LogReg)": oof_preds_psnr,
        "Best-Fixed (Retinex)": np.array(["retinex"] * n),
        "Heuristic Hand-Rule": None,  # Will compute below
        "Uniform Random": None,
    }

    # Heuristic Hand-Rule
    from src.selectors import make_selector as ms
    rule_selector = ms("rule")
    strategies["Heuristic Hand-Rule"] = rule_selector.predict(X)

    tradeoff_records = []
    print("\n--- Dual-Metric Performance Across Strategies ---")
    print(f"{'Strategy':<28} | {'PSNR (dB)':<10} | {'PSNR Regret (dB)':<17} | {'SSIM':<8} | {'SSIM Regret':<12}")
    print("-" * 85)

    for name, picks in strategies.items():
        if name == "Uniform Random":
            # Compute analytical expectation over random picks
            psnr_vals = np.mean(psnr_matrix, axis=1)
            ssim_vals = np.mean(ssim_matrix, axis=1)
        else:
            psnr_vals = np.array([df.iloc[i][f"psnr_{picks[i]}"] for i in range(n)])
            ssim_vals = np.array([df.iloc[i][f"ssim_{picks[i]}"] for i in range(n)])

        mean_p = float(np.mean(psnr_vals))
        p_regret = float(np.mean(np.maximum(0.0, oracle_max_psnr - psnr_vals)))
        mean_s = float(np.mean(ssim_vals))
        s_regret = float(np.mean(np.maximum(0.0, oracle_max_ssim - ssim_vals)))

        print(f"{name:<28} | {mean_p:>9.3f} | {p_regret:>16.3f} | {mean_s:>7.4f} | {s_regret:>11.4f}")

        tradeoff_records.append({
            "strategy": name,
            "mean_psnr_db": round(mean_p, 3),
            "psnr_regret_db": round(p_regret, 3),
            "mean_ssim": round(mean_s, 4),
            "ssim_regret": round(s_regret, 4),
        })

    df_tradeoff = pd.DataFrame(tradeoff_records)
    df_tradeoff.to_csv("results/phase1/rq2_metric_tradeoff.csv", index=False)
    print("Saved: results/phase1/rq2_metric_tradeoff.csv")

    # Save summary agreement metrics
    df_agreement = pd.DataFrame([{
        "exact_oracle_agreement_pct": round(exact_agreement_rate, 2),
        "mean_spearman_rho": round(mean_rho, 4),
        "mean_kendall_tau": round(mean_tau, 4),
        "n_samples": n,
    }])
    df_agreement.to_csv("results/phase1/rq2_oracle_agreement.csv", index=False)
    print("Saved: results/phase1/rq2_oracle_agreement.csv")

    # 6. Generate Figure 14: PSNR vs SSIM Tradeoff & Confusion Matrix Heatmap
    os.makedirs("results/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Heatmap of Oracle Agreement
    c_matrix = contingency.values.astype(int)
    im = ax1.imshow(c_matrix, cmap="Blues", interpolation="nearest")
    ax1.set_xticks(range(len(methods)))
    ax1.set_yticks(range(len(methods)))
    ax1.set_xticklabels([m.replace("_", "\\_") for m in methods], rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels([m.replace("_", "\\_") for m in methods], fontsize=9)
    ax1.set_xlabel("SSIM Oracle Preferred Operator", fontsize=11, fontweight="bold")
    ax1.set_ylabel("PSNR Oracle Preferred Operator", fontsize=11, fontweight="bold")
    ax1.set_title(f"(a) Oracle Alignment ({exact_agreement_rate:.1f}% Agreement)", fontsize=12, fontweight="bold")

    # Text annotations inside heatmap cells
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = c_matrix[i, j]
            color = "white" if val > (c_matrix.max() / 2.0) else "black"
            ax1.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Panel B: Scatter plot of Regret Tradeoff
    p_reg = [r["psnr_regret_db"] for r in tradeoff_records]
    s_reg = [r["ssim_regret"] for r in tradeoff_records]
    names = [r["strategy"] for r in tradeoff_records]
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e", "#9467bd", "#7f7f7f"]
    markers = ["*", "P", "o", "s", "^", "X"]

    for x, y, name, color, m in zip(p_reg, s_reg, names, colors, markers):
        sz = 140 if "Deployed" in name or "Oracle" in name else 100
        ax2.scatter(x, y, color=color, marker=m, s=sz, label=name, edgecolors="black", linewidth=1.2, zorder=5)

    ax2.set_xlabel("PSNR Regret vs Oracle (dB) $\\downarrow$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("SSIM Regret vs Oracle $\\downarrow$", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Multi-Metric Objective Space", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig_path = "results/figures/fig14_psnr_ssim_tradeoff.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated Figure 14: {fig_path}")

    return df_agreement, df_tradeoff


if __name__ == "__main__":
    run_rq2_study()
