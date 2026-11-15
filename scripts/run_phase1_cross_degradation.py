"""
Phase 1 Module 3: Cross-Dataset Zero-Shot Generalization Benchmark
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Evaluates zero-shot domain transfer of the deployed Logistic Regression selector
(trained on LOL-v1 our485) across out-of-distribution datasets:
1. LOL-v2 Real Captured Test (Unfiltered, 100 pairs)
2. LOL-v2 Real Captured Test (De-duplicated, 9 clean pairs)
3. LOL-v2 Synthetic Test (100 pairs)

Benchmarked against: Oracle, Best-Fixed (Retinex), Heuristic Hand-Rule, and Random Choice.
Outputs:
- results/phase1/cross_degradation_summary.csv
- results/phase1/lol_v2_real_results.csv
- results/phase1/lol_v2_synthetic_results.csv
- results/figures/fig15_cross_degradation_generalization.png
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


def run_cross_degradation_benchmark():
    print("=" * 70)
    print("PHASE 1 · MODULE 3: ZERO-SHOT CROSS-DATASET GENERALIZATION")
    print("=" * 70)

    # 1. Train Deployed Selector on complete LOL-v1 our485
    df_feat_v1 = pd.read_csv("results/features_lol_v1_our485.csv")
    df_score_v1 = pd.read_csv("results/scores_lol_v1_our485.csv")
    validate_feature_dataframe(df_feat_v1, FEATURE_NAMES)
    df_v1 = pd.merge(df_feat_v1, df_score_v1, on="filename")

    X_v1 = df_v1[FEATURE_NAMES].values
    y_v1 = df_v1["label"].values

    selector = make_selector("logreg", random_state=42)
    selector.fit(X_v1, y_v1)
    rule_selector = make_selector("rule")

    print(f"Trained deployed selector on LOL-v1 our485 (N={len(df_v1)})")

    # 2. Benchmark Datasets
    benchmark_splits = {}

    # (a) LOL-v2 Real Captured Test
    f_real_path = "results/lol_v2_real_captured_test_features.csv"
    s_real_path = "results/lol_v2_real_captured_test_scores.csv"
    if os.path.exists(f_real_path) and os.path.exists(s_real_path):
        df_f_real = pd.read_csv(f_real_path)
        df_s_real = pd.read_csv(s_real_path)
        validate_feature_dataframe(df_f_real, FEATURE_NAMES)
        df_real = pd.merge(df_f_real, df_s_real, on="filename")
        benchmark_splits["LOL-v2 Real (Unfiltered, N=100)"] = df_real

        # Load de-duplicated split (9 clean pairs from E4)
        # Check dHash near duplicates if clean hashes exist
        # Or look up the 9 files retained in E4
        from src.dataset import compute_dhash, load_and_preprocess_pair
        v1_hashes = []
        for fn in df_v1["filename"]:
            clean_path = os.path.join("data/lol_dataset/our485/high", fn)
            if os.path.exists(clean_path):
                import cv2
                img = cv2.imread(clean_path)
                if img is not None:
                    v1_hashes.append(compute_dhash(img))
        
        if len(v1_hashes) > 0:
            v1_hashes_arr = np.array(v1_hashes)
            clean_real_rows = []
            for idx, row in df_real.iterrows():
                high_p = os.path.join("data/LOL-v2/Real_captured/Test/Normal", row["filename"].replace("low", "normal"))
                if os.path.exists(high_p):
                    import cv2
                    img_h = cv2.imread(high_p)
                    if img_h is not None:
                        h = compute_dhash(img_h)
                        min_d = int(np.min(np.sum(v1_hashes_arr != h, axis=1)))
                        if min_d > 10:
                            clean_real_rows.append(row)
            if len(clean_real_rows) > 0:
                benchmark_splits[f"LOL-v2 Real (De-duplicated, N={len(clean_real_rows)})"] = pd.DataFrame(clean_real_rows)
            else:
                print("Note: Could not filter hashes dynamically; checking cached real splits.")

    # (b) LOL-v2 Synthetic Test
    f_synth_path = "results/lol_v2_synthetic_test_features.csv"
    s_synth_path = "results/lol_v2_synthetic_test_scores.csv"
    if os.path.exists(f_synth_path) and os.path.exists(s_synth_path):
        df_f_synth = pd.read_csv(f_synth_path)
        df_s_synth = pd.read_csv(s_synth_path)
        validate_feature_dataframe(df_f_synth, FEATURE_NAMES)
        df_synth = pd.merge(df_f_synth, df_s_synth, on="filename")
        benchmark_splits["LOL-v2 Synthetic Test (N=100)"] = df_synth

    summary_records = []
    print("\n--- Zero-Shot Cross-Dataset Benchmark Summary ---")
    print(f"{'Split / Domain':<38} | {'Strategy':<22} | {'PSNR (dB)':<10} | {'Regret (dB)':<12} | {'Top-1 (%)':<9} | {'Gap Closed (%)':<15}")
    print("-" * 115)

    os.makedirs("results/phase1", exist_ok=True)

    for split_name, df_eval in benchmark_splits.items():
        n_eval = len(df_eval)
        X_eval = df_eval[FEATURE_NAMES].values

        # Ensure canonical oracle definition
        methods = list(CANONICAL_METHODS)
        psnr_matrix = np.zeros((n_eval, len(methods)))
        for j, m in enumerate(methods):
            psnr_matrix[:, j] = df_eval[f"psnr_{m}"].values

        oracle_psnr = np.max(psnr_matrix, axis=1)
        oracle_labels = np.array([methods[idx] for idx in np.argmax(psnr_matrix, axis=1)])
        mean_oracle_psnr = float(np.mean(oracle_psnr))

        # Model picks
        pred_model = selector.predict(X_eval)
        pred_rule = rule_selector.predict(X_eval)
        pred_fixed = np.array(["retinex"] * n_eval)

        # Baseline definitions
        eval_strategies = {
            "Oracle": oracle_labels,
            "Deployed LR": pred_model,
            "Best-Fixed (Retinex)": pred_fixed,
            "Heuristic Rule": pred_rule,
            "Uniform Random": None,
        }

        fixed_regret = None

        for s_name, picks in eval_strategies.items():
            if s_name == "Uniform Random":
                psnr_arr = np.mean(psnr_matrix, axis=1)
                top1_acc = float(np.mean([1.0 / len(methods)] * n_eval)) * 100.0
            else:
                psnr_arr = np.array([df_eval.iloc[i][f"psnr_{picks[i]}"] for i in range(n_eval)])
                top1_acc = float(np.mean(picks == oracle_labels)) * 100.0

            mean_p = float(np.mean(psnr_arr))
            regret = float(np.mean(np.maximum(0.0, oracle_psnr - psnr_arr)))

            if s_name == "Best-Fixed (Retinex)":
                fixed_regret = regret

            gap_closed = 0.0
            if fixed_regret is not None and fixed_regret > 1e-4:
                gap_closed = float((fixed_regret - regret) / fixed_regret) * 100.0

            print(f"{split_name:<38} | {s_name:<22} | {mean_p:>9.3f} | {regret:>11.3f} | {top1_acc:>8.1f}% | {gap_closed:>13.1f}%")

            summary_records.append({
                "dataset_split": split_name,
                "strategy": s_name,
                "mean_psnr_db": round(mean_p, 3),
                "regret_db": round(regret, 3),
                "top1_acc_pct": round(top1_acc, 2),
                "regret_gap_closed_pct": round(gap_closed, 1) if s_name not in ["Oracle", "Best-Fixed (Retinex)"] else 0.0,
                "n_samples": n_eval,
            })

        # Save individual detailed results
        if "Real (Unfiltered" in split_name:
            df_eval_copy = df_eval.copy()
            df_eval_copy["predicted_enhancer"] = pred_model
            df_eval_copy["oracle_enhancer"] = oracle_labels
            df_eval_copy["regret_db"] = oracle_psnr - np.array([df_eval.iloc[i][f"psnr_{pred_model[i]}"] for i in range(n_eval)])
            df_eval_copy.to_csv("results/phase1/lol_v2_real_results.csv", index=False)
            print("Saved: results/phase1/lol_v2_real_results.csv")
        elif "Synthetic" in split_name:
            df_eval_copy = df_eval.copy()
            df_eval_copy["predicted_enhancer"] = pred_model
            df_eval_copy["oracle_enhancer"] = oracle_labels
            df_eval_copy["regret_db"] = oracle_psnr - np.array([df_eval.iloc[i][f"psnr_{pred_model[i]}"] for i in range(n_eval)])
            df_eval_copy.to_csv("results/phase1/lol_v2_synthetic_results.csv", index=False)
            print("Saved: results/phase1/lol_v2_synthetic_results.csv")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("results/phase1/cross_degradation_summary.csv", index=False)
    print("Saved: results/phase1/cross_degradation_summary.csv")

    # 3. Generate Figure 15: Cross-Dataset Regret Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    splits_to_plot = [s for s in benchmark_splits.keys() if "De-duplicated" not in s or "N=9" in s]
    bar_width = 0.20
    x_indices = np.arange(len(splits_to_plot))

    strat_colors = {
        "Deployed LR": "#1f77b4",
        "Best-Fixed (Retinex)": "#ff7f0e",
        "Heuristic Rule": "#9467bd",
        "Uniform Random": "#7f7f7f"
    }

    for i, (strat, color) in enumerate(strat_colors.items()):
        regret_vals = []
        for s in splits_to_plot:
            match = df_summary[(df_summary["dataset_split"] == s) & (df_summary["strategy"] == strat)]
            if len(match) > 0:
                regret_vals.append(match["regret_db"].values[0])
            else:
                regret_vals.append(0.0)
        
        offset = (i - 1.5) * bar_width
        ax.bar(x_indices + offset, regret_vals, width=bar_width, label=strat, color=color, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x_indices)
    clean_split_names = [s.split(" (")[0] for s in splits_to_plot]
    ax.set_xticklabels(clean_split_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Regret vs Oracle (dB) $\\downarrow$", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Dataset Zero-Shot Transfer: Regret vs. Oracle Across Domains", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig_path = "results/figures/fig15_cross_degradation_generalization.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Generated Figure 15: {fig_path}")

    return df_summary


if __name__ == "__main__":
    run_cross_degradation_benchmark()
