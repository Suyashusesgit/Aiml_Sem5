"""
Main Execution Script
Runs the end-to-end experimental pipeline from raw dataset to final regret tables:
1. Validates environment and directories
2. Extracts features and computes oracle labels on LOL v1 training pairs
3. Executes 5-fold GroupKFold cross-validation over 3 random seeds (15 runs per model)
4. Evaluates baselines (Random, Best Fixed, Hand-Written Rule) and ML classifiers (LogReg, RF, XGBoost)
5. Generates publication-ready figures for IEEE paper
6. Runs feature ablation studies and official eval15 test split verification
7. Performs error analysis on the 10 worst predictions
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import yaml
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, ENGINEERED_FEATURE_NAMES
from src.train import (
    seed_everything,
    build_or_load_dataset_cache,
    run_cross_validation,
)
from src.evaluate import (
    plot_oracle_distribution,
    plot_confusion_matrix,
    plot_regret_cdf,
    plot_feature_importances,
    run_feature_ablations,
    evaluate_official_test_split,
    extract_worst_errors,
    compute_statistical_significance,
    plot_qualitative_strip,
)


def main():
    print("=" * 75)
    print("IMAGE ENHANCEMENT AND PER-IMAGE SELECTION (PHASE 0 BENCHMARK)")
    print("=" * 75)

    config_path = "configs/base.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("random_seed", 42))

    fig_dir = config["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(config["paths"]["reports_dir"], exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Step 1 & 2: Dataset Caching & Feature Extraction
    print("\n>>> Step 1 & 2: Dataset Cache (Features & Oracle PSNR/SSIM Scores)...")
    df_scores, df_features = build_or_load_dataset_cache(config)

    print("\n--- Oracle Label Distribution ---")
    print(df_scores["label"].value_counts())

    # Generate Oracle Distribution Figure
    oracle_fig = os.path.join(fig_dir, "fig1_oracle_distribution.png")
    plot_oracle_distribution(df_scores, oracle_fig)

    # Step 3: Run Grouped Cross-Validation Benchmark
    print("\n>>> Step 3: 5-Fold Grouped Cross-Validation (3 Seeds = 15 Folds per Model)...")
    base_features = FEATURE_NAMES
    cv_results = run_cross_validation(df_scores, df_features, base_features, config)

    df_summary = cv_results["summary"]
    print("\n" + "=" * 75)
    print("PRIMARY BENCHMARK RESULTS (GroupKFold CV)")
    print("=" * 75)
    print(df_summary.to_string(index=False))
    print("=" * 75)

    metrics_csv = config["paths"]["metrics_output"]
    df_summary.to_csv(metrics_csv, index=False)
    print(f"\nSaved primary metrics table to {metrics_csv}")

    # Step 4: Generate Publication Figures
    print("\n>>> Step 4: Generating Figures for IEEE Paper...")
    y_true = cv_results["true_labels"]
    best_model_name = "Random Forest"
    y_pred = np.array(cv_results["fold_predictions"][best_model_name])

    cm_fig = os.path.join(fig_dir, "fig2_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, best_model_name, cm_fig)

    cdf_fig = os.path.join(fig_dir, "fig3_regret_cdf.png")
    plot_regret_cdf(cv_results["fold_predictions"], df_scores, cdf_fig)

    feat_fig = os.path.join(fig_dir, "fig4_feature_importances.png")
    plot_feature_importances(df_features, df_scores, base_features, feat_fig)

    qual_fig = os.path.join(fig_dir, "fig5_qualitative_strip.png")
    plot_qualitative_strip(df_scores, df_features, qual_fig)

    # Step 5: Statistical Significance Testing
    print("\n>>> Step 5: Statistical Significance Testing (Paired t-test, Wilcoxon, 95% CI)...")
    stats_csv = "results/statistical_tests.csv"
    df_stats = compute_statistical_significance(cv_results["raw_fold_results"], stats_csv)
    print(df_stats[["Baseline_Model", "Regret_Reduction_Mean_dB", "Regret_p_value", "Accuracy_Gain_Mean_pct", "Accuracy_p_value"]].to_string(index=False))

    # Step 6: Feature Ablation Study
    print("\n>>> Step 6: Running Feature Ablation Experiments...")
    df_ablation = run_feature_ablations(df_scores, df_features, config)

    # Step 7: Official Test Split (eval15)
    print("\n>>> Step 7: Evaluating on Official eval15 Test Split...")
    df_test_res = evaluate_official_test_split(
        df_scores, df_features, base_features, config
    )
    print("\n--- eval15 Test Results ---")
    print(df_test_res.to_string(index=False))

    # Step 8: Error Analysis (10 Worst Predictions)
    print("\n>>> Step 8: Qualitative Error Analysis (Top-10 Regret Cases)...")
    worst_csv = "results/worst_10_errors.csv"
    df_worst = extract_worst_errors(y_true, y_pred, df_scores, df_features, worst_csv)
    print(df_worst[["Rank", "Filename", "Oracle_Method", "Predicted_Method", "Regret_dB", "Brightness", "Contrast"]].to_string(index=False))

    print("\n" + "=" * 75)
    print("EXPERIMENT EXECUTION COMPLETE. All deliverables generated.")
    print("=" * 75)


if __name__ == "__main__":
    main()
