"""
Phase 1 Master Pipeline Orchestrator
Authors: Team Triathlon (GSV Vadodara / SVNIT Surat)
Supervisor: Prof. Kishor P. Upla

Orchestrates the entire Phase 1 investigation:
1. Conformal Prediction & Risk-Controlled Abstention sets
2. RQ2 Study on PSNR vs. SSIM Oracle Agreement
3. Zero-Shot Cross-Dataset Transfer (LOL-v2 Real & Synthetic)
4. Leave-One-Enhancer-Out (LOEO) Resilience Analysis
5. 10D Physical Feature Expansion Benchmark

Outputs comprehensive summary table and ensures all Phase 1 artifacts are generated.
"""

import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_phase1_conformal import run_conformal_benchmark
from scripts.run_phase1_rq2_oracle_study import run_rq2_study
from scripts.run_phase1_cross_degradation import run_cross_degradation_benchmark
from scripts.run_phase1_leave_one_out import run_loeo_analysis
from scripts.run_phase1_feature_expansion import run_feature_expansion


def run_all_phase1():
    t_start = time.time()
    print("=" * 80)
    print("   PROJECT TRIATHLON · PHASE 1 MASTER EXPERIMENTAL PIPELINE")
    print("   Edge Feasibility, Conformal Risk Control & Cross-Degradation")
    print("=" * 80)

    # Module 1
    t0 = time.time()
    df_calib, df_abstain = run_conformal_benchmark()
    print(f">> Module 1 completed in {time.time() - t0:.2f}s\n")

    # Module 2
    t0 = time.time()
    df_agreement, df_tradeoff = run_rq2_study()
    print(f">> Module 2 completed in {time.time() - t0:.2f}s\n")

    # Module 3
    t0 = time.time()
    df_cross = run_cross_degradation_benchmark()
    print(f">> Module 3 completed in {time.time() - t0:.2f}s\n")

    # Module 4
    t0 = time.time()
    df_loeo = run_loeo_analysis()
    print(f">> Module 4 completed in {time.time() - t0:.2f}s\n")

    # Module 5
    t0 = time.time()
    df_feat_cmp, df_coef = run_feature_expansion()
    print(f">> Module 5 completed in {time.time() - t0:.2f}s\n")

    total_time = time.time() - t_start

    print("=" * 80)
    print(f"   PHASE 1 EXECUTION COMPLETE IN {total_time:.2f}s")
    print("=" * 80)

    # Executive Summary Printout
    print("\n--- PHASE 1 EXECUTIVE RESULTS SUMMARY ---")
    print("1. Conformal Coverage at 90% Target: "
          f"{df_calib[df_calib['target_coverage_pct'] == 90.0]['empirical_coverage_pct'].values[0]}% "
          f"(Avg Set Size: {df_calib[df_calib['target_coverage_pct'] == 90.0]['average_set_size'].values[0]})")
    
    print(f"2. Risk-Controlled Abstention at tau=0.60: Top-1 Acc = "
          f"{df_abstain[df_abstain['confidence_threshold_tau'] == 0.60]['accepted_top1_acc_pct'].values[0]}%, "
          f"Regret = {df_abstain[df_abstain['confidence_threshold_tau'] == 0.60]['accepted_mean_regret_db'].values[0]} dB "
          f"(Abstain Rate: {df_abstain[df_abstain['confidence_threshold_tau'] == 0.60]['abstain_rate_pct'].values[0]}%)")

    print(f"3. RQ2 PSNR vs. SSIM Oracle Agreement: "
          f"{df_agreement['exact_oracle_agreement_pct'].values[0]}% "
          f"(Ranking Spearman rho = {df_agreement['mean_spearman_rho'].values[0]})")

    synth_lr = df_cross[(df_cross["dataset_split"] == "LOL-v2 Synthetic Test (N=100)") & (df_cross["strategy"] == "Deployed LR")]
    synth_fixed = df_cross[(df_cross["dataset_split"] == "LOL-v2 Synthetic Test (N=100)") & (df_cross["strategy"] == "Best-Fixed (Retinex)")]
    print(f"4. Cross-Dataset Synthetic Generalization: Deployed LR = {synth_lr['mean_psnr_db'].values[0]} dB "
          f"vs. Best-Fixed Retinex = {synth_fixed['mean_psnr_db'].values[0]} dB (+{synth_fixed['regret_db'].values[0] - synth_lr['regret_db'].values[0]:.2f} dB advantage)")

    most_imp = df_loeo[df_loeo["marginal_importance_rank"] == 1]["excluded_enhancer"].values[0]
    imp_regret = df_loeo[df_loeo["marginal_importance_rank"] == 1]["regret_vs_full_oracle_db"].values[0]
    print(f"5. LOEO Most Critical Enhancer: '{most_imp}' (regret rises to {imp_regret:.3f} dB without it)")

    diff_val = df_feat_cmp[df_feat_cmp["feature_set"] == "10D Physical"]["d_vs_8d_db"].values[0]
    p_val = df_feat_cmp[df_feat_cmp["feature_set"] == "10D Physical"]["p_value"].values[0]
    print(f"6. 10D Physical Feature Expansion: Delta d = {diff_val:+.4f} dB (p = {p_val:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    run_all_phase1()
