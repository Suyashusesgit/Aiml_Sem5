"""
Downstream Computer Vision Utility Evaluation (Phase 1 Early Preview)
Team Triathlon: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan
Supervisor: Prof. Kishor P. Upla, SVNIT Surat

Evaluates whether fidelity-driven algorithm selection translates directly into
superior downstream computer vision performance (keypoint recovery and descriptor
matching against ground-truth clean scenes).

Benchmark Protocol:
1. Computes ORB keypoints and visual descriptors on the ground truth clean reference.
2. Evaluates feature recovery rate on:
   - Raw Degraded Input (Low-Light)
   - Best Fixed Single Method (Retinex)
   - Proposed Random Forest Selector Output
3. Computes Keypoint Repeatability & Match Ratio (%) using cross-checked Hamming matching.
4. Generates publication figure (Fig 6) and summary CSV.
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import argparse
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.dataset import load_and_preprocess_pair, resolve_dataset_dir


def evaluate_downstream_keypoints(
    data_dir: str = "/Users/home/Desktop/LOLdataset",
    split: str = "eval15",
    model_path: str = "models/rf_selector.joblib",
    output_csv: str = "results/downstream_metrics.csv",
    output_fig: str = "results/figures/fig6_downstream_utility.png",
    n_features: int = 1000,
) -> pd.DataFrame:
    """Evaluates downstream keypoint recovery across the test set."""
    data_dir = resolve_dataset_dir(data_dir)
    print("=" * 70)
    print("DOWNSTREAM COMPUTER VISION UTILITY BENCHMARK (Phase 1 Preview)")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run main.py first.")

    model = joblib.load(model_path)
    orb = cv2.ORB_create(nfeatures=n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    low_dir = os.path.join(data_dir, split, "low")
    high_dir = os.path.join(data_dir, split, "high")
    image_files = sorted([f for f in os.listdir(low_dir) if f.endswith(".png")])

    print(f"Evaluating {len(image_files)} test pairs from {split}...")

    results = []

    for fn in image_files:
        low_p = os.path.join(low_dir, fn)
        high_p = os.path.join(high_dir, fn)

        clean, deg, _ = load_and_preprocess_pair(low_p, high_p, target_long_side=512)

        # 1. Clean reference keypoints
        kp_clean, des_clean = orb.detectAndCompute(clean, None)
        n_clean = len(kp_clean)
        if n_clean == 0 or des_clean is None:
            continue

        # 2. Raw Degraded input
        kp_deg, des_deg = orb.detectAndCompute(deg, None)
        m_deg = matcher.match(des_deg, des_clean) if (des_deg is not None and len(des_deg) > 0) else []

        # 3. Best Fixed Method (Retinex)
        ret_img = METHODS["retinex"](deg)
        kp_ret, des_ret = orb.detectAndCompute(ret_img, None)
        m_ret = matcher.match(des_ret, des_clean) if (des_ret is not None and len(des_ret) > 0) else []

        # 4. Proposed Selector Choice
        feats = extract_features(deg, include_engineered=True)
        X = np.array([[feats[k] for k in FEATURE_NAMES]])
        pred_m = model.predict(X)[0]
        sel_img = METHODS[pred_m](deg)
        kp_sel, des_sel = orb.detectAndCompute(sel_img, None)
        m_sel = matcher.match(des_sel, des_clean) if (des_sel is not None and len(des_sel) > 0) else []

        # Ratios
        r_deg = (len(m_deg) / float(n_clean)) * 100.0
        r_ret = (len(m_ret) / float(n_clean)) * 100.0
        r_sel = (len(m_sel) / float(n_clean)) * 100.0

        results.append({
            "Filename": fn,
            "Clean_Keypoints": n_clean,
            "Predicted_Method": pred_m,
            "Degraded_Matches": len(m_deg),
            "Retinex_Matches": len(m_ret),
            "Selector_Matches": len(m_sel),
            "Degraded_Match_Rate_pct": round(r_deg, 2),
            "Retinex_Match_Rate_pct": round(r_ret, 2),
            "Selector_Match_Rate_pct": round(r_sel, 2),
            "Gain_over_Retinex_pct": round(r_sel - r_ret, 2),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved downstream metrics table to {output_csv}")

    # Summary
    mean_deg = df["Degraded_Match_Rate_pct"].mean()
    mean_ret = df["Retinex_Match_Rate_pct"].mean()
    mean_sel = df["Selector_Match_Rate_pct"].mean()
    gain = mean_sel - mean_ret

    print("\n" + "=" * 70)
    print("DOWNSTREAM KEYPOINT MATCHING PERFORMANCE SUMMARY (eval15)")
    print("=" * 70)
    print(f"  • Raw Degraded Input Match Rate    : {mean_deg:.2f}%")
    print(f"  • Best Fixed Single Method (Retinex): {mean_ret:.2f}%")
    print(f"  • Proposed Selector Match Rate     : {mean_sel:.2f}%")
    print(f"  • Net Match Recovery Gain          : +{gain:.2f}% (over static Retinex)")
    print("=" * 70)

    # Plot Visual Strip for Downstream Utility
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Barplot comparison of mean match rate
    models = ["Raw Degraded", "Best Fixed (Retinex)", "Random Forest Selector"]
    rates = [mean_deg, mean_ret, mean_sel]
    colors = ["#94a3b8", "#3b82f6", "#16a34a"]

    bars = axes[0].bar(models, rates, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    axes[0].set_ylabel("ORB Keypoint Match Rate with Ground Truth (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Downstream Visual Feature Recovery Rate", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, max(rates) * 1.25)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Subplot 2: Per-image comparison lines across eval15
    axes[1].plot(range(len(df)), df["Retinex_Match_Rate_pct"], marker="o", label="Best Fixed (Retinex)", color="#3b82f6", linewidth=2.0)
    axes[1].plot(range(len(df)), df["Selector_Match_Rate_pct"], marker="s", label="Proposed Selector", color="#16a34a", linewidth=2.5)
    axes[1].plot(range(len(df)), df["Degraded_Match_Rate_pct"], marker="x", label="Raw Degraded", color="#94a3b8", linestyle="--")
    axes[1].set_xlabel("eval15 Image Index", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Match Rate with Ground Truth (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Per-Image Feature Recovery (eval15 Test Set)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    plt.suptitle("Figure 6: Phase 1 Downstream Computer Vision Utility Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved downstream utility figure to {output_fig}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Downstream CV Utility Benchmark")
    parser.add_argument("--data_dir", type=str, default="data/lol_dataset")
    parser.add_argument("--split", type=str, default="eval15")
    args = parser.parse_args()

    evaluate_downstream_keypoints(data_dir=args.data_dir, split=args.split)


if __name__ == "__main__":
    main()
