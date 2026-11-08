#!/usr/bin/env python3
"""
Downstream Computer Vision & Geometric Evaluation Module
Evaluates the impact of enhancement selection on:
1. SIFT & ORB Feature Matching with RANSAC Inliers
2. Structural Edge Boundary Recovery (Precision, Recall, F1-Score)
3. Shi-Tomasi Corner Repeatability

Outputs to results/downstream_geometric_metrics.csv and results/figures/fig7_downstream_geometric_eval.png
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import resolve_dataset_dir, get_dataset_manifest, load_and_preprocess_pair
from src.features import extract_features, FEATURE_NAMES
from src.methods import METHODS
from src.demo import get_or_train_selector


def evaluate_sift_matching(img_clean: np.ndarray, img_cand: np.ndarray) -> Tuple[int, int, float]:
    """
    Computes SIFT feature matches and RANSAC inliers against the clean reference.
    Returns: (total_matches, inlier_matches, inlier_ratio)
    """
    sift = cv2.SIFT_create(nfeatures=1000)
    kp_clean, des_clean = sift.detectAndCompute(img_clean, None)
    kp_cand, des_cand = sift.detectAndCompute(img_cand, None)

    if des_clean is None or des_cand is None or len(kp_clean) < 4 or len(kp_cand) < 4:
        return 0, 0, 0.0

    # FLANN matcher for SIFT
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des_cand, des_clean, k=2)
    except Exception:
        return 0, 0, 0.0

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return len(good_matches), 0, 0.0

    # RANSAC homography estimation
    src_pts = np.float32([kp_cand[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_clean[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(np.sum(mask)) if mask is not None else 0
    except Exception:
        inliers = 0

    ratio = inliers / max(1, len(good_matches))
    return len(good_matches), inliers, float(ratio)


def evaluate_boundary_fscore(img_clean: np.ndarray, img_cand: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluates Canny boundary precision, recall, and F1-score against the clean reference.
    """
    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    gray_cand = cv2.cvtColor(img_cand, cv2.COLOR_BGR2GRAY)

    edge_clean = cv2.Canny(gray_clean, 50, 150) > 0
    edge_cand = cv2.Canny(gray_cand, 50, 150) > 0

    # Dilate slightly for tolerance (within 2 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_clean_dil = cv2.dilate(edge_clean.astype(np.uint8), kernel) > 0
    edge_cand_dil = cv2.dilate(edge_cand.astype(np.uint8), kernel) > 0

    true_pos_cand = np.sum(edge_cand & edge_clean_dil)
    total_cand = np.sum(edge_cand)
    precision = true_pos_cand / max(1, total_cand)

    true_pos_clean = np.sum(edge_clean & edge_cand_dil)
    total_clean = np.sum(edge_clean)
    recall = true_pos_clean / max(1, total_clean)

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def evaluate_corner_repeatability(img_clean: np.ndarray, img_cand: np.ndarray) -> float:
    """
    Evaluates Shi-Tomasi corner detection repeatability.
    """
    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    gray_cand = cv2.cvtColor(img_cand, cv2.COLOR_BGR2GRAY)

    pts_clean = cv2.goodFeaturesToTrack(gray_clean, maxCorners=500, qualityLevel=0.01, minDistance=10)
    pts_cand = cv2.goodFeaturesToTrack(gray_cand, maxCorners=500, qualityLevel=0.01, minDistance=10)

    if pts_clean is None or pts_cand is None or len(pts_clean) == 0 or len(pts_cand) == 0:
        return 0.0

    pts_clean = pts_clean.reshape(-1, 2)
    pts_cand = pts_cand.reshape(-1, 2)

    # Count how many clean corners have a matching candidate corner within 3 pixels
    repeated = 0
    for pt in pts_clean:
        dists = np.linalg.norm(pts_cand - pt, axis=1)
        if np.min(dists) <= 3.0:
            repeated += 1

    return float(repeated / len(pts_clean))


def run_downstream_benchmark():
    data_dir = resolve_dataset_dir()
    manifest = get_dataset_manifest(data_dir, split_name="eval15")
    print(f"Running Downstream Semantic Benchmark on {len(manifest)} eval15 pairs...")

    model = get_or_train_selector()

    results = []

    for item in manifest:
        fn = item["filename"]
        clean, deg, _ = load_and_preprocess_pair(item["low_path"], item["high_path"], target_long_side=512)

        # 1. Feature extraction and prediction
        feats = extract_features(deg, include_engineered=True)
        X = np.array([[feats[k] for k in FEATURE_NAMES]])
        pred_method = model.predict(X)[0]

        # Candidates to compare
        img_raw = deg
        img_fixed = METHODS["retinex"](deg)
        img_selected = METHODS[pred_method](deg)

        candidates = {
            "Raw Degraded": img_raw,
            "Best Fixed (Retinex)": img_fixed,
            "Proposed RF Selector": img_selected,
        }

        row = {"Filename": fn, "Predicted_Method": pred_method}

        for c_name, c_img in candidates.items():
            # SIFT RANSAC
            matches, inliers, inlier_ratio = evaluate_sift_matching(clean, c_img)
            # Boundary F1
            prec, rec, f1 = evaluate_boundary_fscore(clean, c_img)
            # Corner Repeatability
            corner_rep = evaluate_corner_repeatability(clean, c_img)

            row[f"{c_name}_SIFT_Matches"] = matches
            row[f"{c_name}_SIFT_Inliers"] = inliers
            row[f"{c_name}_Boundary_F1"] = round(f1 * 100.0, 2)
            row[f"{c_name}_Corner_Repeatability"] = round(corner_rep * 100.0, 2)

        results.append(row)

    df_res = pd.DataFrame(results)
    out_csv = "results/downstream_geometric_metrics.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved downstream metrics to {out_csv}")

    # Summary Statistics
    summary = []
    for c_name in ["Raw Degraded", "Best Fixed (Retinex)", "Proposed RF Selector"]:
        m_sift = df_res[f"{c_name}_SIFT_Matches"].mean()
        m_inl = df_res[f"{c_name}_SIFT_Inliers"].mean()
        m_f1 = df_res[f"{c_name}_Boundary_F1"].mean()
        m_rep = df_res[f"{c_name}_Corner_Repeatability"].mean()

        summary.append({
            "Pipeline Stage": c_name,
            "SIFT Matches": round(m_sift, 1),
            "RANSAC Inliers": round(m_inl, 1),
            "Boundary F1-Score (%)": round(m_f1, 2),
            "Corner Repeatability (%)": round(m_rep, 2),
        })

    df_sum = pd.DataFrame(summary)
    print("\n" + "=" * 75)
    print("DOWNSTREAM COMPUTER VISION & GEOMETRIC UTILITY SUMMARY")
    print("=" * 75)
    print(df_sum.to_string(index=False))
    print("=" * 75)

    # Generate Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#94A3B8", "#2563EB", "#0D9488"]

    # 1. SIFT RANSAC Inliers
    sns.barplot(
        x="Pipeline Stage",
        y="RANSAC Inliers",
        data=df_sum,
        palette=palette,
        ax=axes[0],
        hue="Pipeline Stage",
        legend=False
    )
    axes[0].set_title("SIFT Feature Matches with RANSAC Inliers", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Inlier Count", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Boundary F1-Score
    sns.barplot(
        x="Pipeline Stage",
        y="Boundary F1-Score (%)",
        data=df_sum,
        palette=palette,
        ax=axes[1],
        hue="Pipeline Stage",
        legend=False
    )
    axes[1].set_title("Structural Boundary F1-Score (Canny)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("F1-Score (%)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Corner Repeatability
    sns.barplot(
        x="Pipeline Stage",
        y="Corner Repeatability (%)",
        data=df_sum,
        palette=palette,
        ax=axes[2],
        hue="Pipeline Stage",
        legend=False
    )
    axes[2].set_title("Shi-Tomasi Corner Repeatability (≤ 3px)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Repeatability Rate (%)", fontsize=11)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes:
        ax.set_xticklabels(["Raw Degraded", "Best Fixed\n(Retinex)", "Proposed\nRF Selector"], fontsize=10)

    plt.tight_layout()
    out_fig = "results/figures/fig7_downstream_geometric_eval.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved publication figure to {out_fig}")


if __name__ == "__main__":
    run_downstream_benchmark()

