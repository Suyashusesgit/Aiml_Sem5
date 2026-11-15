#!/usr/bin/env python3
"""
Benchmark: 8 Multi-Hazard Enhancers + 10 Physical Descriptors
-------------------------------------------------------------
Evaluates the full 8-enhancer portfolio and 10-descriptor feature space:
1. Enhancers (8):
   - identity
   - gamma
   - clahe
   - hist_eq
   - retinex
   - multi_scale_retinex
   - guided_filter
   - bilat_unsharp

2. Descriptors (10):
   - brightness
   - contrast
   - entropy
   - noise
   - dark_channel
   - air_light_transmission
   - dynamic_range_ratio
   - edge_mean
   - edge_p90
   - colourfulness

Datasets evaluated:
- LOL-v2 Real_captured (100 Test pairs)
- LOL-v2 Synthetic (100 Test pairs)
"""

import os
import sys
import time
from typing import Dict, List, Any
import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import ENHANCERS_8
from src.features import extract_features, PHYSICAL_10_DESCRIPTORS
from src.labels import compute_psnr

ENHANCER_NAMES = list(ENHANCERS_8.keys())


def load_pairs(dataset: str) -> List[Dict[str, str]]:
    base = os.path.join("data", "LOL-v2")
    if dataset == "real":
        low_dir = os.path.join(base, "Real_captured", "Test", "Low")
        high_dir = os.path.join(base, "Real_captured", "Test", "Normal")
    else:
        low_dir = os.path.join(base, "Synthetic", "Test", "Low")
        high_dir = os.path.join(base, "Synthetic", "Test", "Normal")

    pairs = []
    for fname in sorted(os.listdir(low_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        low_path = os.path.join(low_dir, fname)
        if dataset == "real":
            high_path = os.path.join(high_dir, fname.replace("low", "normal"))
        else:
            high_path = os.path.join(high_dir, fname)
        if os.path.isfile(high_path):
            pairs.append({"low": low_path, "high": high_path, "name": fname})
    return pairs


def benchmark_dataset(dataset_name: str) -> Dict[str, Any]:
    pairs = load_pairs(dataset_name)
    n = len(pairs)
    print(f"\n========================================================")
    print(f"LOL-v2 {dataset_name.upper()} — 8 Enhancers & 10 Descriptors ({n} pairs)")
    print(f"========================================================")

    psnr_matrix = np.zeros((n, len(ENHANCER_NAMES)))
    feature_matrix = np.zeros((n, len(PHYSICAL_10_DESCRIPTORS)))
    oracle_psnr = np.zeros(n)
    oracle_methods = []

    t0 = time.time()
    for i, pair in enumerate(pairs):
        low_bgr = cv2.imread(pair["low"])
        high_bgr = cv2.imread(pair["high"])

        if low_bgr.shape[:2] != high_bgr.shape[:2]:
            high_bgr = cv2.resize(high_bgr, (low_bgr.shape[1], low_bgr.shape[0]))

        high_rgb = cv2.cvtColor(high_bgr, cv2.COLOR_BGR2RGB)

        # 1. Evaluate all 8 enhancers
        for j, mname in enumerate(ENHANCER_NAMES):
            enh_bgr = ENHANCERS_8[mname](low_bgr)
            enh_rgb = cv2.cvtColor(enh_bgr, cv2.COLOR_BGR2RGB)
            if enh_rgb.shape[:2] != high_rgb.shape[:2]:
                enh_rgb = cv2.resize(enh_rgb, (high_rgb.shape[1], high_rgb.shape[0]))
            psnr_matrix[i, j] = compute_psnr(high_rgb, enh_rgb, max_val=255.0)

        # Oracle
        best_j = int(np.argmax(psnr_matrix[i]))
        oracle_psnr[i] = psnr_matrix[i, best_j]
        oracle_methods.append(ENHANCER_NAMES[best_j])

        # 2. Extract 10 physical descriptors from degraded only
        feats = extract_features(low_bgr, include_engineered=False, include_all_physical=True)
        for k, fname in enumerate(PHYSICAL_10_DESCRIPTORS):
            feature_matrix[i, k] = feats[fname]

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{n} images in {time.time()-t0:.1f}s...")

    print(f"  Feature & PSNR matrix computation completed in {time.time()-t0:.1f}s.")

    # Oracle distribution
    from collections import Counter
    dist = Counter(oracle_methods)
    print(f"\n  Oracle Portfolio Distribution (8 Enhancers):")
    for m, c in dist.most_common():
        print(f"    {m:<22}: {c:>3} images ({c/n*100:4.1f}%)")

    # Train & Evaluate Classifiers with 5-Fold Cross-Validation on this dataset
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y = np.array(oracle_methods)
    X = feature_matrix

    models = {
        "Multinomial Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        "Random Forest Classifier": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        "HistGradientBoosting": make_pipeline(StandardScaler(), HistGradientBoostingClassifier(max_iter=100, random_state=42)),
    }

    results = {}
    for mname, model in models.items():
        preds = np.empty(n, dtype=object)
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            preds[val_idx] = model.predict(X[val_idx])

        # Compute regrets
        regrets = []
        correct = 0
        for i in range(n):
            chosen = preds[i]
            chosen_j = ENHANCER_NAMES.index(chosen)
            psnr_chosen = psnr_matrix[i, chosen_j]
            r = oracle_psnr[i] - psnr_chosen
            regrets.append(r)
            if chosen == oracle_methods[i]:
                correct += 1

        results[mname] = {
            "top1": correct / n * 100.0,
            "mean_regret": float(np.mean(regrets)),
            "median_regret": float(np.median(regrets)),
            "max_regret": float(np.max(regrets)),
            "psnr_mean": float(np.mean(oracle_psnr - np.array(regrets))),
        }

    # Best-Fixed Baselines (single fixed enhancer)
    fixed_results = {}
    for j, mname in enumerate(ENHANCER_NAMES):
        fixed_regrets = oracle_psnr - psnr_matrix[:, j]
        correct = sum(1 for i in range(n) if oracle_methods[i] == mname)
        fixed_results[f"Best-Fixed ({mname})"] = {
            "top1": correct / n * 100.0,
            "mean_regret": float(np.mean(fixed_regrets)),
            "median_regret": float(np.median(fixed_regrets)),
            "max_regret": float(np.max(fixed_regrets)),
            "psnr_mean": float(np.mean(psnr_matrix[:, j])),
        }

    # Print summary table
    print(f"\n--- Benchmark Results: LOL-v2 {dataset_name.upper()} (8 Enhancers, 10 Descriptors) ---")
    print(f"{'Method / Model':<35} {'Top-1 %':>8} {'Mean Regret':>12} {'Median':>8} {'Max':>8} {'Mean PSNR':>10}")
    print("-" * 85)

    for k in sorted(results.keys(), key=lambda x: results[x]["mean_regret"]):
        r = results[k]
        print(f"{k:<35} {r['top1']:>7.1f}% {r['mean_regret']:>11.3f} {r['median_regret']:>7.3f} {r['max_regret']:>7.3f} {r['psnr_mean']:>9.2f}")

    print("-" * 85)
    # Top 3 fixed methods
    for k in sorted(fixed_results.keys(), key=lambda x: fixed_results[x]["mean_regret"])[:3]:
        r = fixed_results[k]
        print(f"{k:<35} {r['top1']:>7.1f}% {r['mean_regret']:>11.3f} {r['median_regret']:>7.3f} {r['max_regret']:>7.3f} {r['psnr_mean']:>9.2f}")

    return {"ml": results, "fixed": fixed_results, "dist": dict(dist), "oracle_psnr": float(np.mean(oracle_psnr))}


def main():
    print(f"Registered 8 Enhancers: {ENHANCER_NAMES}")
    print(f"Registered 10 Descriptors: {PHYSICAL_10_DESCRIPTORS}")

    real_res = benchmark_dataset("real")
    syn_res = benchmark_dataset("synthetic")

    print("\n" + "="*85)
    print("EXECUTIVE SUMMARY: 8 MULTI-HAZARD ENHANCERS + 10 PHYSICAL DESCRIPTORS")
    print("="*85)
    print(f"{'Dataset':<15} {'Best Model':<32} {'Top-1':>8} {'Mean Regret':>12} {'Best Fixed':<20} {'Regret Saving':>14}")
    print("-" * 105)

    for ds_name, res in [("LOL-v2 Real", real_res), ("LOL-v2 Synthetic", syn_res)]:
        best_ml = min(res["ml"].keys(), key=lambda k: res["ml"][k]["mean_regret"])
        best_fixed = min(res["fixed"].keys(), key=lambda k: res["fixed"][k]["mean_regret"])
        ml_reg = res["ml"][best_ml]["mean_regret"]
        fixed_reg = res["fixed"][best_fixed]["mean_regret"]
        saving = fixed_reg - ml_reg
        print(f"{ds_name:<15} {best_ml:<32} {res['ml'][best_ml]['top1']:>7.1f}% {ml_reg:>11.3f} dB {best_fixed:<20} {f'+{saving:.3f} dB':>14}")
    print()


if __name__ == "__main__":
    main()
