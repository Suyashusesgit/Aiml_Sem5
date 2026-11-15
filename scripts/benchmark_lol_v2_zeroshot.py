#!/usr/bin/env python3
"""
Phase 3: Zero-Shot LOL-v2 Generalization Test
----------------------------------------------
Tests frozen LOL-v1 models (LR, RF, HGB, rf_selector.joblib) on
LOL-v2 Real_captured/Test and Synthetic/Test (100 pairs each).

For each image:
  1. Enhance with ALL 7 methods → compute PSNR vs ground-truth
  2. Extract 10 features from degraded-only image
  3. Each model predicts which of its 5 known classes is best
  4. Compute regret = oracle_psnr - predicted_method_psnr

Also computes the "Best-Fixed" baseline (single method chosen to minimize
average regret across the entire test set).
"""

import os, sys, pickle, time
import numpy as np
import cv2
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES, ENGINEERED_FEATURE_NAMES

# ---------- helpers ----------

METHOD_NAMES = list(METHODS.keys())  # 7 methods

# The 5-class label mapping used during LOL-v1 training (no bilat_unsharp, no invert_dehaze)
LOL_V1_CLASSES = ['clahe', 'gamma', 'hist_eq', 'identity', 'retinex']

from src.labels import compute_psnr


def load_model(path: str):
    """Load a model dict {model, scaler} or a joblib pipeline."""
    if path.endswith('.joblib'):
        pipe = joblib.load(path)
        return pipe, 'pipeline'
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj, 'dict'


def predict_with_model(obj, kind: str, features_arr: np.ndarray, int_to_class: list | None = None) -> str:
    """Return predicted class name (string)."""
    if kind == 'pipeline':
        pred = obj.predict(features_arr.reshape(1, -1))[0]
        return str(pred)
    # dict with {model, scaler}
    scaler = obj['scaler']
    model = obj['model']
    X_scaled = scaler.transform(features_arr.reshape(1, -1))
    pred = model.predict(X_scaled)[0]
    # If prediction is integer, map to class name
    if isinstance(pred, (int, np.integer)):
        if int_to_class is not None:
            return int_to_class[int(pred)]
        return LOL_V1_CLASSES[int(pred)]
    return str(pred)


def get_lol_v2_pairs(split: str) -> list[dict]:
    """Get list of (low_path, high_path, filename) for a LOL-v2 split."""
    base = os.path.join("data", "LOL-v2")
    if split == "real":
        low_dir = os.path.join(base, "Real_captured", "Test", "Low")
        high_dir = os.path.join(base, "Real_captured", "Test", "Normal")
    elif split == "synthetic":
        low_dir = os.path.join(base, "Synthetic", "Test", "Low")
        high_dir = os.path.join(base, "Synthetic", "Test", "Normal")
    else:
        raise ValueError(f"Unknown split: {split}")

    pairs = []
    for fname in sorted(os.listdir(low_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        low_path = os.path.join(low_dir, fname)
        # Determine matching high path
        if split == "real":
            high_fname = fname.replace("low", "normal")
            high_path = os.path.join(high_dir, high_fname)
        else:  # synthetic: same filename in Normal dir
            high_path = os.path.join(high_dir, fname)

        if os.path.isfile(high_path):
            pairs.append({"low": low_path, "high": high_path, "name": fname})
        else:
            print(f"  WARNING: No matching normal for {fname}")
    return pairs


def evaluate_split(pairs: list[dict], models: dict, all_features: list[str]) -> dict:
    """
    Evaluate all models on one LOL-v2 split.
    Returns per-model metrics + best-fixed baseline.
    """
    n = len(pairs)
    # psnr_matrix[i, j] = PSNR of method j on image i
    psnr_matrix = np.zeros((n, len(METHOD_NAMES)))
    feature_matrix = np.zeros((n, len(all_features)))
    oracle_psnr = np.zeros(n)
    oracle_label = []

    print(f"  Processing {n} pairs...")
    t0 = time.time()
    for i, pair in enumerate(pairs):
        low_bgr = cv2.imread(pair["low"])
        high_bgr = cv2.imread(pair["high"])

        if low_bgr is None or high_bgr is None:
            print(f"  ERROR reading pair {pair['name']}, skipping")
            continue

        # Resize to match if needed
        if low_bgr.shape[:2] != high_bgr.shape[:2]:
            high_bgr = cv2.resize(high_bgr, (low_bgr.shape[1], low_bgr.shape[0]))

        # Convert to RGB for PSNR (SOP: BGR→RGB before metrics)
        low_rgb = cv2.cvtColor(low_bgr, cv2.COLOR_BGR2RGB)
        high_rgb = cv2.cvtColor(high_bgr, cv2.COLOR_BGR2RGB)

        # Enhance with all 7 methods and compute PSNR
        for j, mname in enumerate(METHOD_NAMES):
            enhanced_bgr = METHODS[mname](low_bgr)
            enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            # Resize enhanced to match reference
            if enhanced_rgb.shape[:2] != high_rgb.shape[:2]:
                enhanced_rgb = cv2.resize(enhanced_rgb, (high_rgb.shape[1], high_rgb.shape[0]))
            psnr_matrix[i, j] = compute_psnr(high_rgb, enhanced_rgb, max_val=255.0)

        # Oracle
        oracle_idx = np.argmax(psnr_matrix[i])
        oracle_psnr[i] = psnr_matrix[i, oracle_idx]
        oracle_label.append(METHOD_NAMES[oracle_idx])

        # Features (from degraded BGR only — no label leakage)
        feat_dict = extract_features(low_bgr, include_engineered=True)
        for k, fname in enumerate(all_features):
            feature_matrix[i, k] = feat_dict.get(fname, 0.0)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n} done ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  All {n} pairs processed in {elapsed:.1f}s")

    # Oracle distribution
    from collections import Counter
    oracle_dist = Counter(oracle_label)
    print(f"  Oracle distribution: {dict(oracle_dist)}")

    # Evaluate each model
    results = {}
    for model_name, (obj, kind) in models.items():
        correct = 0
        regrets = []

        # Determine feature count model expects
        if kind == 'pipeline':
            # rf_selector.joblib was trained with 8 base features only
            n_features_model = 8
        else:
            scaler = obj['scaler']
            n_features_model = scaler.n_features_in_

        for i in range(n):
            feat_vec = feature_matrix[i, :n_features_model]
            try:
                pred_class = predict_with_model(obj, kind, feat_vec,
                    int_to_class=LOL_V1_CLASSES if kind == 'dict' else None)
            except Exception as e:
                pred_class = 'identity'  # fallback

            # Map predicted class to index in METHOD_NAMES
            if pred_class in METHOD_NAMES:
                pred_idx = METHOD_NAMES.index(pred_class)
            else:
                pred_idx = METHOD_NAMES.index('identity')

            pred_psnr = psnr_matrix[i, pred_idx]
            regret = oracle_psnr[i] - pred_psnr
            regrets.append(regret)

            if oracle_label[i] == pred_class:
                correct += 1

        top1 = correct / n * 100
        mean_regret = np.mean(regrets)
        max_regret = np.max(regrets)
        median_regret = np.median(regrets)

        results[model_name] = {
            'top1_acc': top1,
            'mean_regret_db': mean_regret,
            'max_regret_db': max_regret,
            'median_regret_db': median_regret,
        }

    # Best-Fixed baseline: pick the single method with lowest average regret
    for j, mname in enumerate(METHOD_NAMES):
        fixed_regrets = oracle_psnr - psnr_matrix[:, j]
        mean_r = np.mean(fixed_regrets)
        correct = sum(1 for i in range(n) if oracle_label[i] == mname)
        results[f'Best-Fixed ({mname})'] = {
            'top1_acc': correct / n * 100,
            'mean_regret_db': mean_r,
            'max_regret_db': np.max(fixed_regrets),
            'median_regret_db': np.median(fixed_regrets),
        }

    return results


def main():
    all_features = FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
    print(f"Features ({len(all_features)}): {all_features}")
    print(f"Methods ({len(METHOD_NAMES)}): {METHOD_NAMES}")

    # Load models
    models = {}
    model_files = {
        'Logistic Regression': 'results/models/logistic_regression.pkl',
        'Random Forest': 'results/models/random_forest.pkl',
        'HistGradientBoosting': 'results/models/hist_gradient_boosting.pkl',
        'RF Pipeline (joblib)': 'models/rf_selector.joblib',
    }
    for name, path in model_files.items():
        if os.path.isfile(path):
            obj, kind = load_model(path)
            models[name] = (obj, kind)
            print(f"  Loaded: {name} ({kind}) from {path}")
        else:
            print(f"  SKIP: {path} not found")

    if not models:
        print("ERROR: No models found. Aborting.")
        return

    # Evaluate on both splits
    for split_name in ['synthetic', 'real']:
        print(f"\n{'='*60}")
        print(f"LOL-v2 {split_name.upper()} — Zero-Shot Evaluation")
        print(f"{'='*60}")
        pairs = get_lol_v2_pairs(split_name)
        print(f"  Found {len(pairs)} pairs")

        results = evaluate_split(pairs, models, all_features)

        # Print results table
        print(f"\n--- LOL-v2 {split_name.upper()} Results ---")
        print(f"{'Model':<30} {'Top-1 %':>8} {'Mean Regret':>12} {'Median':>8} {'Max':>8}")
        print("-" * 70)

        # Sort: ML models first, then Best-Fixed sorted by mean regret
        ml_keys = [k for k in results if not k.startswith('Best-Fixed')]
        fixed_keys = sorted(
            [k for k in results if k.startswith('Best-Fixed')],
            key=lambda k: results[k]['mean_regret_db']
        )

        for k in sorted(ml_keys, key=lambda k: results[k]['mean_regret_db']):
            r = results[k]
            print(f"{k:<30} {r['top1_acc']:>7.1f}% {r['mean_regret_db']:>11.3f} {r['median_regret_db']:>7.3f} {r['max_regret_db']:>7.3f}")

        print("-" * 70)
        # Show top 3 fixed baselines
        for k in fixed_keys[:3]:
            r = results[k]
            print(f"{k:<30} {r['top1_acc']:>7.1f}% {r['mean_regret_db']:>11.3f} {r['median_regret_db']:>7.3f} {r['max_regret_db']:>7.3f}")
        print()


if __name__ == "__main__":
    main()
