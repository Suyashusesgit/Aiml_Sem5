#!/usr/bin/env python3
"""
LOL-v2 Comprehensive Training and Evaluation Pipeline
Trains and evaluates multiple algorithm selectors on:
1. LOL-v2 Real_captured (689 Train pairs, 100 Test pairs) - Sensor noise stress test
2. LOL-v2 Synthetic (900 Train pairs, 100 Test pairs) - Pure photon starvation check

Models Evaluated:
- Logistic Regression (Multinomial with L2 regularization)
- Random Forest (Class-balanced, tuned depth)
- Extra Trees (Extremely randomized tree ensemble)
- HistGradientBoosting (Histogram-based gradient boosted trees)
- XGBoost Classifier (Extreme gradient boosting)
- Continuous Predict-Then-Optimize (6-Head / 7-Head Ridge Regressors)
- Best-Fixed Single Baseline (Oracle-determined optimal static filter)
- Theoretical Oracle Upper Bound
"""

import os
import sys
import time
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.base import clone
import xgboost as xgb
import joblib

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.labels import score_pair
from src.dataset_v2 import load_and_preprocess_pair_v2

CANONICAL_METHODS = list(METHODS.keys())


class XGBWrapper:
    """Wrapper for XGBoost to handle label encoding/decoding seamlessly."""
    def __init__(self, model, le):
        self.model = model
        self.le = le

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))


def load_dataset_manifest(dataset_type: str, split: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (low_path, norm_path, filename) for LOL-v2 split.
    dataset_type: 'Real_captured' or 'Synthetic'
    split: 'Train' or 'Test'
    """
    base = os.path.join("data", "LOL-v2", dataset_type, split)
    low_dir = os.path.join(base, "Low")
    norm_dir = os.path.join(base, "Normal")

    low_files = sorted(os.listdir(low_dir))
    manifest = []
    for f in low_files:
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        low_path = os.path.join(low_dir, f)
        if dataset_type == "Real_captured":
            norm_name = f.replace("low", "normal")
        else:
            norm_name = f
        norm_path = os.path.join(norm_dir, norm_name)
        if os.path.exists(norm_path):
            manifest.append((low_path, norm_path, f))
    return manifest


def extract_features_and_scores_batch(
    manifest: List[Tuple[str, str, str]],
    dataset_label: str,
    cache_prefix: str,
    max_workers: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts 10D physical features and computes all method PSNR/SSIM scores in parallel.
    Uses disk caching to avoid redundant computation.
    """
    feat_cache = f"results/{cache_prefix}_features.csv"
    score_cache = f"results/{cache_prefix}_scores.csv"

    if os.path.exists(feat_cache) and os.path.exists(score_cache):
        print(f"[{dataset_label}] Loading cached features and scores from {feat_cache}...")
        df_feat = pd.read_csv(feat_cache)
        df_score = pd.read_csv(score_cache)
        return df_feat, df_score

    print(f"[{dataset_label}] Extracting features and scoring {len(manifest)} pairs across {len(CANONICAL_METHODS)} methods...")
    t0 = time.time()

    def process_item(item):
        low_p, norm_p, fname = item
        clean, deg, _ = load_and_preprocess_pair_v2(low_p, norm_p, target_long_side=512)
        feats = extract_features(deg, include_engineered=True)
        feats["filename"] = fname
        scores = score_pair(clean, deg, tie_threshold_db=0.05)
        scores["filename"] = fname
        return feats, scores

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(process_item, manifest))

    dt = time.time() - t0
    print(f"[{dataset_label}] Finished {len(manifest)} pairs in {dt:.1f}s ({dt/len(manifest)*1000:.1f} ms/pair).")

    feats_list = [r[0] for r in results]
    scores_list = [r[1] for r in results]

    df_feat = pd.DataFrame(feats_list)
    df_score = pd.DataFrame(scores_list)

    os.makedirs("results", exist_ok=True)
    df_feat.to_csv(feat_cache, index=False)
    df_score.to_csv(score_cache, index=False)
    print(f"[{dataset_label}] Cached to {feat_cache} and {score_cache}.")

    return df_feat, df_score


def evaluate_selector_on_test(
    model: Any,
    scaler: StandardScaler,
    X_test: np.ndarray,
    df_test_scores: pd.DataFrame,
    is_pto: bool = False
) -> Dict[str, float]:
    """
    Evaluates a trained model on test data, computing Regret, Accuracy, PSNR, SSIM, and Latency.
    """
    t0 = time.time()
    if is_pto:
        # Predict PSNR with each head, then argmax
        pred_matrix = np.zeros((X_test.shape[0], len(CANONICAL_METHODS)))
        for idx, m in enumerate(CANONICAL_METHODS):
            pred_matrix[:, idx] = model[m].predict(X_test)
        chosen_indices = np.argmax(pred_matrix, axis=1)
        chosen_methods = [CANONICAL_METHODS[i] for i in chosen_indices]
    else:
        # Standard classifier
        chosen_methods = model.predict(X_test)

    inf_time_ms = (time.time() - t0) * 1000 / len(X_test)

    # Compute metrics against oracle
    regrets = []
    chosen_psnrs = []
    chosen_ssims = []
    oracle_psnrs = df_test_scores["oracle_psnr"].values
    oracle_labels = df_test_scores["label"].values

    correct_top1 = 0
    for i, m_chosen in enumerate(chosen_methods):
        o_label = oracle_labels[i]
        o_psnr = oracle_psnrs[i]
        c_psnr = df_test_scores.loc[i, f"psnr_{m_chosen}"]
        c_ssim = df_test_scores.loc[i, f"ssim_{m_chosen}"]

        if m_chosen == o_label:
            correct_top1 += 1

        regret = max(0.0, o_psnr - c_psnr)
        regrets.append(regret)
        chosen_psnrs.append(c_psnr)
        chosen_ssims.append(c_ssim)

    return {
        "top1_acc": correct_top1 / len(X_test) * 100.0,
        "mean_regret_db": float(np.mean(regrets)),
        "std_regret_db": float(np.std(regrets)),
        "max_regret_db": float(np.max(regrets)),
        "mean_psnr_db": float(np.mean(chosen_psnrs)),
        "mean_ssim": float(np.mean(chosen_ssims)),
        "latency_ms": inf_time_ms
    }


def run_full_evaluation(dataset_type: str = "Real_captured"):
    print("\n" + "=" * 80)
    print(f"RUNNING COMPREHENSIVE TRAINING & EVALUATION ON LOL-v2: [{dataset_type}]")
    print("=" * 80)

    train_manifest = load_dataset_manifest(dataset_type, "Train")
    test_manifest = load_dataset_manifest(dataset_type, "Test")

    print(f"Loaded manifest: {len(train_manifest)} Train pairs, {len(test_manifest)} Test pairs.")

    cache_prefix = f"lol_v2_{dataset_type.lower()}"
    df_train_feat, df_train_scores = extract_features_and_scores_batch(
        train_manifest, f"{dataset_type}-Train", f"{cache_prefix}_train"
    )
    df_test_feat, df_test_scores = extract_features_and_scores_batch(
        test_manifest, f"{dataset_type}-Test", f"{cache_prefix}_test"
    )

    feature_cols = [c for c in df_train_feat.columns if c != "filename"]
    print(f"Using {len(feature_cols)} physical features: {feature_cols}")

    X_train_raw = df_train_feat[feature_cols].values.astype(np.float32)
    y_train = df_train_scores["label"].values
    X_test_raw = df_test_feat[feature_cols].values.astype(np.float32)

    # Scale features inside train split
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Class distribution
    print("\nTrain Oracle Class Distribution:")
    for m in CANONICAL_METHODS:
        count = int(np.sum(y_train == m))
        pct = count / len(y_train) * 100
        print(f"  {m:15s}: {count:3d} ({pct:5.1f}%)")

    # Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Random Forest (Tuned)": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, min_samples_split=4, class_weight="balanced", random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42, eval_metric="mlogloss"
        ),
    }

    # 1. 5-Fold Cross Validation on Train Set
    print("\nRunning 5-Fold Cross-Validation on Train Set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_regrets = {name: [] for name in models}
    cv_accs = {name: [] for name in models}

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[trn_idx], X_train[val_idx]
        y_tr, y_val = y_train[trn_idx], y_train[val_idx]

        for name, clf in models.items():
            model_fold = clone(clf)
            if name == "XGBoost":
                fold_le = LabelEncoder()
                y_tr_int = fold_le.fit_transform(y_tr)
                model_fold.fit(X_tr, y_tr_int)
                preds_int = model_fold.predict(X_val)
                preds = fold_le.inverse_transform(preds_int)
            else:
                model_fold.fit(X_tr, y_tr)
                preds = model_fold.predict(X_val)

            # Measure CV regret
            fold_regrets = []
            for i_val, m_pred in enumerate(preds):
                orig_idx = val_idx[i_val]
                o_psnr = df_train_scores.loc[orig_idx, "oracle_psnr"]
                c_psnr = df_train_scores.loc[orig_idx, f"psnr_{m_pred}"]
                fold_regrets.append(max(0.0, o_psnr - c_psnr))

            acc = np.mean(preds == y_val) * 100.0
            cv_regrets[name].append(np.mean(fold_regrets))
            cv_accs[name].append(acc)

    print("\n5-Fold CV Performance Summary (Train Set):")
    for name in models:
        print(f"  {name:25s} | CV Accuracy: {np.mean(cv_accs[name]):5.2f}% | CV Regret: {np.mean(cv_regrets[name]):.3f} dB")

    # 2. Train Full Models on Train Set and Evaluate on Held-Out Test Set
    print("\nTraining on full Train set and evaluating on Official Held-Out Test Set (100 pairs)...")
    results_list = []

    trained_models = {}
    for name, clf in models.items():
        t0 = time.time()
        if name == "XGBoost":
            full_le = LabelEncoder()
            y_train_int = full_le.fit_transform(y_train)
            clf.fit(X_train, y_train_int)
            wrapped = XGBWrapper(clf, full_le)
            trained_models[name] = wrapped
        else:
            clf.fit(X_train, y_train)
            trained_models[name] = clf
        train_time = time.time() - t0

        metrics = evaluate_selector_on_test(
            trained_models[name], scaler, X_test, df_test_scores, is_pto=False
        )
        metrics["model"] = name
        metrics["train_time_s"] = train_time
        metrics["cv_regret_db"] = float(np.mean(cv_regrets[name]))
        results_list.append(metrics)

    # 3. Train Continuous Predict-Then-Optimize (PTO) Ridge Regressors
    print("Training Continuous Predict-Then-Optimize (6-Head Ridge Regressors)...")
    t0 = time.time()
    pto_heads = {}
    for m in CANONICAL_METHODS:
        r = Ridge(alpha=1.0)
        r.fit(X_train, df_train_scores[f"psnr_{m}"].values)
        pto_heads[m] = r
    pto_time = time.time() - t0

    metrics_pto = evaluate_selector_on_test(
        pto_heads, scaler, X_test, df_test_scores, is_pto=True
    )
    metrics_pto["model"] = "Continuous PTO (Ridge Regressors)"
    metrics_pto["train_time_s"] = pto_time
    metrics_pto["cv_regret_db"] = np.nan
    results_list.append(metrics_pto)

    # 4. Single Static Filter Baselines (Best-Fixed)
    oracle_test_psnr = float(np.mean(df_test_scores["oracle_psnr"].values))
    oracle_test_ssim = float(np.mean(df_test_scores["oracle_ssim"].values))

    best_fixed_method = None
    best_fixed_regret = float("inf")
    best_fixed_psnr = 0.0
    best_fixed_ssim = 0.0

    print("\nStatic Baseline Performance on Test Set:")
    for m in CANONICAL_METHODS:
        psnrs = df_test_scores[f"psnr_{m}"].values
        ssims = df_test_scores[f"ssim_{m}"].values
        regrets = np.maximum(0.0, df_test_scores["oracle_psnr"].values - psnrs)
        m_reg = float(np.mean(regrets))
        m_psnr = float(np.mean(psnrs))
        m_ssim = float(np.mean(ssims))
        acc = float(np.mean(df_test_scores["label"].values == m) * 100.0)

        print(f"  Fixed {m:15s} | Regret: {m_reg:6.3f} dB | Mean PSNR: {m_psnr:5.2f} dB | SSIM: {m_ssim:.4f} | Win Rate: {acc:4.1f}%")

        if m_reg < best_fixed_regret:
            best_fixed_regret = m_reg
            best_fixed_method = m
            best_fixed_psnr = m_psnr
            best_fixed_ssim = m_ssim

    results_list.append({
        "model": f"Best-Fixed Single Filter ({best_fixed_method})",
        "top1_acc": float(np.mean(df_test_scores["label"].values == best_fixed_method) * 100.0),
        "mean_regret_db": best_fixed_regret,
        "std_regret_db": float(np.std(np.maximum(0.0, df_test_scores["oracle_psnr"].values - df_test_scores[f"psnr_{best_fixed_method}"].values))),
        "max_regret_db": float(np.max(df_test_scores["oracle_psnr"].values - df_test_scores[f"psnr_{best_fixed_method}"].values)),
        "mean_psnr_db": best_fixed_psnr,
        "mean_ssim": best_fixed_ssim,
        "latency_ms": 0.0,
        "train_time_s": 0.0,
        "cv_regret_db": np.nan
    })

    # 5. Theoretical Oracle
    results_list.append({
        "model": "Theoretical Oracle Upper Bound",
        "top1_acc": 100.0,
        "mean_regret_db": 0.0,
        "std_regret_db": 0.0,
        "max_regret_db": 0.0,
        "mean_psnr_db": oracle_test_psnr,
        "mean_ssim": oracle_test_ssim,
        "latency_ms": 0.0,
        "train_time_s": 0.0,
        "cv_regret_db": 0.0
    })

    df_results = pd.DataFrame(results_list)
    out_csv = f"results/lol_v2_{dataset_type.lower()}_benchmark_comparison.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")

    # Display comparison table sorted by Mean Regret
    df_sorted = df_results.sort_values(by="mean_regret_db", ascending=True).reset_index(drop=True)
    print("\n" + "=" * 95)
    print(f"OFFICIAL TEST SET RESULTS: LOL-v2 [{dataset_type}] (Sorted by Lowest Regret)")
    print("=" * 95)
    header = f"{'Model':<35} | {'Acc (%)':>8} | {'Regret (dB)':>12} | {'Max Reg (dB)':>12} | {'PSNR (dB)':>10} | {'SSIM':>7} | {'Latency':>8}"
    print(header)
    print("-" * 95)
    for _, row in df_sorted.iterrows():
        print(f"{row['model']:<35} | {row['top1_acc']:8.1f} | {row['mean_regret_db']:12.3f} | {row['max_regret_db']:12.3f} | {row['mean_psnr_db']:10.2f} | {row['mean_ssim']:7.4f} | {row['latency_ms']:6.2f}ms")
    print("=" * 95)

    # Save best model to disk
    best_row = df_sorted[~df_sorted["model"].str.contains("Oracle")].iloc[0]
    best_name = best_row["model"]
    print(f"\n>>> BEST MODEL for LOL-v2 [{dataset_type}]: {best_name} <<<")
    print(f"    Mean Regret: {best_row['mean_regret_db']:.3f} dB (vs Best-Fixed: {best_fixed_regret:.3f} dB)")
    print(f"    Regret Reduction over Best-Fixed: {(best_fixed_regret - best_row['mean_regret_db'])/best_fixed_regret*100:.1f}%")
    print(f"    Output PSNR: {best_row['mean_psnr_db']:.2f} dB (Oracle: {oracle_test_psnr:.2f} dB)")

    os.makedirs("models/lol_v2", exist_ok=True)
    best_path = f"models/lol_v2/best_model_{dataset_type.lower()}.pkl"
    if "PTO" in best_name:
        joblib.dump({"heads": pto_heads, "scaler": scaler, "methods": CANONICAL_METHODS}, best_path)
    elif best_name in trained_models:
        joblib.dump({"model": trained_models[best_name], "scaler": scaler, "methods": CANONICAL_METHODS}, best_path)
    else:
        joblib.dump({"model": "fixed", "fixed_method": best_fixed_method, "scaler": scaler, "methods": CANONICAL_METHODS}, best_path)
    print(f"Saved best model package to {best_path}")

    return df_results


if __name__ == "__main__":
    # Run on Real_captured first (the real-world sensor stress test)
    run_full_evaluation("Real_captured")

    # Run on Synthetic (the photon-starvation Poisson benchmark)
    run_full_evaluation("Synthetic")
