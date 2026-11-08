"""
Training and Cross-Validation Pipeline
Implements the experimental protocol specified in Chapters 16, 17, 18, and 22:
- GroupKFold 5-fold cross-validation by scene ID (prevents duplicate leakage)
- StandardScaler strictly fitted on training fold inside make_pipeline (prevents preprocessing leakage)
- Feature extraction strictly on degraded input (prevents label leakage)
- Baselines: Random Choice, Best Fixed Method, Hand-written Heuristic Rule, Oracle
- Classifiers: Multinomial Logistic Regression, Random Forest, XGBoost
- Multi-seed variance reporting (5 folds x 3 seeds = 15 runs)
- Full feature ablations and error analysis
"""

import os
import random
import yaml
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
import xgboost as xgb

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES, ENGINEERED_FEATURE_NAMES
from src.labels import score_pair
from src.dataset import get_dataset_manifest, load_and_preprocess_pair


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and environment for bit-reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_or_load_dataset_cache(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts features and scores all image pairs once, caching them to CSV
    to form the permanent audit trail (Chapter 22.4).
    """
    scores_path = config["paths"]["scores_cache"]
    features_path = config["paths"]["features_cache"]
    data_dir = config["data"]["data_dir"]
    train_split = config["data"]["train_split"]
    target_size = config.get("target_long_side", 512)
    tie_thresh = config.get("tie_threshold_db", 0.05)

    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    if os.path.exists(scores_path) and os.path.exists(features_path):
        print(f"Loading cached dataset from {scores_path} and {features_path}...")
        df_scores = pd.read_csv(scores_path)
        df_features = pd.read_csv(features_path)
        return df_scores, df_features

    print("Building dataset cache: preprocessing, feature extraction, and candidate scoring...")
    manifest = get_dataset_manifest(data_dir, split_name=train_split)
    print(f"Found {len(manifest)} image pairs in {train_split}.")

    scores_rows = []
    features_rows = []

    for i, item in enumerate(manifest):
        fn = item["filename"]
        clean, deg, scale = load_and_preprocess_pair(
            item["low_path"], item["high_path"], target_long_side=target_size
        )

        # 1. Handcrafted features strictly on degraded image
        feats = extract_features(deg, include_engineered=True)
        feats["filename"] = fn
        feats["scene_id"] = item["scene_id"]
        features_rows.append(feats)

        # 2. Candidate scores and oracle ground truth
        scores = score_pair(clean, deg, tie_threshold_db=tie_thresh)
        scores["filename"] = fn
        scores["scene_id"] = item["scene_id"]
        scores_rows.append(scores)

        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            print(f"  Processed {i + 1}/{len(manifest)} pairs...")

    df_scores = pd.DataFrame(scores_rows)
    df_features = pd.DataFrame(features_rows)

    df_scores.to_csv(scores_path, index=False)
    df_features.to_csv(features_path, index=False)
    print(f"Saved cache to {scores_path} and {features_path}.")

    return df_scores, df_features


def evaluate_hand_written_rule(feats: pd.Series) -> str:
    """
    Principled heuristic baseline representing classical domain engineering (Chapter 14.3):
    - Dark and noisy -> Bilateral + Unsharp
    - Severe underexposure -> Gamma (0.5)
    - Low local contrast -> CLAHE
    - Very low entropy -> Hist Equalization
    - Non-uniform lighting / moderate underexposure -> Retinex
    - Already adequate -> Identity
    """
    b = feats["brightness"]
    c = feats["contrast"]
    h = feats["entropy"]
    n = feats["noise"]

    if b > 140 and c > 50 and n < 3.0:
        return "identity"
    if b < 40 and n > 6.0:
        return "bilat_unsharp"
    if b < 55:
        return "gamma"
    if c < 35:
        return "clahe"
    if h < 4.0:
        return "hist_eq"
    return "retinex"


def compute_regrets(
    predictions: np.ndarray,
    df_scores_sub: pd.DataFrame
) -> np.ndarray:
    """Computes per-sample regret: r(x) = s_{a*}(x) - s_{a_pred}(x) in dB."""
    regrets = []
    for pred, (_, row) in zip(predictions, df_scores_sub.iterrows()):
        psnr_col = f"psnr_{pred}"
        loss = row["oracle_psnr"] - row[psnr_col]
        regrets.append(max(0.0, float(loss)))
    return np.array(regrets)


def run_cross_validation(
    df_scores: pd.DataFrame,
    df_features: pd.DataFrame,
    feature_cols: List[str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes GroupKFold cross-validation across 3 seeds for all models and baselines.
    """
    seeds = config.get("seeds", [42, 123, 999])
    n_splits = config.get("n_splits", 5)

    X = df_features[feature_cols].values
    y = df_scores["label"].values
    groups = df_features["scene_id"].values

    method_names = list(METHODS.keys())

    # Determine Best Fixed Method from majority class across entire dataset
    best_fixed_method = df_scores["label"].value_counts().index[0]

    models_to_test = {
        "Random Choice": "baseline_random",
        "Best Fixed Method": "baseline_fixed",
        "Hand-Written Rule": "baseline_rule",
        "Logistic Regression": "logreg",
        "Random Forest": "rf",
        "XGBoost": "xgb",
    }

    results: Dict[str, Dict[str, List[float]]] = {
        m: {"accuracy": [], "mean_regret": [], "max_regret": []}
        for m in models_to_test
    }

    # Tracking for confusion matrices and error analysis
    fold_predictions = {m: [] for m in models_to_test}
    true_labels = []

    for seed in seeds:
        seed_everything(seed)
        gkf = GroupKFold(n_splits=n_splits)

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            scores_va = df_scores.iloc[va_idx]
            feats_va = df_features.iloc[va_idx]

            if seed == seeds[0]:
                true_labels.extend(y_va)

            # 1. Random Choice Baseline
            np.random.seed(seed + fold * 17)
            rand_preds = np.random.choice(method_names, size=len(y_va))
            rand_acc = float(np.mean(rand_preds == y_va))
            rand_reg = compute_regrets(rand_preds, scores_va)
            results["Random Choice"]["accuracy"].append(rand_acc)
            results["Random Choice"]["mean_regret"].append(float(np.mean(rand_reg)))
            results["Random Choice"]["max_regret"].append(float(np.max(rand_reg)))
            if seed == seeds[0]:
                fold_predictions["Random Choice"].extend(rand_preds)

            # 2. Best Fixed Method Baseline (trained on training fold majority)
            fold_best_fixed = pd.Series(y_tr).value_counts().index[0]
            fixed_preds = np.array([fold_best_fixed] * len(y_va))
            fixed_acc = float(np.mean(fixed_preds == y_va))
            fixed_reg = compute_regrets(fixed_preds, scores_va)
            results["Best Fixed Method"]["accuracy"].append(fixed_acc)
            results["Best Fixed Method"]["mean_regret"].append(float(np.mean(fixed_reg)))
            results["Best Fixed Method"]["max_regret"].append(float(np.max(fixed_reg)))
            if seed == seeds[0]:
                fold_predictions["Best Fixed Method"].extend(fixed_preds)

            # 3. Hand-written Domain Rule Baseline
            rule_preds = np.array([
                evaluate_hand_written_rule(feats_va.iloc[k])
                for k in range(len(feats_va))
            ])
            rule_acc = float(np.mean(rule_preds == y_va))
            rule_reg = compute_regrets(rule_preds, scores_va)
            results["Hand-Written Rule"]["accuracy"].append(rule_acc)
            results["Hand-Written Rule"]["mean_regret"].append(float(np.mean(rule_reg)))
            results["Hand-Written Rule"]["max_regret"].append(float(np.max(rule_reg)))
            if seed == seeds[0]:
                fold_predictions["Hand-Written Rule"].extend(rule_preds)

            # 4. Supervised ML Classifiers with Scaler inside Pipeline
            classifiers = {
                "Logistic Regression": make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=config["models"]["logreg"]["max_iter"],
                        C=config["models"]["logreg"]["C"],
                        random_state=seed,
                    ),
                ),
                "Random Forest": make_pipeline(
                    StandardScaler(),
                    RandomForestClassifier(
                        n_estimators=config["models"]["rf"]["n_estimators"],
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
                "XGBoost": make_pipeline(
                    StandardScaler(),
                    xgb.XGBClassifier(
                        n_estimators=config["models"]["xgb"]["n_estimators"],
                        learning_rate=config["models"]["xgb"]["learning_rate"],
                        max_depth=config["models"]["xgb"]["max_depth"],
                        subsample=config["models"]["xgb"]["subsample"],
                        eval_metric="mlogloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            }

            for model_name, pipe in classifiers.items():
                # Train strictly on tr_idx
                # Label encoding for XGBoost if needed
                classes = sorted(list(np.unique(y_tr)))
                label_to_idx = {c: idx for idx, c in enumerate(classes)}
                idx_to_label = {idx: c for idx, c in enumerate(classes)}

                if "XGBoost" in model_name:
                    y_tr_enc = np.array([label_to_idx[c] for c in y_tr])
                    pipe.fit(X_tr, y_tr_enc)
                    preds_enc = pipe.predict(X_va)
                    preds = np.array([idx_to_label[idx] for idx in preds_enc])
                else:
                    pipe.fit(X_tr, y_tr)
                    preds = pipe.predict(X_va)

                acc = float(np.mean(preds == y_va))
                reg = compute_regrets(preds, scores_va)

                results[model_name]["accuracy"].append(acc)
                results[model_name]["mean_regret"].append(float(np.mean(reg)))
                results[model_name]["max_regret"].append(float(np.max(reg)))
                if seed == seeds[0]:
                    fold_predictions[model_name].extend(preds)

    summary = []
    for model_name, metrics in results.items():
        acc_arr = np.array(metrics["accuracy"]) * 100.0
        mreg_arr = np.array(metrics["mean_regret"])
        xreg_arr = np.array(metrics["max_regret"])

        summary.append({
            "Model": model_name,
            "Accuracy_Mean (%)": round(float(acc_arr.mean()), 2),
            "Accuracy_Std (%)": round(float(acc_arr.std()), 2),
            "Mean_Regret_dB": round(float(mreg_arr.mean()), 3),
            "Mean_Regret_Std_dB": round(float(mreg_arr.std()), 3),
            "Max_Regret_dB": round(float(xreg_arr.mean()), 3),
        })

    df_summary = pd.DataFrame(summary)

    return {
        "summary": df_summary,
        "fold_predictions": fold_predictions,
        "true_labels": np.array(true_labels),
        "best_fixed_method": best_fixed_method,
        "raw_fold_results": results,
    }
