"""
Predict-Then-Optimize (PTO) Enhancement Selector (Phase 0 Deferred Milestone)
Authors: Project Triathlon Engineering Team
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Implements Page 9 Deferred Future Scope:
"Predict-then-optimize: six regressors (one per enhancer) predicting PSNR,
 then pick argmax - target 0.8 to 1.2 dB mean regret."
"""

import os
import sys
import time
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

# Ensure package root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import FEATURE_NAMES, ENGINEERED_FEATURE_NAMES, extract_features

PTO_FEATURE_NAMES = FEATURE_NAMES + ENGINEERED_FEATURE_NAMES
CANONICAL_METHODS = ["identity", "gamma", "clahe", "hist_eq", "retinex", "bilat_unsharp"]


class PredictThenOptimizeSelector:
    """
    Predict-Then-Optimize architecture:
    Instead of discrete multi-class classification, fits K continuous regression models
    f_k(x) -> PSNR_k(x) and selects a*(x) = argmax_k f_k(x).
    """

    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.methods = list(CANONICAL_METHODS)

    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]):
        self.models = {}
        for m in self.methods:
            if self.model_type == "ridge":
                reg = RidgeCV(alphas=np.logspace(-2, 3, 25))
            else:
                reg = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42)
            reg.fit(X, y_dict[m])
            self.models[m] = reg

    def predict_psnr(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        preds = {}
        for m in self.methods:
            preds[m] = self.models[m].predict(X)
        return preds

    def select(self, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
        preds = self.predict_psnr(X)
        pred_matrix = np.column_stack([preds[m] for m in self.methods])
        best_indices = np.argmax(pred_matrix, axis=1)
        selected_methods = [self.methods[idx] for idx in best_indices]
        return selected_methods, pred_matrix

    def save(self, path: str = "models/pto_regressors.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "models/pto_regressors.joblib") -> "PredictThenOptimizeSelector":
        return joblib.load(path)


def evaluate_pto_cross_validation(
    features_csv: str = "results/features.csv",
    scores_csv: str = "results/scores.csv",
    n_splits: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates Predict-Then-Optimize using 5-fold GroupKFold by physical scene.
    """
    df_f = pd.read_csv(features_csv)
    df_s = pd.read_csv(scores_csv)
    df = pd.merge(
        df_f,
        df_s[["filename", "psnr_identity", "psnr_gamma", "psnr_clahe", "psnr_hist_eq", "psnr_retinex", "psnr_bilat_unsharp", "label", "oracle_psnr"]],
        on="filename"
    )

    X_all = df[PTO_FEATURE_NAMES].values
    groups = df["scene_id"].values
    y_all = {m: df[f"psnr_{m}"].values for m in CANONICAL_METHODS}
    oracle_psnrs = df["oracle_psnr"].values
    true_labels = df["label"].values

    gkf = GroupKFold(n_splits=n_splits)
    regrets = []
    correct_count = 0

    for train_idx, val_idx in gkf.split(X_all, groups=groups):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train = {m: y_all[m][train_idx] for m in CANONICAL_METHODS}
        
        pto = PredictThenOptimizeSelector(model_type="ridge")
        pto.fit(X_train, y_train)

        preds_val, matrix_val = pto.select(X_val)

        for i, pred_m in enumerate(preds_val):
            val_global_idx = val_idx[i]
            actual_psnr = y_all[pred_m][val_global_idx]
            oracle = oracle_psnrs[val_global_idx]
            regret = max(0.0, oracle - actual_psnr)
            regrets.append(regret)
            if pred_m == true_labels[val_global_idx]:
                correct_count += 1

    mean_regret = float(np.mean(regrets))
    top1_acc = float(correct_count / len(X_all)) * 100.0

    return {
        "mean_regret_db": round(mean_regret, 3),
        "top1_acc": round(top1_acc, 2),
        "total_samples": len(X_all),
        "scenes": len(np.unique(groups)),
    }


def train_and_save_production_pto():
    df_f = pd.read_csv("results/features.csv")
    df_s = pd.read_csv("results/scores.csv")
    df = pd.merge(
        df_f,
        df_s[["filename", "psnr_identity", "psnr_gamma", "psnr_clahe", "psnr_hist_eq", "psnr_retinex", "psnr_bilat_unsharp", "label", "oracle_psnr"]],
        on="filename"
    )

    X_all = df[PTO_FEATURE_NAMES].values
    y_all = {m: df[f"psnr_{m}"].values for m in CANONICAL_METHODS}

    pto = PredictThenOptimizeSelector(model_type="ridge")
    pto.fit(X_all, y_all)
    pto.save("models/pto_regressors.joblib")
    print("Successfully trained and saved production PTO selector to models/pto_regressors.joblib")


if __name__ == "__main__":
    res = evaluate_pto_cross_validation()
    print("=== Predict-Then-Optimize (PTO) 5-Fold GroupKFold Results ===")
    print(f"Mean Regret: {res['mean_regret_db']} dB")
    print(f"Top-1 Accuracy: {res['top1_acc']}%")
    print(f"Samples: {res['total_samples']} across {res['scenes']} scenes")

    train_and_save_production_pto()
