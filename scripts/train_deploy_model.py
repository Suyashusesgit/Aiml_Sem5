"""
Train and Deploy Winning Phase 0 Selector Model (Logistic Regression).
Serialized strictly according to pre-committed decision rule verdict in results/decision/verdict.json.
"""

import os
import sys
import json
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import CANONICAL_METHODS
from src.features import FEATURE_NAMES, validate_feature_dataframe
from src.selectors import make_selector
from src.pipeline import prepare


def main():
    print("=" * 70)
    print("DEPLOYING WINNING MODEL ACCORDING TO PRE-COMMITTED VERDICT")
    print("=" * 70)

    verdict_path = "results/decision/verdict.json"
    if not os.path.exists(verdict_path):
        raise FileNotFoundError(f"Verdict file not found at {verdict_path}. Run experiments first!")

    with open(verdict_path, "r") as fp:
        verdict_data = json.load(fp)

    print(f"Loaded Verdict : {verdict_data['verdict']}")
    print(f"Deploying Model: {verdict_data['deployed_model']}")

    # Load LOL-v1 our485 canonical data
    data = prepare(dataset="lol_v1", split="our485", hamming_threshold=10)
    X = data["X"]
    y = data["y"]
    df_scores = data["df_scores"]

    # Fit Logistic Regression with C=1.0, max_iter=2000, random_state=42
    print(f"\nFitting Logistic Regression on all {len(X)} training samples...")
    selector = make_selector(
        "logreg",
        feature_names=FEATURE_NAMES,
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=42,
    )
    selector.fit(X, y, df_scores=df_scores)

    os.makedirs("models", exist_ok=True)
    model_path = "models/selector.joblib"
    meta_path = "models/selector.json"

    # Save joblib model
    joblib.dump(selector, model_path)
    print(f"Saved model checkpoint to {model_path}")

    # Compute SHA-256
    with open(model_path, "rb") as fp:
        sha256 = hashlib.sha256(fp.read()).hexdigest()

    # Load CV metrics from results/decision/explore_default/summary_cv.csv
    summary_path = "results/decision/explore_default/summary_cv.csv"
    cv_metrics = {}
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        lr_row = df_sum[df_sum["selector"] == "Logistic Regression"].iloc[0]
        cv_metrics = {
            "top1_acc_pct": float(lr_row["top1_acc"]),
            "mean_regret_db": float(lr_row["mean_regret_db"]),
            "regret_95_ci_lo": float(lr_row["ci_lo"]),
            "regret_95_ci_hi": float(lr_row["ci_hi"]),
            "p90_regret_db": float(lr_row["p90_regret_db"]),
            "max_regret_db": float(lr_row["max_regret_db"]),
            "gap_closed_pct": float(lr_row["gap_closed_pct"]),
        }

    # Meta dictionary
    meta = {
        "model_name": "Multinomial Logistic Regression",
        "selector_type": "logreg",
        "feature_names": FEATURE_NAMES,
        "classes": [str(c) for c in selector.classes_],
        "canonical_methods": CANONICAL_METHODS,
        "hyperparameters": {
            "C": 1.0,
            "class_weight": None,
            "max_iter": 2000,
            "random_state": 42,
            "solver": "lbfgs",
        },
        "training_dataset": "LOL-v1 our485",
        "n_training_samples": len(X),
        "pipeline_stages": ["StandardScaler", "LogisticRegression"],
        "model_file": model_path,
        "sha256": sha256,
        "evidence_file": verdict_path,
        "verdict": verdict_data["verdict"],
        "cv_performance_5fold_grouped": cv_metrics,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "packages": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        meta["git_commit"] = commit
    except Exception:
        meta["git_commit"] = "unknown"

    with open(meta_path, "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"Saved metadata to {meta_path}")
    print(f"SHA-256: {sha256}")
    print("\nModel training and deployment complete.")


if __name__ == "__main__":
    main()
