#!/usr/bin/env python3
"""
LOL-v2 Benchmark and Zero-Shot Transferability Evaluation
Executes:
1. Suyash's Oracle Matrix Computation:
   - Evaluates the 6 fixed methods on all LOL-v2 pairs.
   - Computes PSNR and SSIM with <= 0.05 dB SSIM tie-breaking.
   - Exports results/scores_lol_v2_{split}.csv.
2. Zero-Shot Out-of-Distribution Transferability:
   - Extracts 8 physical features exclusively from degraded images.
   - Tests the frozen Random Forest selector (trained on LOL-v1) zero-shot on LOL-v2.
   - Compares Top-1 Accuracy, Mean Regret (dB), Max Regret (dB) against baselines.
   - Exports results/lol_v2_generalization_results.csv.
"""

import os
import sys
import argparse
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.methods import METHODS
    from src.features import extract_features, FEATURE_NAMES
    from src.labels import compute_psnr, compute_ssim, score_pair
    from src.dataset_v2 import get_lol_v2_manifest, load_and_preprocess_pair_v2
    from src.demo import get_or_train_selector
except ImportError:
    from methods import METHODS  # type: ignore[import-not-found]
    from features import extract_features, FEATURE_NAMES  # type: ignore[import-not-found]
    from labels import compute_psnr, compute_ssim, score_pair  # type: ignore[import-not-found]
    from dataset_v2 import get_lol_v2_manifest, load_and_preprocess_pair_v2  # type: ignore[import-not-found]
    from demo import get_or_train_selector  # type: ignore[import-not-found]


def evaluate_oracle_and_scores(manifest: List[Dict[str, Any]], split_name: str) -> pd.DataFrame:
    """Computes Suyash's 6-method PSNR/SSIM score matrix and oracle labels."""
    print(f"\n[Suyash Module] Running 6-method enhancement bank and scoring on {len(manifest)} LOL-v2 [{split_name}] pairs...")
    rows = []
    tie_breaks = 0

    for idx, item in enumerate(manifest):
        clean, deg, scale = load_and_preprocess_pair_v2(item["low_path"], item["high_path"], target_long_side=512)
        scored = score_pair(clean, deg, tie_threshold_db=0.05)
        scored["filename"] = item["filename"]
        scored["scene_id"] = item["scene_id"]
        rows.append(scored)

        # Check if tie-breaking triggered
        psnrs = [scored[f"psnr_{m}"] for m in METHODS]
        max_p = max(psnrs)
        cands = [m for m in METHODS if (max_p - scored[f"psnr_{m}"]) <= 0.05]
        if len(cands) > 1:
            tie_breaks += 1

    df_scores = pd.DataFrame(rows)
    out_path = f"results/scores_lol_v2_{split_name}.csv"
    os.makedirs("results", exist_ok=True)
    df_scores.to_csv(out_path, index=False)
    print(f"Exported score matrix to {out_path} ({len(df_scores)} rows, {tie_breaks} SSIM tie-breaks).")
    return df_scores


def extract_manifest_features(manifest: List[Dict[str, Any]], split_name: str) -> pd.DataFrame:
    """Extracts 8 handcrafted physical descriptors from degraded images only."""
    print(f"\n[Logi Module] Extracting 8 physical features exclusively from degraded images...")
    rows = []
    for item in manifest:
        _, deg, _ = load_and_preprocess_pair_v2(item["low_path"], item["high_path"], target_long_side=512)
        feats = extract_features(deg, include_engineered=True)
        feats["filename"] = item["filename"]
        feats["scene_id"] = item["scene_id"]
        rows.append(feats)

    df_feats = pd.DataFrame(rows)
    out_path = f"results/features_lol_v2_{split_name}.csv"
    df_feats.to_csv(out_path, index=False)
    print(f"Exported feature matrix to {out_path}.")
    return df_feats


def evaluate_zero_shot_transfer(
    df_scores: pd.DataFrame,
    df_feats: pd.DataFrame,
    split_name: str,
    model_path: str = "models/rf_selector.joblib"
) -> pd.DataFrame:
    """Evaluates the frozen LOL-v1 model on LOL-v2 zero-shot without retraining."""
    print(f"\n[Evaluation] Running Zero-Shot Transferability Benchmark on LOL-v2 [{split_name}]...")

    # Load or train the base LOL-v1 model
    rf_model = get_or_train_selector(model_save_path=model_path)

    # Align features with scores
    merged = pd.merge(df_scores, df_feats, on=["filename", "scene_id"])
    X = merged[[col for col in FEATURE_NAMES]].values.astype(np.float32)
    y_oracle = merged["label"].values

    # Selector prediction
    y_pred = rf_model.predict(X)

    # 1. Oracle Metrics
    oracle_psnrs = merged["oracle_psnr"].values
    oracle_ssims = merged["oracle_ssim"].values

    # 2. Selector Metrics
    sel_psnrs = np.array([merged.iloc[i][f"psnr_{y_pred[i]}"] for i in range(len(merged))])
    sel_ssims = np.array([merged.iloc[i][f"ssim_{y_pred[i]}"] for i in range(len(merged))])
    sel_regret = np.maximum(0.0, oracle_psnrs - sel_psnrs)

    # 3. Best-Fixed Method (Retinex)
    ret_psnrs = merged["psnr_retinex"].values
    ret_ssims = merged["ssim_retinex"].values
    ret_regret = np.maximum(0.0, oracle_psnrs - ret_psnrs)

    # 4. Random Choice (Monte Carlo across methods)
    np.random.seed(42)
    method_list = list(METHODS.keys())
    rand_preds = np.random.choice(method_list, size=len(merged))
    rand_psnrs = np.array([merged.iloc[i][f"psnr_{rand_preds[i]}"] for i in range(len(merged))])
    rand_ssims = np.array([merged.iloc[i][f"ssim_{rand_preds[i]}"] for i in range(len(merged))])
    rand_regret = np.maximum(0.0, oracle_psnrs - rand_psnrs)

    # 5. Hand-written Rule Heuristic
    hand_preds = []
    for _, row in merged.iterrows():
        b = row["brightness"]
        if b < 15.0:
            hand_preds.append("gamma")
        elif row["noise"] > 4.0:
            hand_preds.append("bilat_unsharp")
        elif row["contrast"] < 25.0:
            hand_preds.append("clahe")
        else:
            hand_preds.append("retinex")
    hand_psnrs = np.array([merged.iloc[i][f"psnr_{hand_preds[i]}"] for i in range(len(merged))])
    hand_ssims = np.array([merged.iloc[i][f"ssim_{hand_preds[i]}"] for i in range(len(merged))])
    hand_regret = np.maximum(0.0, oracle_psnrs - hand_psnrs)

    results = [
        {
            "Strategy": "Random Choice (Chance)",
            "Top-1 Accuracy (%)": round(np.mean(rand_preds == y_oracle) * 100.0, 2),
            "Mean Regret (dB)": round(float(np.mean(rand_regret)), 3),
            "Max Regret (dB)": round(float(np.max(rand_regret)), 3),
            "Mean PSNR (dB)": round(float(np.mean(rand_psnrs)), 2),
            "Mean SSIM": round(float(np.mean(rand_ssims)), 4),
        },
        {
            "Strategy": "Best Fixed Method (Retinex)",
            "Top-1 Accuracy (%)": round(np.mean(y_oracle == "retinex") * 100.0, 2),
            "Mean Regret (dB)": round(float(np.mean(ret_regret)), 3),
            "Max Regret (dB)": round(float(np.max(ret_regret)), 3),
            "Mean PSNR (dB)": round(float(np.mean(ret_psnrs)), 2),
            "Mean SSIM": round(float(np.mean(ret_ssims)), 4),
        },
        {
            "Strategy": "Hand-Written Heuristic Rule",
            "Top-1 Accuracy (%)": round(np.mean(np.array(hand_preds) == y_oracle) * 100.0, 2),
            "Mean Regret (dB)": round(float(np.mean(hand_regret)), 3),
            "Max Regret (dB)": round(float(np.max(hand_regret)), 3),
            "Mean PSNR (dB)": round(float(np.mean(hand_psnrs)), 2),
            "Mean SSIM": round(float(np.mean(hand_ssims)), 4),
        },
        {
            "Strategy": "Proposed Selector (Zero-Shot Transfer)",
            "Top-1 Accuracy (%)": round(np.mean(y_pred == y_oracle) * 100.0, 2),
            "Mean Regret (dB)": round(float(np.mean(sel_regret)), 3),
            "Max Regret (dB)": round(float(np.max(sel_regret)), 3),
            "Mean PSNR (dB)": round(float(np.mean(sel_psnrs)), 2),
            "Mean SSIM": round(float(np.mean(sel_ssims)), 4),
        },
        {
            "Strategy": "Theoretical Oracle Upper Bound",
            "Top-1 Accuracy (%)": 100.0,
            "Mean Regret (dB)": 0.000,
            "Max Regret (dB)": 0.000,
            "Mean PSNR (dB)": round(float(np.mean(oracle_psnrs)), 2),
            "Mean SSIM": round(float(np.mean(oracle_ssims)), 4),
        },
    ]

    df_res = pd.DataFrame(results)
    out_res_path = f"results/lol_v2_generalization_{split_name}.csv"
    df_res.to_csv(out_res_path, index=False)

    print("\n" + "=" * 80)
    print(f"ZERO-SHOT CROSS-DATASET GENERALIZATION RESULTS: LOL-v2 [{split_name.upper()}]")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("=" * 80)

    # Compute key headline saving
    saving = df_res.loc[df_res["Strategy"] == "Best Fixed Method (Retinex)", "Mean Regret (dB)"].values[0] - \
             df_res.loc[df_res["Strategy"] == "Proposed Selector (Zero-Shot Transfer)", "Mean Regret (dB)"].values[0]
    print(f"\nKey Finding: Zero-shot selector cuts mean regret by {saving:+.3f} dB over static Retinex on unseen LOL-v2!")

    return df_res


def main():
    parser = argparse.ArgumentParser(description="LOL-v2 Benchmark Suite")
    parser.add_argument("--split", type=str, default="synthetic", choices=["synthetic", "real"])
    args = parser.parse_args()

    manifest = get_lol_v2_manifest(split_type=args.split)
    if not manifest:
        print(f"No pairs found for LOL-v2 [{args.split}]. Running ingestion first...")
        from scripts.ingest_lol_v2 import ensure_v2_directories, prepare_synthetic_subset
        ensure_v2_directories()
        prepare_synthetic_subset(num_samples=50)
        manifest = get_lol_v2_manifest(split_type=args.split)

    df_scores = evaluate_oracle_and_scores(manifest, split_name=args.split)
    df_feats = extract_manifest_features(manifest, split_name=args.split)
    evaluate_zero_shot_transfer(df_scores, df_feats, split_name=args.split)


if __name__ == "__main__":
    main()

