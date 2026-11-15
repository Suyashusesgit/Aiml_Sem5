"""
Pipeline Module
High-level data preparation, caching, and preprocessing orchestration
for Project Triathlon per Appendix A & Section 6.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import pandas as pd
import yaml

from src.methods import CANONICAL_METHODS, METHODS
from src.features import extract_features, FEATURE_NAMES, validate_feature_dataframe
from src.labels import compute_psnr, compute_ssim, score_pair
from src.dataset import (
    resolve_dataset_dir,
    get_dataset_manifest,
    load_and_preprocess_pair,
    compute_dhash,
    compute_scene_groups_dhash,
)


def prepare(
    config: Optional[Dict[str, Any]] = None,
    config_path: str = "configs/base.yaml",
    dataset: str = "lol_v1",
    split: str = "our485",
    hamming_threshold: int = 10,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Prepares dataset features, oracle scores, and dHash scene groups.
    Enforces Appendix A rules:
    - 512px downscaling on long side via INTER_AREA
    - 64-bit dHash on clean reference image at Hamming <= threshold
    - 8 physical degradation features extracted strictly on degraded input
    - Anti-leakage validation on feature columns
    """
    if config is None:
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    target_long_side = config.get("experiment", {}).get("target_long_side", 512)
    tie_threshold_db = config.get("experiment", {}).get("tie_threshold_db", 0.05)
    
    # Resolve root dataset directory
    raw_data_dir = config.get("data", {}).get("data_dir", None)
    data_dir = resolve_dataset_dir(raw_data_dir)
    manifest = get_dataset_manifest(data_dir, split_name=split)

    if len(manifest) == 0:
        raise FileNotFoundError(f"No paired images found for dataset '{dataset}', split '{split}' in {data_dir}")

    scores_cache = f"results/scores_{dataset}_{split}.csv"
    features_cache = f"results/features_{dataset}_{split}.csv"
    os.makedirs("results", exist_ok=True)

    df_scores = None
    df_features = None

    if not force_recompute and os.path.isfile(scores_cache) and os.path.isfile(features_cache):
        try:
            df_s = pd.read_csv(scores_cache)
            df_f = pd.read_csv(features_cache)
            if len(df_s) == len(manifest) and len(df_f) == len(manifest):
                df_scores = df_s
                df_features = df_f
        except Exception:
            df_scores, df_features = None, None

    if df_scores is None or df_features is None:
        print(f"Building cache for {dataset} [{split}] ({len(manifest)} pairs)...")
        score_rows = []
        feature_rows = []
        hashes = []

        for item in manifest:
            clean, deg, scale = load_and_preprocess_pair(
                item["low_path"], item["high_path"], target_long_side=target_long_side
            )
            # 1. Oracle scores & label (using 6 canonical methods)
            clean_rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
            deg_rgb = cv2.cvtColor(deg, cv2.COLOR_BGR2RGB)

            s_row = {"filename": item["filename"]}
            for m in CANONICAL_METHODS:
                enhanced_bgr = METHODS[m](deg)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                if enhanced_rgb.shape[:2] != clean_rgb.shape[:2]:
                    enhanced_rgb = cv2.resize(enhanced_rgb, (clean_rgb.shape[1], clean_rgb.shape[0]))
                p = compute_psnr(clean_rgb, enhanced_rgb, max_val=255.0)
                s = compute_ssim(clean_rgb, enhanced_rgb, max_val=255.0)
                s_row[f"psnr_{m}"] = p
                s_row[f"ssim_{m}"] = s

            # Oracle decision with 0.05 dB tie breaking
            max_p = max(s_row[f"psnr_{m}"] for m in CANONICAL_METHODS)
            candidates = [m for m in CANONICAL_METHODS if (max_p - s_row[f"psnr_{m}"]) <= tie_threshold_db]
            if len(candidates) == 1:
                oracle_label = candidates[0]
            else:
                oracle_label = max(candidates, key=lambda m: s_row[f"ssim_{m}"])

            s_row["oracle_psnr"] = max_p
            s_row["label"] = oracle_label
            score_rows.append(s_row)

            # 2. Features from degraded ONLY
            feats = extract_features(deg, include_engineered=False)
            feats["filename"] = item["filename"]
            feature_rows.append(feats)

            # 3. 64-bit dHash on reference
            h = compute_dhash(clean)
            hashes.append(h)

        df_scores = pd.DataFrame(score_rows)
        df_features = pd.DataFrame(feature_rows)

        df_scores.to_csv(scores_cache, index=False)
        df_features.to_csv(features_cache, index=False)
        print(f"Saved scores to {scores_cache} and features to {features_cache}")
    else:
        # Load reference hashes
        hashes = []
        for item in manifest:
            clean, _, _ = load_and_preprocess_pair(
                item["low_path"], item["high_path"], target_long_side=target_long_side
            )
            hashes.append(compute_dhash(clean))

    # Compute dHash scene groups
    hashes_arr = np.array(hashes)
    scene_groups = compute_scene_groups_dhash(hashes_arr, hamming_threshold=hamming_threshold)
    df_scores["scene_group"] = scene_groups
    df_features["scene_group"] = scene_groups

    # Validate feature columns (Anti-Leakage G4)
    feature_cols = [c for c in df_features.columns if c not in ("filename", "scene_group")]
    validate_feature_dataframe(df_features, feature_cols)

    X = df_features[feature_cols].values
    y = df_scores["label"].values

    return {
        "X": X,
        "y": y,
        "groups": scene_groups,
        "df_scores": df_scores,
        "df_features": df_features,
        "feature_names": feature_cols,
        "manifest": manifest,
        "dataset": dataset,
        "split": split,
        "hashes": hashes_arr,
    }
