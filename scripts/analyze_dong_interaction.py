"""
Analysis of Dong et al. Invert-Dehaze Interaction with Handcrafted Physical Features
Evaluates how the Invert-Dehaze operator interacts with the 7 physical features:
Brightness, Contrast, Entropy, Noise, Dark Channel, Edge Statistics, Colourfulness.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.labels import compute_psnr, compute_ssim


def analyze_dataset(low_dir: str, high_dir: str, max_samples: int = 50) -> pd.DataFrame:
    records = []
    low_files = sorted(glob.glob(os.path.join(low_dir, "*.png")))[:max_samples]

    for lf in low_files:
        base = os.path.basename(lf)
        hf = os.path.join(high_dir, base)
        if not os.path.exists(hf):
            continue

        deg = cv2.imread(lf)
        clean = cv2.imread(hf)
        if deg is None or clean is None:
            continue

        feats = extract_features(deg, include_engineered=True)
        method_psnr = {}
        for m_name, m_fn in METHODS.items():
            enh = m_fn(deg)
            method_psnr[m_name] = compute_psnr(clean, enh)

        oracle_method = max(method_psnr.keys(), key=lambda k: method_psnr[k])
        oracle_psnr = method_psnr[oracle_method]
        dehaze_psnr = method_psnr["invert_dehaze"]
        dehaze_regret = max(0.0, oracle_psnr - dehaze_psnr)

        row = {
            "image": base,
            "oracle_method": oracle_method,
            "oracle_psnr": round(oracle_psnr, 2),
            "dehaze_psnr": round(dehaze_psnr, 2),
            "dehaze_regret": round(dehaze_regret, 2),
            "dehaze_is_winner": int(oracle_method == "invert_dehaze"),
        }
        for k in FEATURE_NAMES:
            row[k] = round(feats[k], 3)
        records.append(row)

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df_eval = analyze_dataset("data/lol_dataset/eval15/low", "data/lol_dataset/eval15/high")
    df_train = analyze_dataset("data/lol_dataset/our485/low", "data/lol_dataset/our485/high", max_samples=40)
    df_all = pd.concat([df_eval, df_train], ignore_index=True)

    print("=== Dong Invert-Dehaze Interaction Analysis ===")
    print(f"Total samples analyzed: {len(df_all)}")
    print(f"Invert-Dehaze Oracle Win Count: {df_all['dehaze_is_winner'].sum()} / {len(df_all)}")
    print(f"Mean Invert-Dehaze PSNR: {df_all['dehaze_psnr'].mean():.2f} dB")
    print(f"Mean Invert-Dehaze Regret: {df_all['dehaze_regret'].mean():.2f} dB")

    # Correlations between feature values and Invert-Dehaze PSNR
    corrs = {}
    for feat in FEATURE_NAMES:
        c = df_all[feat].corr(df_all["dehaze_psnr"])
        corrs[feat] = round(c, 3)

    print("\nFeature Correlations with Invert-Dehaze PSNR:")
    for feat, c in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:20s}: r = {c:+.3f}")

    os.makedirs("results", exist_ok=True)
    df_all.to_csv("results/dong_interaction_analysis.csv", index=False)
    print("\nSaved detailed interaction records to results/dong_interaction_analysis.csv")
