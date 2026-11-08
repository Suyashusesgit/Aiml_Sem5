#!/usr/bin/env python3
"""
Noise Robustness & Sensitivity Sweep Module
Demonstrates that the 8-dimensional handcrafted degradation features act as a continuous physical thermostat:
As sensor noise sigma is synthetically varied in [0, 50], class probabilities shift smoothly from
contrast-boosting operators (Retinex/HistEq) to edge-preserving smoothing (Bilateral+Unsharp / Gamma).

Outputs to results/noise_robustness_sweep.csv and results/figures/fig8_robustness_noise_sweep.png
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import resolve_dataset_dir, get_dataset_manifest, load_and_preprocess_pair
from src.features import extract_features, FEATURE_NAMES
from src.demo import get_or_train_selector
from src.methods import METHODS


def run_robustness_sweep():
    data_dir = resolve_dataset_dir()
    manifest = get_dataset_manifest(data_dir, split_name="eval15")
    sample = manifest[0]  # Use sample 1.png
    clean, deg, _ = load_and_preprocess_pair(sample["low_path"], sample["high_path"], target_long_side=512)

    model = get_or_train_selector()
    classes = list(model.classes_)

    noise_levels = np.linspace(0, 50, 11)  # 0 to 50 in steps of 5
    records = []

    np.random.seed(42)

    for sigma in noise_levels:
        # Add zero-mean Gaussian sensor noise
        if sigma > 0:
            noise = np.random.normal(0, sigma, deg.shape).astype(np.float32)
            noisy = np.clip(deg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            noisy = deg.copy()

        feats = extract_features(noisy, include_engineered=True)
        X = np.array([[feats[k] for k in FEATURE_NAMES]])

        pred_cls = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        rec = {
            "Noise_Sigma": sigma,
            "Predicted_Method": pred_cls,
            "Brightness": round(feats["brightness"], 2),
            "Contrast": round(feats["contrast"], 2),
            "Entropy": round(feats["entropy"], 3),
            "Noise_Feature": round(feats["noise"], 3),
            "Dark_Channel": round(feats["dark_channel"], 3),
        }
        for cls_name, p in zip(classes, probs):
            rec[f"Prob_{cls_name}"] = round(float(p) * 100.0, 2)

        records.append(rec)

    df_sweep = pd.DataFrame(records)
    out_csv = "results/noise_robustness_sweep.csv"
    df_sweep.to_csv(out_csv, index=False)
    print(f"Saved noise sensitivity sweep data to {out_csv}")

    # Plot Figure 8: Two-panel sensitivity curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Physical Feature Response
    axes[0].plot(df_sweep["Noise_Sigma"], df_sweep["Noise_Feature"], marker="o", color="#DC2626", linewidth=2.2, label="Estimated Noise (MAD)")
    axes[0].plot(df_sweep["Noise_Sigma"], df_sweep["Entropy"], marker="s", color="#2563EB", linewidth=2.2, label="Shannon Entropy (bits)")
    axes[0].set_title("(a) Physical Feature Response vs. Injected Noise (σ)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Injected Gaussian Noise Standard Deviation (σ)", fontsize=11)
    axes[0].set_ylabel("Feature Magnitude", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # Panel 2: Selection Probability Trajectory
    color_map = {
        "retinex": "#1E3A8A",
        "hist_eq": "#D97706",
        "gamma": "#2563EB",
        "bilat_unsharp": "#0D9488",
        "clahe": "#9333EA",
        "identity": "#64748B",
    }

    for cls_name in classes:
        prob_col = f"Prob_{cls_name}"
        if df_sweep[prob_col].max() > 5.0:  # Only plot active classes
            axes[1].plot(
                df_sweep["Noise_Sigma"],
                df_sweep[prob_col],
                marker="o",
                linewidth=2.2,
                label=cls_name.upper(),
                color=color_map.get(cls_name, "#475569")
            )

    axes[1].set_title("(b) Model Selection Probability Trajectory", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Injected Gaussian Noise Standard Deviation (σ)", fontsize=11)
    axes[1].set_ylabel("Class Probability (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    out_fig = "results/figures/fig8_robustness_noise_sweep.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved robustness sweep figure to {out_fig}")


if __name__ == "__main__":
    run_robustness_sweep()

