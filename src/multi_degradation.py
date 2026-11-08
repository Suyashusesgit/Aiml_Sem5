"""
Multi-Degradation Generalization & Hierarchical Cascading (Phase 2)
Authors: Team Triathlon (Ananya Kasaudhan, Suyash Srivastav, Logi Vasan)
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Implements Phase 2 research specifications:
1. Synthetic and real composite degradations:
   - Regime A: Pure Low-Light (Photon-starvation)
   - Regime B: Optical Defocus Blur (Gaussian/pillbox convolution)
   - Regime C: Atmospheric Haze (Koschmieder physical scattering model)
   - Regime D: Composite Low-Light + Sensor Noise + Blur
2. 10-Dimensional Extended Physical Degradation Vector:
   - 8 base features: Brightness, Contrast, Entropy, Noise, Dark Channel, Edge Mean, Edge p90, Colourfulness
   - 2 multi-degradation descriptors: Blur Attenuation Ratio, Haze Transmission Metric
3. Multi-Degradation Method Bank:
   - Low-light operators: Gamma (0.5), CLAHE, HistEq, Retinex
   - Deblurring operators: Bilateral+Unsharp, High-Boost Acuity Sharpening
   - Dehazing operator: Dark Channel Prior (DCP) Dehazing
   - Fidelity preservation: Identity
4. Hierarchical Cascaded Selector:
   - Stage 1: Degradation Regime Classifier (Diagnoses primary distortion)
   - Stage 2: Targeted Operator Routing
"""

import os
import sys
import time
from typing import Dict, Tuple, List, Any
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure package root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS, gamma, clahe, hist_eq, retinex, bilat_unsharp, identity
from src.features import extract_features, FEATURE_NAMES
from src.labels import compute_psnr, compute_ssim


EXTENDED_FEATURE_NAMES = FEATURE_NAMES + ["blur_metric", "haze_metric"]


def dcp_dehaze(img: np.ndarray, omega: float = 0.95, t0: float = 0.1, r: int = 7) -> np.ndarray:
    """
    Physical Dark Channel Prior Dehazing (He et al., IEEE TPAMI 2011).
    Estimates atmospheric light A and transmission t(x), recovering scene radiance J(x).
    """
    img_f = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # 1. Dark channel: local patch minimum across channels
    dark_channel = cv2.erode(np.min(img_f, axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)))

    # 2. Atmospheric light A: top 0.1% brightest pixels in dark channel
    num_bright = max(1, int(h * w * 0.001))
    flat_dark = dark_channel.ravel()
    indices = np.argpartition(flat_dark, -num_bright)[-num_bright:]
    flat_img = img_f.reshape(-1, 3)
    A = np.median(flat_img[indices], axis=0)
    A = np.clip(A, 0.1, 1.0)

    # 3. Transmission map t(x) = 1 - omega * min_c(I_c(x) / A_c)
    norm_img = img_f / A
    dark_norm = cv2.erode(np.min(norm_img, axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (r, r)))
    t = 1.0 - omega * dark_norm
    t = cv2.GaussianBlur(t, (15, 15), 0) # Smooth transmission to suppress edge artifacts
    t = np.clip(t, t0, 1.0)[:, :, np.newaxis]

    # 4. Scene radiance recovery J(x) = (I(x) - A) / t(x) + A
    J = (img_f - A) / t + A
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)


def high_boost_sharpen(img: np.ndarray, amount: float = 2.0, sigma: float = 1.2) -> np.ndarray:
    """High-boost edge sharpening in LAB color space to recover optical defocus blur without color shifts."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    blur_l = cv2.GaussianBlur(l, (0, 0), sigmaX=sigma, sigmaY=sigma)
    l_sharp = np.clip(cv2.addWeighted(l, 1.0 + amount, blur_l, -amount, 0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_sharp, a, b]), cv2.COLOR_LAB2BGR)


# Extended multi-degradation portfolio
MULTI_METHODS = {
    "identity": identity,
    "gamma": gamma,
    "clahe": clahe,
    "hist_eq": hist_eq,
    "retinex": retinex,
    "bilat_unsharp": bilat_unsharp,
    "high_boost_sharpen": high_boost_sharpen,
    "dcp_dehaze": dcp_dehaze,
}


def synthesize_degradations(clean_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Synthesizes physical degradation models across four distinct optical regimes:
    1. Low-Light: Non-linear photon attenuation + Poisson noise
    2. Defocus Blur: Gaussian optical blur (sigma=3.5)
    3. Atmospheric Haze: Koschmieder physical scattering (A=0.85, beta=1.2)
    4. Composite: Low-Light + Sensor Noise + Soft Defocus Blur
    """
    clean_f = clean_bgr.astype(np.float32) / 255.0
    h, w, _ = clean_bgr.shape

    # Regime A: Pure Low-Light
    # Power-law compression (gamma=2.2) to simulate photon deficiency
    low_light = (clean_f ** 2.2) * 0.4
    low_light = np.clip(low_light * 255.0, 0, 255).astype(np.uint8)

    # Regime B: Defocus Blur
    blur_img = cv2.GaussianBlur(clean_bgr, (0, 0), sigmaX=3.5, sigmaY=3.5)

    # Regime C: Atmospheric Haze (Koschmieder physical model: I = J*t + A*(1-t))
    # Depth map approximated as distance from bottom-center
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    depth = 0.5 + 0.5 * (1.0 - y_coords / float(h))
    t = np.exp(-1.4 * depth)[:, :, np.newaxis]
    A = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    haze_f = clean_f * t + A * (1.0 - t)
    haze_img = np.clip(haze_f * 255.0, 0, 255).astype(np.uint8)

    # Regime D: Composite (Low-Light + Gaussian Noise + Blur)
    comp = cv2.GaussianBlur(low_light, (0, 0), sigmaX=1.5, sigmaY=1.5)
    noise = np.random.normal(0, 8.0, comp.shape).astype(np.float32)
    comp_img = np.clip(comp.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return {
        "low_light": low_light,
        "defocus_blur": blur_img,
        "atmospheric_haze": haze_img,
        "composite": comp_img,
    }


def extract_extended_features(img_bgr: np.ndarray) -> Dict[str, float]:
    """
    Extracts the 10-dimensional physical degradation feature vector.
    Adds blur attenuation metric and haze transmission metric to the base 8 features.
    """
    feats = extract_features(img_bgr, include_engineered=False)

    # 9. Blur metric: Normalized high-frequency Laplacian variance
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Blur index: close to 1.0 for blurry, close to 0.0 for sharp
    blur_metric = float(1.0 / (1.0 + lap_var / 100.0))

    # 10. Haze metric: Dark Channel relative intensity
    # In hazy images, dark channel intensity is high even in natural shadows
    dc = cv2.erode(np.min(img_bgr, axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    haze_metric = float(np.mean(dc) / 255.0)

    feats["blur_metric"] = round(blur_metric, 4)
    feats["haze_metric"] = round(haze_metric, 4)
    feats["laplacian_var"] = round(lap_var, 2)
    return feats


def hierarchical_selector(feats: Dict[str, float]) -> Tuple[str, str, str]:
    """
    Two-Stage Hierarchical / Cascaded Algorithm Selector:
    Stage 1: Physical Degradation Regime Classification
    Stage 2: Targeted Optimal Operator Routing
    """
    mu = feats["brightness"]
    dc = feats["dark_channel"]
    haze_val = feats["haze_metric"]
    blur_val = feats["blur_metric"]
    noise_val = feats["noise"]

    # Stage 1: Regime Diagnosis
    if mu < 45.0 and dc < 22.0:
        regime = "Low-Light"
        # Stage 2: Low-light policy
        if noise_val > 5.0:
            method = "gamma" # Noise-tolerant shadow lifting
        elif feats["contrast"] < 20.0:
            method = "hist_eq" # Narrow histogram expansion
        else:
            method = "retinex" # Illumination field correction
        reason = f"Photon-deficient regime (μ={mu:.1f} < 45). Selected illumination operator: {method}."

    elif haze_val > 0.28 and mu > 60.0:
        regime = "Atmospheric Haze"
        # Stage 2: Haze policy
        method = "dcp_dehaze"
        reason = f"Atmospheric scattering diagnosed (DCP transmission={haze_val:.2f}). Selected physical dehazing."

    elif blur_val > 0.40 and mu > 45.0:
        regime = "Defocus Blur"
        # Stage 2: Blur policy
        if noise_val < 2.5:
            method = "high_boost_sharpen"
        else:
            method = "bilat_unsharp"
        reason = f"Optical defocus diagnosed (blur index={blur_val:.2f}). Selected high-boost acuity sharpening."

    elif mu < 55.0 and noise_val > 6.0:
        regime = "Composite Noise + Dark"
        method = "gamma"
        reason = f"Composite low-light with sensor noise (noise={noise_val:.1f}). Selected noise-safe gamma."

    else:
        regime = "Well-Exposed Clean"
        method = "identity"
        reason = f"Normal exposure (μ={mu:.1f}). Selected Identity to preserve native fidelity."

    return regime, method, reason


def run_phase2_multi_degradation_benchmark(
    eval_dir: str = "data/lol_dataset/eval15/high",
    output_csv: str = "results/multi_degradation_benchmark.csv",
    figure_path: str = "results/figures/fig10_multi_degradation_cascade.png"
) -> pd.DataFrame:
    """
    Evaluates the Hierarchical Cascaded Selector against static baselines
    across all four optical degradation regimes.
    """
    print("\n" + "="*65)
    print("PHASE 2: Multi-Degradation Cross-Transferability & Cascade Benchmark")
    print("="*65)

    if not os.path.exists(eval_dir):
        alt = "/Users/home/Desktop/LOLdataset/eval15/high"
        if os.path.exists(alt):
            eval_dir = alt
        else:
            raise FileNotFoundError(f"Clean reference images not found at {eval_dir}")

    clean_files = sorted([os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".png")])
    if len(clean_files) == 0:
        raise FileNotFoundError(f"No PNG files in {eval_dir}")

    results = []
    print(f"Loaded {len(clean_files)} clean reference scenes. Synthesizing degradation regimes...")

    for clean_path in clean_files:
        scene_name = os.path.basename(clean_path)
        clean_bgr = cv2.imread(clean_path)
        degrads = synthesize_degradations(clean_bgr)

        for regime_name, deg_img in degrads.items():
            feats = extract_extended_features(deg_img)
            pred_regime, pred_method, reason = hierarchical_selector(feats)

            # Compute PSNR for all methods in portfolio
            psnr_scores = {}
            for m_name, m_fn in MULTI_METHODS.items():
                enhanced = m_fn(deg_img)
                psnr_scores[m_name] = compute_psnr(clean_bgr, enhanced)

            # Oracle winner in this regime
            oracle_method = max(psnr_scores.keys(), key=lambda k: psnr_scores[k])
            oracle_psnr = psnr_scores[oracle_method]

            # Fixed baseline (Retinex)
            fixed_psnr = psnr_scores["retinex"]
            fixed_regret = max(0.0, oracle_psnr - fixed_psnr)

            # Proposed hierarchical selector
            sel_psnr = psnr_scores[pred_method]
            sel_regret = max(0.0, oracle_psnr - sel_psnr)

            results.append({
                "scene": scene_name,
                "true_regime": regime_name,
                "diagnosed_regime": pred_regime,
                "selected_method": pred_method,
                "oracle_method": oracle_method,
                "oracle_psnr": round(oracle_psnr, 2),
                "fixed_retinex_psnr": round(fixed_psnr, 2),
                "fixed_retinex_regret": round(fixed_regret, 3),
                "cascade_psnr": round(sel_psnr, 2),
                "cascade_regret": round(sel_regret, 3),
                "is_correct_selection": (pred_method == oracle_method),
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved benchmark results to {output_csv}")

    # Aggregate by true degradation regime
    summary = df.groupby("true_regime").agg(
        Cascade_Accuracy=("is_correct_selection", lambda x: round(np.mean(x) * 100.0, 1)),
        Cascade_Mean_Regret_dB=("cascade_regret", lambda x: round(np.mean(x), 3)),
        Fixed_Retinex_Regret_dB=("fixed_retinex_regret", lambda x: round(np.mean(x), 3)),
        Regret_Reduction_dB=("fixed_retinex_regret", lambda x: round(np.mean(x) - np.mean(df.loc[x.index, "cascade_regret"]), 3)),
        Cascade_Mean_PSNR=("cascade_psnr", lambda x: round(np.mean(x), 2)),
        Fixed_Retinex_PSNR=("fixed_retinex_psnr", lambda x: round(np.mean(x), 2)),
    ).reset_index()

    print("\n" + "="*80)
    print("PHASE 2 SUMMARY: HIERARCHICAL CASCADE vs. FIXED STATIC RETINEX")
    print("="*80)
    print(summary.to_string(index=False))

    overall_cascade_regret = df["cascade_regret"].mean()
    overall_fixed_regret = df["fixed_retinex_regret"].mean()
    print(f"\nOverall Multi-Degradation Mean Regret: Cascade = {overall_cascade_regret:.3f} dB vs. Fixed Retinex = {overall_fixed_regret:.3f} dB")
    print(f"Net Decision Regret Saving across all regimes: +{overall_fixed_regret - overall_cascade_regret:.3f} dB per frame!")

    # Generate Figure 10: Multi-Degradation Performance Bar Chart
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    regimes = summary["true_regime"].tolist()
    x = np.arange(len(regimes))
    width = 0.35

    # Subplot 1: Decision Regret comparison
    bars1 = ax1.bar(x - width/2, summary["Fixed_Retinex_Regret_dB"], width, label="Fixed Single Method (Retinex)", color="#dc2626", alpha=0.85)
    bars2 = ax1.bar(x + width/2, summary["Cascade_Mean_Regret_dB"], width, label="Proposed Hierarchical Cascade", color="#16a34a", alpha=0.9)
    ax1.set_title("(a) Decision Regret across Optical Regimes (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Mean Decision Regret (dB)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.replace("_", " ").title() for r in regimes], fontsize=10)
    ax1.legend(frameon=True, fontsize=10)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: PSNR Restoration Quality comparison
    bars3 = ax2.bar(x - width/2, summary["Fixed_Retinex_PSNR"], width, label="Fixed Single Method (Retinex)", color="#94a3b8", alpha=0.85)
    bars4 = ax2.bar(x + width/2, summary["Cascade_Mean_PSNR"], width, label="Proposed Hierarchical Cascade", color="#2563eb", alpha=0.9)
    ax2.set_title("(b) Restored Image Fidelity PSNR (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Mean PSNR (dB)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.replace("_", " ").title() for r in regimes], fontsize=10)
    ax2.legend(frameon=True, fontsize=10)

    for bar in bars3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Phase 2: Multi-Degradation Cross-Transferability & Hierarchical Cascading", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 10 to {figure_path}")

    return df


if __name__ == "__main__":
    run_phase2_multi_degradation_benchmark()
