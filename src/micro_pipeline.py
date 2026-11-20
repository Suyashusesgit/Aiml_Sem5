"""
Micro-Pipeline Generator: Adaptive Two-Stage Degradation Cascades (Phase 2 Milestone 2.3)
Implements dynamic DAG-based 2-stage restoration micro-pipelines:
    Input Image --> Stage 1: Physical Conditioning (Denoise / Dehaze / Sharpen / Passthrough)
               --> Stage 2: Photometric Restoration (Gamma / CLAHE / Hist-Eq / Retinex)
               --> Output Restored Image

Pruned using physical regime diagnostics to respect edge real-time latency budgets (< 25 ms).
"""

import os
import sys
import time
from typing import Dict, Tuple, Any, List, Optional
import cv2
import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import FEATURE_NAMES, extract_features
from src.methods import (
    identity,
    gamma,
    clahe,
    hist_eq,
    retinex,
    bilat_unsharp,
    invert_dehaze,
)
from src.labels import compute_psnr
from src.multi_degradation import (
    extract_extended_features,
    high_boost_sharpen,
    dcp_dehaze,
    get_champion_model,
)
from src.predict_then_optimize import PredictThenOptimizeSelector


def bilateral_denoise(img_bgr: np.ndarray, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """Edge-preserving bilateral smoothing for high sensor noise suppression."""
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


STAGE1_OPERATORS = {
    "identity": identity,
    "bilateral_denoise": bilateral_denoise,
    "dcp_dehaze": dcp_dehaze,
    "high_boost_sharpen": high_boost_sharpen,
}

STAGE2_OPERATORS = {
    "identity": identity,
    "gamma": gamma,
    "clahe": clahe,
    "hist_eq": hist_eq,
    "retinex": retinex,
    "bilat_unsharp": bilat_unsharp,
}


class AdaptiveMicroPipeline:
    """
    Two-stage dynamic restoration DAG:
    Stage 1: Diagnoses physical artifact regime (Veiling Haze, High Noise, Defocus Blur)
             and executes targeted optical conditioning.
    Stage 2: Photometric restoration routed via Predict-Then-Optimize (PTO)
             or Champion classifier on post-conditioned feature representations.
    """

    def __init__(self, pto_path: str = "models/pto_regressors.joblib"):
        self.pto_selector = None
        if os.path.exists(pto_path):
            try:
                self.pto_selector = PredictThenOptimizeSelector.load(pto_path)
            except Exception:
                self.pto_selector = None
        self.champion_model = get_champion_model()

    def diagnose_stage1(self, feats: Dict[str, float]) -> Tuple[str, str]:
        """
        Diagnoses whether optical pre-conditioning is required prior to tone restoration.
        """
        haze_val = feats.get("haze_metric", 0.0)
        noise_val = feats.get("noise", 0.0)
        blur_val = feats.get("blur_metric", 0.0)
        mu = feats.get("brightness", 0.0)

        if haze_val > 0.28 and mu > 50.0:
            return "dcp_dehaze", f"Volumetric scattering diagnosed (DCP transmission={haze_val:.2f}). Stage 1: Dark Channel Dehazing."
        elif noise_val > 5.5:
            return "bilateral_denoise", f"High sensor noise diagnosed (sigma={noise_val:.1f}). Stage 1: Edge-preserving Bilateral Filter."
        elif blur_val > 0.42 and mu > 45.0:
            return "high_boost_sharpen", f"Defocus attenuation diagnosed (blur index={blur_val:.2f}). Stage 1: High-boost Acuity Sharpening."
        else:
            return "identity", "No dominant optical distortion requiring pre-conditioning. Stage 1: Direct passthrough."

    def select_stage2(self, conditioned_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Selects Stage 2 photometric restoration operator using post-conditioning features.
        """
        feats_cond = extract_extended_features(conditioned_bgr)
        if self.pto_selector is not None:
            method, exp_psnr = self.pto_selector.predict_best_method(feats_cond)
            return method, exp_psnr
        elif self.champion_model is not None:
            X_in = np.array([[feats_cond[k] for k in FEATURE_NAMES]])
            method = self.champion_model.predict(X_in)[0]
            return method, 0.0
        else:
            # Fallback heuristic
            if feats_cond.get("contrast", 0.0) < 20.0:
                return "hist_eq", 0.0
            return "gamma", 0.0

    def process(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Executes full two-stage micro-pipeline with timing and intermediate audit tracking.
        """
        t0 = time.perf_counter()

        # Step 1: Input feature extraction
        feats_input = extract_extended_features(img_bgr)
        t_feats = (time.perf_counter() - t0) * 1000.0

        # Step 2: Stage 1 Routing & Execution
        t1 = time.perf_counter()
        op1_name, op1_reason = self.diagnose_stage1(feats_input)
        op1_fn = STAGE1_OPERATORS.get(op1_name, identity)
        conditioned = op1_fn(img_bgr)
        t_stage1 = (time.perf_counter() - t1) * 1000.0

        # Step 3: Stage 2 Routing & Execution
        t2 = time.perf_counter()
        op2_name, exp_psnr = self.select_stage2(conditioned)
        op2_fn = STAGE2_OPERATORS.get(op2_name, gamma)
        enhanced = op2_fn(conditioned)
        t_stage2 = (time.perf_counter() - t2) * 1000.0

        total_latency = (time.perf_counter() - t0) * 1000.0

        return {
            "stage1_operator": op1_name,
            "stage1_reason": op1_reason,
            "stage2_operator": op2_name,
            "expected_psnr": exp_psnr,
            "conditioned_image": conditioned,
            "enhanced_image": enhanced,
            "pipeline_name": f"{op1_name} -> {op2_name}",
            "latency_ms": {
                "features": round(t_feats, 2),
                "stage1": round(t_stage1, 2),
                "stage2": round(t_stage2, 2),
                "total": round(total_latency, 2),
            },
        }


def benchmark_micro_pipelines(
    eval_dir: str = "data/lol_dataset/eval15/high",
    output_csv: str = "results/phase2/micro_pipeline_benchmark.csv",
) -> pd.DataFrame:
    """
    Benchmarks monolithic single-operator baselines against adaptive 2-stage micro-pipelines
    on compound multi-hazard degradations (Low-Light + Noise, Low-Light + Haze, Low-Light + Blur).
    """
    print("\n" + "=" * 70)
    print("PHASE 2 MILESTONE 2.3: ADAPTIVE 2-STAGE MICRO-PIPELINE BENCHMARK")
    print("=" * 70)

    clean_files = sorted([os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".png")])
    if len(clean_files) == 0:
        raise FileNotFoundError(f"No clean scenes found in {eval_dir}")

    pipeline = AdaptiveMicroPipeline()
    records = []

    print(f"Benchmarking on {len(clean_files)} eval scenes under 3 compound hazard regimes...")

    for path in clean_files:
        scene = os.path.basename(path)
        clean = cv2.imread(path)
        clean_f = clean.astype(np.float32) / 255.0
        h, w, _ = clean.shape

        # Regime 1: Compound Low-Light + Heavy Poisson-Gaussian Noise
        dark1 = (clean_f ** 2.2) * 0.38
        noise1 = np.random.normal(0, 9.0 / 255.0, dark1.shape).astype(np.float32)
        compound_noise = np.clip((dark1 + noise1) * 255.0, 0, 255).astype(np.uint8)

        # Regime 2: Compound Low-Light + Atmospheric Scattering (Veiling Haze)
        y_coords, _ = np.mgrid[0:h, 0:w]
        depth = 0.5 + 0.5 * (1.0 - y_coords / float(h))
        t = np.exp(-1.3 * depth)[:, :, np.newaxis]
        A = np.array([0.82, 0.82, 0.82], dtype=np.float32)
        dark2 = (clean_f ** 2.0) * 0.42
        compound_haze = np.clip((dark2 * t + A * (1.0 - t)) * 255.0, 0, 255).astype(np.uint8)

        # Regime 3: Compound Low-Light + Optical Defocus Blur
        dark3 = np.clip(((clean_f ** 2.2) * 0.40) * 255.0, 0, 255).astype(np.uint8)
        compound_blur = cv2.GaussianBlur(dark3, (0, 0), sigmaX=3.0, sigmaY=3.0)

        regimes = [
            ("Compound Low-Light + Noise", compound_noise),
            ("Compound Low-Light + Haze", compound_haze),
            ("Compound Low-Light + Blur", compound_blur),
        ]

        for reg_name, comp_img in regimes:
            # 1. Monolithic Single Enhancer Baselines
            single_scores = {}
            for name, fn in STAGE2_OPERATORS.items():
                enh = fn(comp_img)
                single_scores[name] = compute_psnr(clean, enh)
            # Add standalone high-boost and dcp
            single_scores["high_boost_sharpen"] = compute_psnr(clean, high_boost_sharpen(comp_img))
            single_scores["dcp_dehaze"] = compute_psnr(clean, dcp_dehaze(comp_img))

            best_single_name = max(single_scores, key=single_scores.get)
            best_single_psnr = single_scores[best_single_name]
            retinex_psnr = single_scores["retinex"]
            gamma_psnr = single_scores["gamma"]

            # 2. Adaptive Two-Stage Micro-Pipeline
            res = pipeline.process(comp_img)
            cascade_enh = res["enhanced_image"]
            cascade_psnr = compute_psnr(clean, cascade_enh)
            total_lat = res["latency_ms"]["total"]

            gain_over_single = cascade_psnr - best_single_psnr
            gain_over_retinex = cascade_psnr - retinex_psnr

            records.append({
                "scene": scene,
                "hazard_regime": reg_name,
                "stage1_operator": res["stage1_operator"],
                "stage2_operator": res["stage2_operator"],
                "micro_pipeline": res["pipeline_name"],
                "cascade_psnr": round(cascade_psnr, 2),
                "best_single_operator": best_single_name,
                "best_single_psnr": round(best_single_psnr, 2),
                "retinex_psnr": round(retinex_psnr, 2),
                "gamma_psnr": round(gamma_psnr, 2),
                "gain_over_best_single_dB": round(gain_over_single, 3),
                "gain_over_retinex_dB": round(gain_over_retinex, 3),
                "latency_ms": round(total_lat, 2),
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Benchmark results saved to {output_csv}")

    # Summary table
    summary = df.groupby("hazard_regime").agg(
        Cascade_Mean_PSNR=("cascade_psnr", "mean"),
        Best_Single_PSNR=("best_single_psnr", "mean"),
        Fixed_Retinex_PSNR=("retinex_psnr", "mean"),
        Net_Gain_Over_Single_dB=("gain_over_best_single_dB", "mean"),
        Net_Gain_Over_Retinex_dB=("gain_over_retinex_dB", "mean"),
        Mean_Latency_ms=("latency_ms", "mean"),
    ).round(2).reset_index()

    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY: ADAPTIVE 2-STAGE MICRO-PIPELINE vs. MONOLITHIC BASELINES")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

    return df


if __name__ == "__main__":
    benchmark_micro_pipelines()
