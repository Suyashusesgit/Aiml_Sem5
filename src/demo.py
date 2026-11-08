"""
Standalone Demo Script for Image Enhancement and Per-Image Selection
Team Triathlon: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan
Supervisor: Prof. Kishor P. Upla, SVNIT Surat

Demonstrates end-to-end inference on a single test image:
1. Loads and preprocesses degraded image (and optional clean reference)
2. Extracts 8 handcrafted physical degradation features in < 2 ms
3. Uses trained Random Forest selector to predict optimal enhancer
4. Executes the selected enhancement method
5. Evaluates against ground truth (PSNR, SSIM, Decision Regret) if reference is provided
6. Generates and saves publication-quality comparison figures
"""

import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import argparse
from typing import Any
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.labels import compute_psnr, compute_ssim, score_pair
from src.dataset import resize_long_side, resolve_dataset_dir


def get_or_train_selector(
    features_csv: str = "results/features.csv",
    scores_csv: str = "results/scores.csv",
    model_save_path: str = "models/rf_selector.joblib",
    random_seed: int = 42,
) -> Any:
    """
    Loads pre-trained Random Forest selector or trains and caches one if not found.
    """
    if os.path.exists(model_save_path):
        return joblib.load(model_save_path)

    print(f"Model cache not found at {model_save_path}. Training Random Forest on cached features...")
    if not os.path.exists(features_csv) or not os.path.exists(scores_csv):
        raise FileNotFoundError(
            f"Required cache files ({features_csv}, {scores_csv}) missing. "
            "Run 'python3 main.py' first to generate dataset cache."
        )

    df_features = pd.read_csv(features_csv)
    df_scores = pd.read_csv(scores_csv)

    X = df_features[FEATURE_NAMES].values
    y = df_scores["label"].values

    rf_pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1)
    )
    rf_pipe.fit(X, y)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf_pipe, model_save_path)
    print(f"Saved trained Random Forest model to {model_save_path}.")
    return rf_pipe


def run_demo(
    input_path: str,
    reference_path: str = None,
    output_dir: str = "results/demo_output",
    model_path: str = "models/rf_selector.joblib",
    target_long_side: int = 512,
    save_grid: bool = True,
) -> None:
    """Executes the full pipeline on a single image instance."""
    # Resolve input path if relative to dataset
    if not os.path.exists(input_path):
        base_dir = resolve_dataset_dir()
        alt_candidates = [
            os.path.join(base_dir, "eval15", "low", os.path.basename(input_path)),
            os.path.join(base_dir, "our485", "low", os.path.basename(input_path)),
            os.path.join(base_dir, input_path),
        ]
        for alt in alt_candidates:
            if os.path.exists(alt):
                input_path = alt
                break

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and Preprocess Input Image
    print("\n" + "=" * 70)
    print("PROJECT TRIATHLON · IMAGE ENHANCEMENT DEMO")
    print("=" * 70)
    print(f"Input Image : {input_path}")

    deg_raw = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if deg_raw is None:
        raise ValueError(f"Failed to read image at {input_path}")

    deg, scale = resize_long_side(deg_raw, target_size=target_long_side)
    print(f"Dimensions  : {deg_raw.shape[1]}x{deg_raw.shape[0]} -> Resized to {deg.shape[1]}x{deg.shape[0]} (scale={scale:.3f})")

    clean = None
    if reference_path is None:
        # Automatically detect matching reference pair if available
        cand_ref = input_path.replace("/low/", "/high/").replace("\\low\\", "\\high\\")
        if cand_ref != input_path and os.path.exists(cand_ref):
            reference_path = cand_ref
            print(f"Auto-detected reference pair at: {reference_path}")

    if reference_path:
        if not os.path.exists(reference_path):
            print(f"Warning: Reference image not found at {reference_path}. Proceeding reference-free.")
        else:
            clean_raw = cv2.imread(reference_path, cv2.IMREAD_COLOR)
            clean, _ = resize_long_side(clean_raw, target_size=target_long_side)
            print(f"Reference   : {reference_path} (Loaded clean ground truth)")

    # 2. Handcrafted Physical Feature Extraction (strictly on degraded image)
    t0_feat = time.perf_counter()
    feats = extract_features(deg, include_engineered=True)
    t_feat_ms = (time.perf_counter() - t0_feat) * 1000.0

    print("\n--- Extracted Handcrafted Physical Statistics (computed in {:.2f} ms) ---".format(t_feat_ms))
    for k in FEATURE_NAMES:
        print(f"  • {k:<15}: {feats[k]:.3f}")

    # 3. Model Prediction
    selector = get_or_train_selector(model_save_path=model_path)
    X_sample = np.array([[feats[k] for k in FEATURE_NAMES]])

    t0_pred = time.perf_counter()
    pred_method = selector.predict(X_sample)[0]
    pred_probs = selector.predict_proba(X_sample)[0]
    t_pred_ms = (time.perf_counter() - t0_pred) * 1000.0

    print(f"\nSelector Decision : '{pred_method}' (Inference: {t_pred_ms:.2f} ms)")
    classes = selector.classes_
    prob_dict = {cls: prob for cls, prob in zip(classes, pred_probs)}
    for cls in sorted(prob_dict, key=prob_dict.get, reverse=True):
        print(f"  - {cls:<15}: {prob_dict[cls] * 100.0:5.2f}% confidence")

    # 4. Enhancement Execution
    enhancer_fn = METHODS[pred_method]
    t0_enh = time.perf_counter()
    enhanced_img = enhancer_fn(deg)
    t_enh_ms = (time.perf_counter() - t0_enh) * 1000.0
    print(f"Enhancement Done  : {t_enh_ms:.2f} ms using '{pred_method}'")

    # Save Enhanced Result
    enhanced_out_path = os.path.join(output_dir, f"enhanced_{pred_method}.png")
    cv2.imwrite(enhanced_out_path, enhanced_img)
    print(f"Saved Output      : {enhanced_out_path}")

    # 5. Full-Reference Scoring (if ground-truth clean image provided)
    oracle_method = None
    regret_db = None
    scores_all = {}

    if clean is not None:
        print("\n--- Quality Scoring against Reference ---")
        for m_name, m_fn in METHODS.items():
            cand_img = m_fn(deg)
            p = compute_psnr(clean, cand_img)
            s = compute_ssim(clean, cand_img)
            scores_all[m_name] = {"psnr": p, "ssim": s, "img": cand_img}

        oracle_res = score_pair(clean, deg)
        oracle_method = oracle_res["label"]
        oracle_psnr = oracle_res["oracle_psnr"]
        selected_psnr = scores_all[pred_method]["psnr"]
        regret_db = max(0.0, oracle_psnr - selected_psnr)

        print(f"  • Oracle Optimal Method : '{oracle_method}' (PSNR: {oracle_psnr:.2f} dB)")
        print(f"  • Selected Model Choice : '{pred_method}' (PSNR: {selected_psnr:.2f} dB, SSIM: {scores_all[pred_method]['ssim']:.3f})")
        print(f"  • Decision Regret       : {regret_db:.3f} dB")
        if pred_method == oracle_method:
            print("  >>> TOP-1 EXACT MATCH: Zero regret!")
        else:
            print(f"  >>> Sub-optimal match: regret is {regret_db:.3f} dB")

    # 6. Generate Visual Comparison Figure
    if save_grid:
        fig_path = os.path.join(output_dir, "enhancement_comparison_grid.png")
        if clean is not None:
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes = axes.flatten()

            # Input
            axes[0].imshow(cv2.cvtColor(deg, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Degraded Input (Low-Light)", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            # Reference
            axes[1].imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Ground Truth Reference", fontsize=11, fontweight="bold", color="darkgreen")
            axes[1].axis("off")

            # Six Candidates
            for idx, (m_name, m_fn) in enumerate(METHODS.items()):
                cand_data = scores_all[m_name]
                ax = axes[idx + 2]
                ax.imshow(cv2.cvtColor(cand_data["img"], cv2.COLOR_BGR2RGB))
                is_selected = (m_name == pred_method)
                is_oracle = (m_name == oracle_method)

                title_tag = ""
                box_color = "black"
                if is_selected and is_oracle:
                    title_tag = "\n★ SELECTED (ORACLE WINNER) ★"
                    box_color = "green"
                elif is_selected:
                    title_tag = f"\n★ SELECTED (Regret: {regret_db:.2f} dB) ★"
                    box_color = "blue"
                elif is_oracle:
                    title_tag = "\n[ORACLE WINNER]"
                    box_color = "darkorange"

                ax.set_title(
                    f"{m_name}\nPSNR: {cand_data['psnr']:.2f} dB | SSIM: {cand_data['ssim']:.3f}{title_tag}",
                    fontsize=10,
                    color=box_color,
                    fontweight="bold" if (is_selected or is_oracle) else "normal"
                )
                ax.axis("off")

            plt.suptitle(
                f"Project Triathlon Algorithm Selection Demo\nPredicted: {pred_method} | Oracle: {oracle_method} | Regret: {regret_db:.3f} dB",
                fontsize=13,
                fontweight="bold"
            )
        else:
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes = axes.flatten()

            axes[0].imshow(cv2.cvtColor(deg, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Degraded Input", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"★ SELECTED: {pred_method} ★", fontsize=11, fontweight="bold", color="blue")
            axes[1].axis("off")

            for idx, (m_name, m_fn) in enumerate(METHODS.items()):
                out = m_fn(deg)
                ax = axes[idx + 2]
                ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{m_name}" + (" (Selected)" if m_name == pred_method else ""), fontsize=10)
                ax.axis("off")

            plt.suptitle(
                f"Project Triathlon Algorithm Selection Demo (Reference-Free)\nSelected Method: {pred_method}",
                fontsize=13,
                fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Grid Image  : {fig_path}")

    print("=" * 70)
    print("Demo execution successfully completed!\n")


def main():
    default_input = "/Users/home/Desktop/LOLdataset/eval15/low/1.png"
    if not os.path.exists(default_input):
        default_input = "data/lol_dataset/eval15/low/1.png"

    parser = argparse.ArgumentParser(
        description="Project Triathlon: Fidelity-Driven Per-Image Enhancement Selector Demo"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=default_input,
        help="Path to degraded low-light input image."
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to clean ground truth image (optional; auto-detected if matching pair exists)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/demo_output",
        help="Directory to save enhanced output and visual comparison grid."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/rf_selector.joblib",
        help="Path to pre-trained selector model."
    )
    args = parser.parse_args()

    run_demo(
        input_path=args.input,
        reference_path=args.reference,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
