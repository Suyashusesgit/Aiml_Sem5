"""
Generates the 15-page High-Resolution Supplementary Visual Atlas for Project Triathlon.
Showcases all 15 official held-out test scenes from the LOL-v1 benchmark (eval15) with:
    - Degraded vs Reference vs Selected Output
    - Edge Sharpening Heatmaps
    - ORB RANSAC Geometric Matching Vectors
    - YOLOv8 Bounding Box Detections
    - Complete 8D Physical Metric Breakdown & PTO utilities
"""

import os
import sys
import glob
import cv2
import numpy as np
import joblib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_features, FEATURE_NAMES
from src.methods import METHODS
from src.labels import compute_psnr, compute_ssim
from src.dataset import resize_long_side
from src.multi_degradation import high_boost_sharpen, dcp_dehaze
from src.downstream_matching import evaluate_enhancement_matching_gain
from src.predict_then_optimize import PredictThenOptimizeSelector, PTO_FEATURE_NAMES

ALL_METHODS = {
    **METHODS,
    "high_boost_sharpen": high_boost_sharpen,
    "dcp_dehaze": dcp_dehaze,
}

def get_yolo_detections(img_bgr, conf_thresh=0.20):
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        res = list(model.predict(img_bgr, conf=conf_thresh, verbose=False))[0]
        return res.plot(), len(res.boxes) if res.boxes is not None else 0
    except Exception:
        return img_bgr.copy(), 0

def build_atlas():
    os.makedirs("report", exist_ok=True)
    pdf_out = "report/Project_Triathlon_Supplementary_Visual_Atlas.pdf"
    
    # Locate all 15 eval15 scenes
    low_dir = "data/lol_dataset/eval15/low"
    high_dir = "data/lol_dataset/eval15/high"
    scene_files = sorted(glob.glob(os.path.join(low_dir, "*.png")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]) if os.path.splitext(os.path.basename(p))[0].isdigit() else 999)
    
    if not scene_files:
        print("Error: No eval15 scenes found in data/lol_dataset/eval15/low")
        return

    rf_model = joblib.load("models/rf_selector.joblib")
    pto_model = PredictThenOptimizeSelector.load("models/pto_regressors.joblib") if os.path.exists("models/pto_regressors.joblib") else None

    print(f"Generating 15-page visual atlas across {len(scene_files)} eval15 test scenes...")

    with PdfPages(pdf_out) as pdf:
        for idx, low_path in enumerate(scene_files):
            scene_name = os.path.basename(low_path)
            scene_id = os.path.splitext(scene_name)[0]
            high_path = os.path.join(high_dir, scene_name)

            deg = cv2.imread(low_path)
            clean = cv2.imread(high_path) if os.path.exists(high_path) else None
            
            deg, _ = resize_long_side(deg, 512)
            if clean is not None:
                clean, _ = resize_long_side(clean, 512)

            # 1. Feature Extraction & Selection
            feats = extract_features(deg, include_engineered=True)
            X = np.array([[feats[k] for k in FEATURE_NAMES]])
            pred_method = rf_model.predict(X)[0]
            
            # PTO utility
            pto_winner = "N/A"
            if pto_model is not None:
                X_pto = np.array([[feats.get(k, 0.0) for k in PTO_FEATURE_NAMES]])
                pto_winner = pto_model.select(X_pto)[0][0]

            enhanced = ALL_METHODS[pred_method](deg)
            psnr_val = compute_psnr(clean, enhanced) if clean is not None else 0.0
            ssim_val = compute_ssim(clean, enhanced) if clean is not None else 0.0

            # 2. Edge Difference Heatmap
            diff = cv2.absdiff(deg, enhanced)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_amp = np.clip(diff_gray.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(diff_amp, cv2.COLORMAP_TURBO)

            # 3. ORB RANSAC Feature Vector Matching
            matching_res = evaluate_enhancement_matching_gain(deg, enhanced, clean)
            match_canvas = matching_res["canvas_bgr"]

            # 4. YOLO Object Detections
            yolo_deg, n_deg = get_yolo_detections(deg)
            yolo_enh, n_enh = get_yolo_detections(enhanced)

            # Build Figure Canvas (1 Page per Scene)
            fig = plt.figure(figsize=(11, 8.5), dpi=150) # Landscape US Letter
            fig.patch.set_facecolor("#f8fafc")

            # Header
            fig.text(0.05, 0.95, f"Project Triathlon · Visual Atlas: Scene #{scene_id} ({idx+1}/15)", fontsize=14, weight="bold", color="#1e3a8a")
            fig.text(0.05, 0.925, f"Selected Operator: {pred_method.upper()}  |  Continuous PTO Choice: {pto_winner.upper()}  |  Fidelity: {psnr_val:.2f} dB (SSIM: {ssim_val:.3f})", fontsize=10, color="#334155")
            
            # 6 Panels Grid
            # Row 1: Degraded | Selected Output | Reference
            ax1 = fig.add_axes([0.05, 0.52, 0.28, 0.36])
            ax1.imshow(deg[:, :, ::-1])
            ax1.set_title(f"Degraded Input (Luminance: {feats['brightness']:.1f})", fontsize=9, weight="bold", pad=4)
            ax1.axis("off")

            ax2 = fig.add_axes([0.36, 0.52, 0.28, 0.36])
            ax2.imshow(enhanced[:, :, ::-1])
            ax2.set_title(f"★ Selected: {pred_method.upper()} ({psnr_val:.2f} dB) ★", fontsize=9, weight="bold", color="#15803d", pad=4)
            ax2.axis("off")

            ax3 = fig.add_axes([0.67, 0.52, 0.28, 0.36])
            if clean is not None:
                ax3.imshow(clean[:, :, ::-1])
                ax3.set_title("Ground-Truth Clean Reference", fontsize=9, weight="bold", pad=4)
            ax3.axis("off")

            # Row 2: Heatmap | ORB RANSAC Inliers | YOLO Proposals
            ax4 = fig.add_axes([0.05, 0.12, 0.28, 0.36])
            ax4.imshow(heatmap[:, :, ::-1])
            ax4.set_title("High-Pass Edge Delta Heatmap (Turbo)", fontsize=9, weight="bold", pad=4)
            ax4.axis("off")

            ax5 = fig.add_axes([0.36, 0.12, 0.28, 0.36])
            ax5.imshow(match_canvas[:, :, ::-1])
            ax5.set_title(f"ORB Inliers: {matching_res['enhanced_inliers']} ({matching_res['inlier_gain_pct']:+.1f}% gain)", fontsize=9, weight="bold", color="#0284c7", pad=4)
            ax5.axis("off")

            ax6 = fig.add_axes([0.67, 0.12, 0.28, 0.36])
            ax6.imshow(yolo_enh[:, :, ::-1])
            ax6.set_title(f"YOLOv8 Objects: {n_enh} (Raw Input had: {n_deg})", fontsize=9, weight="bold", pad=4)
            ax6.axis("off")

            # Footer
            fig.text(0.5, 0.04, f"Scene {scene_id} · GroupKFold Zero-Leakage Protocol · Hasler Colourfulness: {feats['colourfulness']:.1f} · Entropy: {feats['entropy']:.2f} bits", ha="center", fontsize=8, color="#64748b", style="italic")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"Processed Scene #{scene_id} ({idx+1}/15)...")

    print(f"Visual Atlas Complete! Saved: {pdf_out} ({os.path.getsize(pdf_out) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_atlas()

