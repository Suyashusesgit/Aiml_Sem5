"""
Downstream Object Detection Benchmark with YOLOv8 (Phase 1)
Authors: Team Triathlon (Ananya Kasaudhan, Suyash Srivastav, Logi Vasan)
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Evaluates whether fidelity-driven algorithm selection translates directly into
superior high-level perception (object detection confidence, box count, and recall)
using YOLOv8 across:
1. Raw Low-Light (Degraded baseline)
2. Best Fixed Method (Single-Scale Retinex)
3. Proposed Algorithm Selector (Random Forest)
4. Clean Reference (Ground Truth Oracle)
"""

import os
import sys
import time
from typing import Dict, List, Any
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure package root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES


def run_yolo_downstream_eval(
    eval15_low_dir: str = "data/lol_dataset/eval15/low",
    eval15_high_dir: str = "data/lol_dataset/eval15/high",
    model_path: str = "models/rf_selector.joblib",
    output_csv: str = "results/downstream_yolo_metrics.csv",
    figure_path: str = "results/figures/fig11_downstream_yolo_detection.png"
) -> pd.DataFrame:
    print("\n" + "="*70)
    print("PHASE 1: Downstream Object Detection Benchmark (YOLOv8 on Low-Light)")
    print("="*70)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package is required. Install via pip install ultralytics.")

    print("Loading pretrained YOLOv8n detector (COCO 80 classes)...")
    detector = YOLO("yolov8n.pt")

    if not os.path.exists(eval15_low_dir):
        eval15_low_dir = "/Users/home/Desktop/LOLdataset/eval15/low"
        eval15_high_dir = "/Users/home/Desktop/LOLdataset/eval15/high"

    low_files = sorted([os.path.join(eval15_low_dir, f) for f in os.listdir(eval15_low_dir) if f.endswith(".png")])
    if len(low_files) == 0:
        raise FileNotFoundError(f"No PNG files in {eval15_low_dir}")

    rf_model = joblib.load(model_path)
    print(f"Evaluating {len(low_files)} official held-out test scenes...")

    results = []
    qualitative_samples = []

    for low_p in low_files:
        base_name = os.path.basename(low_p)
        high_p = os.path.join(eval15_high_dir, base_name)
        if not os.path.exists(high_p):
            continue

        raw_bgr = cv2.imread(low_p)
        ref_bgr = cv2.imread(high_p)

        # 1. Selector prediction
        feats = extract_features(raw_bgr)
        X = np.array([[feats[k] for k in FEATURE_NAMES]], dtype=np.float32)
        sel_method = rf_model.predict(X)[0]

        # 2. Restorations
        retinex_bgr = METHODS["retinex"](raw_bgr)
        selector_bgr = METHODS[sel_method](raw_bgr)

        # 3. YOLO Detections across all 4 images
        res_raw = detector(raw_bgr, conf=0.15, verbose=False)[0]
        res_retinex = detector(retinex_bgr, conf=0.15, verbose=False)[0]
        res_selector = detector(selector_bgr, conf=0.15, verbose=False)[0]
        res_ref = detector(ref_bgr, conf=0.15, verbose=False)[0]

        n_raw = len(res_raw.boxes)
        n_ret = len(res_retinex.boxes)
        n_sel = len(res_selector.boxes)
        n_ref = max(1, len(res_ref.boxes))

        conf_raw = float(res_raw.boxes.conf.mean()) if n_raw > 0 else 0.0
        conf_ret = float(res_retinex.boxes.conf.mean()) if n_ret > 0 else 0.0
        conf_sel = float(res_selector.boxes.conf.mean()) if n_sel > 0 else 0.0
        conf_ref = float(res_ref.boxes.conf.mean()) if len(res_ref.boxes) > 0 else 0.0

        results.append({
            "image": base_name,
            "selected_method": sel_method,
            "raw_boxes": n_raw,
            "retinex_boxes": n_ret,
            "selector_boxes": n_sel,
            "ref_boxes": len(res_ref.boxes),
            "raw_conf": round(conf_raw, 3),
            "retinex_conf": round(conf_ret, 3),
            "selector_conf": round(conf_sel, 3),
            "ref_conf": round(conf_ref, 3),
            "recall_raw_vs_ref": round(min(1.0, n_raw / n_ref) * 100.0, 1),
            "recall_retinex_vs_ref": round(min(1.0, n_ret / n_ref) * 100.0, 1),
            "recall_selector_vs_ref": round(min(1.0, n_sel / n_ref) * 100.0, 1),
        })

        if base_name in ["1.png", "22.png", "79.png"]:
            qualitative_samples.append((base_name, raw_bgr, retinex_bgr, selector_bgr, ref_bgr, res_raw, res_retinex, res_selector, res_ref))

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved YOLO Downstream Benchmark to {output_csv}")

    print("\n" + "="*80)
    print("PHASE 1 SUMMARY: DOWNSTREAM YOLOv8 OBJECT DETECTION BENCHMARK")
    print("="*80)
    print(f"Total Detected Bounding Boxes across 15 test scenes:")
    print(f"  - Raw Underexposed Input:      {df['raw_boxes'].sum()} boxes (Mean Conf: {df['raw_conf'].mean():.2f})")
    print(f"  - Best Fixed Single (Retinex): {df['retinex_boxes'].sum()} boxes (Mean Conf: {df['retinex_conf'].mean():.2f})")
    print(f"  - Proposed Selector:          {df['selector_boxes'].sum()} boxes (Mean Conf: {df['selector_conf'].mean():.2f})")
    print(f"  - Ground Truth Reference:     {df['ref_boxes'].sum()} boxes (Mean Conf: {df['ref_conf'].mean():.2f})")
    print(f"\nMean Object Detection Recall vs Ground Truth Reference:")
    print(f"  - Raw Underexposed:           {df['recall_raw_vs_ref'].mean():.1f}%")
    print(f"  - Best Fixed Single:          {df['recall_retinex_vs_ref'].mean():.1f}%")
    print(f"  - Proposed Selector:          {df['recall_selector_vs_ref'].mean():.1f}% (+{df['recall_selector_vs_ref'].mean() - df['recall_raw_vs_ref'].mean():.1f}% gain!)")

    # Generate Figure 11: YOLO Detection Metrics & Visual Detection Comparison
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    categories = ["Raw Low-Light", "Fixed Retinex", "Proposed Selector", "Reference (Oracle)"]
    box_totals = [df["raw_boxes"].sum(), df["retinex_boxes"].sum(), df["selector_boxes"].sum(), df["ref_boxes"].sum()]
    colors = ["#94a3b8", "#f59e0b", "#16a34a", "#2563eb"]

    # Subplot 1: Total Bounding Boxes
    bars1 = ax1.bar(categories, box_totals, color=colors, alpha=0.88)
    ax1.set_title("(a) Total Bounding Boxes Detected (YOLOv8n)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Total Detections across eval15 split", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: Detection Recall vs Reference
    recalls = [df["recall_raw_vs_ref"].mean(), df["recall_retinex_vs_ref"].mean(), df["recall_selector_vs_ref"].mean(), 100.0]
    bars2 = ax2.bar(categories, recalls, color=colors, alpha=0.88)
    ax2.set_title("(b) Object Detection Recall Relative to Reference (%)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Recall Relative to Reference (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Phase 1: Downstream Object Detection Performance with YOLOv8", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 11 to {figure_path}")

    return df


if __name__ == "__main__":
    run_yolo_downstream_eval()

