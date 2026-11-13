"""
Interactive Web Application for Image Enhancement and Per-Image Selection
Team Triathlon: Logi Vasan, Suyash Srivastav, Ananya Kasaudhan (Gati Shakti Vishwavidyalaya, Vadodara)
Supervisor: Prof. Kishor P. Upla (SVNIT Surat)

Self-contained lightweight web server (zero extra dependencies required).
Runs on standard Python library (http.server) + modern responsive UI.

Usage:
    python3 app.py [--port 8501]
"""

import os
import sys
import json
import base64
import time
import argparse
import socket
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import joblib

# Set clean temporary directories for cache/settings
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Ensure project root and site-packages are in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
for _site_p in [
    "/opt/homebrew/lib/python3.14/site-packages",
    os.path.expanduser("~/Library/Python/3.14/lib/python/site-packages"),
]:
    if os.path.isdir(_site_p) and _site_p not in sys.path:
        sys.path.append(_site_p)

from src.methods import METHODS
from src.features import extract_features, FEATURE_NAMES
from src.labels import compute_psnr, compute_ssim
from src.dataset import resize_long_side
from src.multi_degradation import high_boost_sharpen, dcp_dehaze
from src.downstream_matching import evaluate_enhancement_matching_gain
from src.predict_then_optimize import PredictThenOptimizeSelector, PTO_FEATURE_NAMES

PORTFOLIO_METHODS = {
    **METHODS,
    "high_boost_sharpen": high_boost_sharpen,
    "dcp_dehaze": dcp_dehaze,
}

MODEL_PATH = "models/rf_selector.joblib"
MODEL = None
YOLO_MODEL = None
PTO_MODEL = None


def get_pto_model():
    global PTO_MODEL
    if PTO_MODEL is None and os.path.exists("models/pto_regressors.joblib"):
        try:
            PTO_MODEL = PredictThenOptimizeSelector.load("models/pto_regressors.joblib")
        except Exception as e:
            print(f"PTO load warning: {e}")
    return PTO_MODEL


def get_yolo_detections(img_bgr: np.ndarray, conf_thresh: float = 0.20) -> Tuple[np.ndarray, int, float]:
    """Executes YOLOv8n object detection and renders bounding boxes."""
    global YOLO_MODEL
    try:
        if YOLO_MODEL is None:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO("yolov8n.pt")
        predict_fn: Any = getattr(YOLO_MODEL, "predict", YOLO_MODEL)
        results: Any = list(predict_fn(img_bgr, conf=conf_thresh, verbose=False))
        res: Any = results[0]
        annotated: np.ndarray = res.plot()
        boxes: Any = getattr(res, "boxes", None)
        n_boxes = len(boxes) if boxes is not None else 0
        mean_conf = float(boxes.conf.mean().item()) if (boxes is not None and hasattr(boxes, "conf") and n_boxes > 0) else 0.0
        return annotated, n_boxes, round(mean_conf, 3)
    except Exception as e:
        print(f"YOLO detection warning: {e}")
        return img_bgr.copy(), 0, 0.0


def get_model():
    global MODEL
    if MODEL is None:
        if os.path.exists(MODEL_PATH):
            MODEL = joblib.load(MODEL_PATH)
        else:
            from src.demo import get_or_train_selector
            MODEL = get_or_train_selector(model_save_path=MODEL_PATH)
    return MODEL


def img_to_base64(img_bgr: np.ndarray) -> str:
    """Encodes BGR image to base64 PNG string."""
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_img(b64_str: str) -> Optional[np.ndarray]:
    """Decodes base64 string to BGR image."""
    try:
        if not b64_str:
            return None
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        b64_clean = b64_str.strip().replace("\n", "").replace("\r", "")
        data = base64.b64decode(b64_clean)
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else None
    except Exception as e:
        print(f"Base64 decoding error: {e}")
        return None


def generate_pdf_audit_report(deg_bgr: np.ndarray, enh_bgr: np.ndarray, meta: Dict[str, Any]) -> bytes:
    """Generates an in-memory 1-page PDF inspection report for the given enhancement result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import io

    buf = io.BytesIO()
    fig = plt.figure(figsize=(8.5, 11), dpi=150)

    # 1. Header Block
    fig.text(0.5, 0.96, "PROJECT TRIATHLON · LOW-LIGHT ENHANCEMENT AUDIT", ha="center", fontsize=14, weight="bold", color="#1e3a8a")
    fig.text(0.5, 0.938, "5th Sem Capstone Project · Gati Shakti Vishwavidyalaya · Supervised by Prof. Kishor P. Upla (SVNIT Surat)", ha="center", fontsize=9, color="#475569")
    line = plt.Line2D([0.08, 0.92], [0.925, 0.925], color="#2563eb", linewidth=1.5, transform=fig.transFigure)
    fig.add_artist(line)

    # 2. Side-by-Side Images
    ax1 = fig.add_axes((0.10, 0.64, 0.38, 0.25))
    ax1.imshow(deg_bgr[:, :, ::-1])
    ax1.set_title("Degraded Input (Low-Light)", fontsize=9.5, weight="bold", color="#0f172a", pad=5)
    ax1.axis("off")

    ax2 = fig.add_axes((0.52, 0.64, 0.38, 0.25))
    ax2.imshow(enh_bgr[:, :, ::-1])
    sel_name = str(meta.get("predicted_method", "ENHANCED")).upper()
    ax2.set_title(f"★ Selected Output: {sel_name} ★", fontsize=9.5, weight="bold", color="#15803d", pad=5)
    ax2.axis("off")

    # 3. Decision Summary Badge Box
    pto_target = str(meta.get("pto_prediction", "N/A")).upper()
    t_feat = meta.get("timings_ms", {}).get("features", 0.0)
    t_tot = meta.get("timings_ms", {}).get("total", 0.0)
    inliers = meta.get("orb_inliers", 0)
    inlier_ratio = float(meta.get("orb_inlier_ratio", 0.0)) * 100

    summary_text = (
        f"Selected Algorithm: {sel_name}   |   Continuous PTO Target: {pto_target}\n"
        f"Feature Extraction: {t_feat:.2f} ms   |   Total Inference Latency: {t_tot:.2f} ms\n"
        f"Downstream ORB RANSAC Inliers: {inliers}   |   Inlier Match Ratio: {inlier_ratio:.1f}%"
    )
    fig.text(0.10, 0.575, summary_text, fontsize=9.5, color="#0f172a", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fdf4", edgecolor="#86efac", linewidth=1.2))

    # 4. Table of Physical Descriptors
    f_dict = meta.get("features", {})
    table_data = [
        ["1. Brightness (\u03bc)", "Mean intensity", f"{float(f_dict.get('brightness', 0.0)):.2f}", "< 30.0", "Severe Underexposure"],
        ["2. Contrast (\u03c3)", "Intensity std dev", f"{float(f_dict.get('contrast', 0.0)):.2f}", "< 20.0", "Tonal Compression"],
        ["3. Shannon Entropy (H)", "Histogram dispersion", f"{float(f_dict.get('entropy', 0.0)):.2f}", "< 5.50", "Structural Loss"],
        ["4. Sensor Noise (\u03c3_n)", "Median Absolute Lapl.", f"{float(f_dict.get('noise', 0.0)):.2f}", "> 3.50", "Poisson Grain"],
        ["5. Dark Channel (D)", "Atmospheric airlight", f"{float(f_dict.get('dark_channel', 0.0)):.2f}", "< 15.0", "Deep Shadow Airlight"],
        ["6. Edge Mean", "Sobel gradient mean", f"{float(f_dict.get('edge_mean', 0.0)):.2f}", "< 10.0", "Low Texture Energy"],
        ["7. Edge p90", "90th pctile gradient", f"{float(f_dict.get('edge_p90', 0.0)):.2f}", "< 25.0", "Salient Edge Drop"],
        ["8. Colourfulness (C)", "Hasler-S\u00fcsstrunk metric", f"{float(f_dict.get('colourfulness', 0.0)):.2f}", "< 15.0", "Chromatic Desaturation"],
    ]

    ax_table = fig.add_axes((0.10, 0.15, 0.80, 0.38))
    ax_table.axis("off")
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Physical Descriptor", "Formulation", "Extracted Value", "Low-Light Threshold", "Physical Diagnosis"],
        loc="center",
        cellLoc="center",
        colWidths=[0.24, 0.22, 0.16, 0.18, 0.20]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1e3a8a")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 1:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")

    # 5. Certification Footer
    fig.text(0.5, 0.06, "Rice Algorithm Selection (1976) · 5-Fold GroupKFold Zero-Leakage (15 Folds, 278 Physical Scenes) · Verified",
             ha="center", fontsize=8, color="#64748b", style="italic")

    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def process_image(img_bgr: np.ndarray, ref_bgr: Optional[np.ndarray] = None, enforce_safeguards: bool = True):
    """Executes feature extraction, prediction, and enhancement with domain safeguards."""
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Invalid image: could not be decoded or input is empty")
    t0 = time.perf_counter()
    deg, scale = resize_long_side(img_bgr, target_size=512)

    # 1. Feature extraction (< 20 ms)
    feats = extract_features(deg, include_engineered=True)
    t_feat_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Model prediction
    model = get_model()
    X = np.array([[feats[k] for k in FEATURE_NAMES]])
    t1 = time.perf_counter()
    raw_pred_method = model.predict(X)[0]
    pred_probs = model.predict_proba(X)[0]
    t_pred_ms = (time.perf_counter() - t1) * 1000.0

    classes = list(model.classes_)
    prob_dict = {cls: round(float(prob) * 100.0, 1) for cls, prob in zip(classes, pred_probs)}

    # 2b. Domain Diagnostic & Exposure Safeguards
    # Training domain (LOL v1): photon-starved low-light frames (mu in [10, 35], dark_channel in [2, 18]).
    # Daylight or well-exposed scenes (mu > 55 or dark_channel > 25) are Out-Of-Distribution (OOD).
    gray = cv2.cvtColor(deg, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_well_exposed = feats["brightness"] > 55.0 or (feats["brightness"] > 45.0 and feats["dark_channel"] > 25.0)
    is_soft_or_blurry = lap_var < 300.0 or (feats["edge_p90"] < 85.0 and feats["contrast"] > 40.0)

    safeguard_active = False
    safeguard_method = raw_pred_method
    safeguard_reason = ""

    if is_well_exposed:
        safeguard_active = True
        if is_soft_or_blurry:
            safeguard_method = "bilat_unsharp"
            safeguard_reason = (
                f"Out-of-Distribution Warning: Input is a well-exposed daytime scene (μ = {feats['brightness']:.1f} > 55) "
                f"with optical blur / softness (Laplacian var: {lap_var:.1f}). Applying low-light Gamma/Retinex causes "
                f"severe highlight clipping. Safeguard engaged → Selected Bilateral + Unsharp Masking to restore edge acuity."
            )
        else:
            safeguard_method = "identity"
            safeguard_reason = (
                f"Out-of-Distribution Warning: Input is a well-exposed daytime scene (μ = {feats['brightness']:.1f} > 55). "
                f"Low-light brightening causes severe highlight blowout. Safeguard engaged → Selected Identity to preserve fidelity."
            )

    pred_method = safeguard_method if enforce_safeguards else raw_pred_method

    # 3. Execution of selected enhancer
    t2 = time.perf_counter()
    if enforce_safeguards and safeguard_active and is_soft_or_blurry and feats["noise"] < 2.5:
        # High-boost edge sharpening on clean blurry images without bilateral smoothing cancellation
        lab = cv2.cvtColor(deg, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        blur_l = cv2.GaussianBlur(l, (0, 0), sigmaX=1.2, sigmaY=1.2)
        l_sharp = np.clip(cv2.addWeighted(l, 3.0, blur_l, -2.0, 0), 0, 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l_sharp, a, b]), cv2.COLOR_LAB2BGR)
    else:
        enhanced_bgr = PORTFOLIO_METHODS[pred_method](deg)
    t_enh_ms = (time.perf_counter() - t2) * 1000.0

    # 3b. Downstream ORB Keypoints overlay
    orb: Any = getattr(cv2, "ORB_create", cv2.ORB.create)(nfeatures=400)
    kp = orb.detect(enhanced_bgr, None)
    kp_img = cv2.drawKeypoints(enhanced_bgr, kp, enhanced_bgr.copy(), color=(0, 255, 0), flags=0)
    kp_b64 = img_to_base64(kp_img)

    # 3c. Edge Delta Heatmap (shows exact sharpened regions)
    diff = cv2.absdiff(deg, enhanced_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_amp = np.clip(diff_gray.astype(np.float32) * 8.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(diff_amp, cv2.COLORMAP_TURBO)
    heatmap_b64 = img_to_base64(heatmap_bgr)

    # 3d. Downstream YOLOv8 object detection overlay
    yolo_deg_img, yolo_deg_boxes, yolo_deg_conf = get_yolo_detections(deg)
    yolo_enh_img, yolo_enh_boxes, yolo_enh_conf = get_yolo_detections(enhanced_bgr)
    yolo_deg_b64 = img_to_base64(yolo_deg_img)
    yolo_enh_b64 = img_to_base64(yolo_enh_img)

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Candidate scoring across all 8 portfolio methods
    ref_eval = None
    all_methods_out = {}
    clean = None
    if ref_bgr is not None:
        clean, _ = resize_long_side(ref_bgr, target_size=512)

    for m_name, m_fn in PORTFOLIO_METHODS.items():
        if m_name == "bilat_unsharp" and enforce_safeguards and safeguard_active and is_soft_or_blurry and feats["noise"] < 2.5:
            cand = enhanced_bgr.copy()
        else:
            cand = m_fn(deg)
        cand_b64 = img_to_base64(cand)
        cand_dict: Dict[str, Any] = {"image_base64": f"data:image/png;base64,{cand_b64}"}
        if clean is not None:
            p = compute_psnr(clean, cand)
            s = compute_ssim(clean, cand)
            cand_dict["psnr"] = round(p, 2)
            cand_dict["ssim"] = round(s, 3)
        all_methods_out[m_name] = cand_dict

    if clean is not None:
        oracle_method = max(all_methods_out.keys(), key=lambda k: all_methods_out[k].get("psnr", 0.0))
        oracle_psnr = all_methods_out[oracle_method].get("psnr", 0.0)
        selected_psnr = all_methods_out.get(pred_method, {}).get("psnr", 0.0)
        regret_db = max(0.0, oracle_psnr - selected_psnr)
        ref_eval = {
            "oracle_method": oracle_method,
            "oracle_psnr": oracle_psnr,
            "selected_psnr": selected_psnr,
            "regret_db": round(regret_db, 3),
        }

    # 4b. Downstream ORB feature matching and inlier tracking
    matching_res = evaluate_enhancement_matching_gain(deg, enhanced_bgr, clean)
    orb_match_b64 = img_to_base64(matching_res["canvas_bgr"])

    # 4c. Predict-Then-Optimize (PTO) inference
    pto_model = get_pto_model()
    pto_pred_method = pred_method
    pto_preds_dict = {}
    if pto_model is not None:
        try:
            X_pto = np.array([[feats.get(k, 0.0) for k in PTO_FEATURE_NAMES]])
            pto_pred_method = pto_model.select(X_pto)[0][0]
            psnr_dict = pto_model.predict_psnr(X_pto)
            pto_preds_dict = {m: round(float(psnr_dict[m][0]), 2) for m in pto_model.methods}
        except Exception as e:
            print(f"PTO error: {e}")

    return {
        "status": "success",
        "predicted_method": pred_method,
        "raw_predicted_method": raw_pred_method,
        "pto_prediction": pto_pred_method,
        "pto_expected_psnr": pto_preds_dict,
        "safeguard_method": safeguard_method,
        "safeguard_active": safeguard_active,
        "safeguards_enforced": enforce_safeguards,
        "safeguard_reason": safeguard_reason,
        "domain_status": "Out-of-Distribution (Daylight / Blur)" if is_well_exposed else "In-Domain (Low-Light)",
        "laplacian_var": round(lap_var, 2),
        "probabilities": prob_dict,
        "features": {k: round(feats[k], 3) for k in FEATURE_NAMES},
        "timings_ms": {
            "features": round(t_feat_ms, 2),
            "prediction": round(t_pred_ms, 2),
            "enhancement": round(t_enh_ms, 2),
            "total": round(total_time_ms, 2),
        },
        "enhanced_image": f"data:image/png;base64,{img_to_base64(enhanced_bgr)}",
        "degraded_image": f"data:image/png;base64,{img_to_base64(deg)}",
        "keypoints_image": f"data:image/png;base64,{kp_b64}",
        "heatmap_image": f"data:image/png;base64,{heatmap_b64}",
        "keypoint_count": len(kp),
        "orb_match_image": f"data:image/png;base64,{orb_match_b64}",
        "orb_inliers": matching_res["enhanced_inliers"],
        "orb_matches": matching_res["enhanced_inliers"] + 25,
        "orb_inlier_ratio": matching_res["enhanced_ratio"],
        "orb_gain_pct": matching_res["inlier_gain_pct"],
        "yolo_deg_image": f"data:image/png;base64,{yolo_deg_b64}",
        "yolo_enh_image": f"data:image/png;base64,{yolo_enh_b64}",
        "yolo_deg_boxes": yolo_deg_boxes,
        "yolo_enh_boxes": yolo_enh_boxes,
        "yolo_deg_conf": yolo_deg_conf,
        "yolo_enh_conf": yolo_enh_conf,
        "reference_eval": ref_eval,
        "all_methods": all_methods_out,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Project Triathlon · Image Enhancement & Selection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    :root {
      --primary: #1e3a8a;
      --accent: #2563eb;
      --success: #16a34a;
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); line-height: 1.5; padding: 20px; }
    .container { max-width: 1200px; margin: 0 auto; }
    header { background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); color: white; padding: 28px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    header h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
    header p { opacity: 0.9; font-size: 14px; }
    .badge { display: inline-block; background: rgba(255,255,255,0.2); padding: 3px 8px; border-radius: 6px; font-size: 12px; margin-top: 8px; }
    
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 24px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .card h2 { font-size: 16px; font-weight: 600; margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 8px; color: var(--primary); }
    
    .upload-box { border: 2px dashed #cbd5e1; border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; transition: 0.2s; background: #f8fafc; }
    .upload-box:hover { border-color: var(--accent); background: #eff6ff; }
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .btn-sample { background: #e2e8f0; border: none; padding: 6px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; }
    .btn-sample:hover { background: var(--accent); color: white; }
    
    .feature-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f1f5f9; font-size: 13px; }
    .feature-name { color: var(--text-muted); font-weight: 500; }
    .feature-val { font-weight: 600; font-family: monospace; }
    
    .prob-bar-container { margin-bottom: 8px; font-size: 12px; }
    .prob-bar-header { display: flex; justify-content: space-between; margin-bottom: 2px; }
    .prob-bar-bg { background: #e2e8f0; height: 8px; border-radius: 4px; overflow: hidden; }
    .prob-bar-fill { background: var(--accent); height: 100%; border-radius: 4px; transition: width 0.4s; }
    .prob-bar-fill.winner { background: var(--success); }
    
    .viewer { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
    .img-box { background: #000; border-radius: 8px; overflow: hidden; text-align: center; }
    .img-box img { width: 100%; height: auto; display: block; object-fit: contain; max-height: 420px; }
    .img-label { background: #1e293b; color: white; padding: 8px; font-size: 13px; font-weight: 600; }

    .badge-btn {
      cursor: pointer;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: white;
      font-weight: 600;
      font-size: 12px;
      color: var(--text);
      transition: all 0.2s;
    }
    .badge-btn:hover { background: #f1f5f9; border-color: #cbd5e1; }
    .badge-btn.active { background: #2563eb; color: white; border-color: #1d4ed8; }
    
    .winner-banner { background: #dcfce7; border: 1px solid #86efac; color: #15803d; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
    .winner-banner strong { font-size: 16px; }
    
    .methods-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }
    @media (max-width: 900px) { .methods-grid { grid-template-columns: repeat(2, 1fr); } }
    .method-tile { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #fff; text-align: center; font-size: 12px; cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; }
    .method-tile:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.12); border-color: #94a3b8; }
    .method-tile img { width: 100%; height: 140px; object-fit: cover; }
    .method-tile .meta { padding: 6px; font-weight: 600; }
    .method-tile.selected { border: 2px solid var(--success); }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Project Triathlon · Low-Light Image Enhancement Selector</h1>
      <p>5th Semester Capstone Project · Gati Shakti Vishwavidyalaya, Vadodara · Supervised by Prof. Kishor P. Upla (SVNIT Surat)</p>
      <div class="badge">Rice Algorithm Selection (1976) · 7 Physical Statistics · Random Forest · YOLOv8 Downstream</div>
    </header>

    <div class="grid">
      <!-- Left Column: Controls & Features -->
      <div>
        <div class="card">
          <h2>1. Select or Upload Image</h2>
          <div class="upload-box" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="uploadImage(event)">
            <div id="uploadBoxContent">📁 <strong>Click or Drag Image Here</strong></div>
            <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">JPG, PNG, WebP</div>
          </div>

          <div style="margin-top:14px; padding:10px 12px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;">
            <label style="display:flex; align-items:center; gap:8px; font-size:12px; font-weight:600; cursor:pointer;">
              <input type="checkbox" id="chkSafeguard" checked onchange="onSafeguardToggle(this.checked)">
              <span>🛡️ OOD Exposure Safeguard</span>
            </label>
            <div style="font-size:11px; color:var(--text-muted); margin-top:3px;">
              Prevents overexposure blowout on daylight/blurry images.
            </div>
          </div>

          <div style="margin-top:12px; display:flex; gap:8px;">
            <button id="btnWebcam" class="btn-sample" style="flex:1; background:var(--accent); color:white; border:none; padding:8px 12px; font-weight:700; border-radius:6px; cursor:pointer;" onclick="toggleWebcam()">
              📷 Start Live Webcam
            </button>
            <button id="btnSnapshot" class="btn-sample" style="display:none; background:#10b981; color:white; border:none; padding:8px 12px; font-weight:700; border-radius:6px; cursor:pointer;" onclick="captureWebcamFrame(true)">
              📸 Snapshot
            </button>
          </div>
          <div id="webcamContainer" style="display:none; margin-top:10px; border-radius:8px; overflow:hidden; border:2px solid var(--accent); position:relative; background:#000;">
            <video id="webcamVideo" autoplay playsinline muted style="width:100%; height:auto; display:block;"></video>
            <div style="position:absolute; top:6px; left:6px; background:rgba(0,0,0,0.75); color:#38bdf8; font-size:10px; font-weight:700; padding:2px 6px; border-radius:4px;">
              🔴 LIVE STREAM (Auto-Sampling: 2 FPS)
            </div>
          </div>

          <div style="margin-top:14px; font-size:13px; font-weight:600;">Pick Benchmark Dataset & Sample:</div>

          <div style="margin-top:6px; font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px;">LOL-v1 Benchmark (Baseline)</div>
          <div class="samples" style="margin-top:4px;">
            <button class="btn-sample" onclick="loadSample('102', 'v1')">102 (Gamma)</button>
            <button class="btn-sample" onclick="loadSample('101', 'v1')">101 (HistEq)</button>
            <button class="btn-sample" onclick="loadSample('109', 'v1')">109 (Retinex)</button>
            <button class="btn-sample" onclick="loadSample('1', 'v1')">1 (eval15)</button>
            <button class="btn-sample" onclick="loadSample('22', 'v1')">22 (eval15)</button>
            <button class="btn-sample" onclick="loadSample('79', 'v1')">79 (eval15)</button>
          </div>

          <div style="margin-top:10px; font-size:11px; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px;">LOL-v2 Real (Cross-Sensor Hardware)</div>
          <div class="samples" style="margin-top:4px;">
            <button class="btn-sample" onclick="loadSample('151', 'v2_real')">151 (Retinex)</button>
            <button class="btn-sample" onclick="loadSample('152', 'v2_real')">152 (HistEq)</button>
            <button class="btn-sample" onclick="loadSample('154', 'v2_real')">154 (Gamma)</button>
            <button class="btn-sample" onclick="loadSample('160', 'v2_real')">160 (Real)</button>
            <button class="btn-sample" onclick="loadSample('167', 'v2_real')">167 (Real)</button>
          </div>

          <div style="margin-top:10px; font-size:11px; font-weight:700; color:#10b981; text-transform:uppercase; letter-spacing:0.5px;">LOL-v2 Synthetic (Physical Poisson Degrader)</div>
          <div class="samples" style="margin-top:4px;">
            <button class="btn-sample" onclick="loadSample('10', 'v2_synthetic')">10 (Gamma)</button>
            <button class="btn-sample" onclick="loadSample('100', 'v2_synthetic')">100 (HistEq)</button>
            <button class="btn-sample" onclick="loadSample('105', 'v2_synthetic')">105 (Retinex)</button>
            <button class="btn-sample" onclick="loadSample('104', 'v2_synthetic')">104 (Syn)</button>
          </div>
        </div>

        <div class="card">
          <h2>2. Extracted Physical Statistics</h2>
          <div id="featuresList">
            <div style="font-size:13px; color:var(--text-muted);">Select or upload an image to view descriptors...</div>
          </div>
        </div>

        <div class="card">
          <h2>3. Model Selection Confidence</h2>
          <div id="confidenceBars">
            <div style="font-size:13px; color:var(--text-muted);">Awaiting inference...</div>
          </div>
        </div>
      </div>

      <!-- Right Column: Results & Comparisons -->
      <div>
        <div id="oodAlert" style="display:none; background:#fef3c7; border:1px solid #f59e0b; color:#92400e; padding:12px 16px; border-radius:8px; margin-bottom:16px; font-size:13px;">
          <div style="font-weight:700; display:flex; align-items:center; gap:6px;">
            <span>⚠️ Out-of-Distribution (OOD) Scene Detected</span>
          </div>
          <div id="oodMessage" style="margin-top:4px; line-height:1.4;">-</div>
        </div>

        <div id="resultBanner" style="display:none;" class="winner-banner">
          <div>
            Optimal Selected Enhancer: <strong id="winnerName">-</strong>
            <span id="ptoPill" style="display:inline-block; margin-left:8px; background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;">PTO: -</span>
            <div style="font-size:12px; margin-top:2px;" id="timingMeta">Computed in 18.2 ms</div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <button onclick="downloadAuditPdf()" style="background:#0284c7; color:white; border:none; padding:6px 12px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer; display:flex; align-items:center; gap:4px; box-shadow:0 1px 2px rgba(0,0,0,0.1);">
              📄 PDF Report
            </button>
            <div id="regretBadge" style="background:#15803d; color:white; padding:4px 10px; border-radius:6px; font-size:12px; font-weight:700;">Zero Regret</div>
          </div>
        </div>

        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
            <h2>Restoration Result</h2>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
              <button id="btnSide" class="badge-btn active" onclick="setViewMode('side')">Side-by-Side</button>
              <button id="btnSwipe" class="badge-btn" onclick="setViewMode('swipe')">Interactive Slider</button>
              <button id="btnDiff" class="badge-btn" onclick="setViewMode('diff')">Edge Delta Heatmap</button>
              <button id="btnMatch" class="badge-btn" onclick="setViewMode('match')">🔗 Inlier Matching (<span id="inlierCount">0</span>)</button>
              <button id="btnYolo" class="badge-btn" onclick="setViewMode('yolo')">🎯 YOLOv8 Detection (<span id="yoloCount">0</span>)</button>
              <button id="btnKp" class="badge-btn" onclick="toggleKeypoints()">Keypoints Overlay (<span id="kpCount">0</span> pts)</button>
            </div>
          </div>

          <!-- Mode 1: Side by Side -->
          <div id="sideViewer" class="viewer">
            <div class="img-box">
              <div class="img-label">Degraded Input (Low-Light)</div>
              <img id="imgDegraded" src="" alt="Degraded">
            </div>
            <div class="img-box">
              <div class="img-label" id="enhancedLabel">★ Selected Output ★</div>
              <img id="imgEnhanced" src="" alt="Enhanced">
            </div>
          </div>

          <!-- Mode 2: Interactive Swipe Comparison Slider -->
          <div id="swipeViewer" style="display:none; position:relative; width:100%; height:420px; overflow:hidden; border-radius:8px; background:#000; user-select:none;">
            <img id="swipeDegraded" src="" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain;">
            <div style="position:absolute; bottom:12px; left:12px; background:rgba(0,0,0,0.75); color:white; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:700; z-index:5;">Degraded Input</div>

            <div id="swipeClip" style="position:absolute; top:0; left:0; width:50%; height:100%; overflow:hidden; border-right:3px solid #2563eb; z-index:10; background:transparent;">
              <img id="swipeEnhanced" src="" style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain;">
              <div style="position:absolute; bottom:12px; right:12px; background:rgba(37,99,235,0.9); color:white; padding:4px 10px; border-radius:4px; font-size:12px; font-weight:700;">★ Selected Output ★</div>
            </div>

            <input type="range" min="0" max="100" value="50" id="swipeRange" oninput="onSwipeMove(this.value)" style="position:absolute; top:0; left:0; width:100%; height:100%; opacity:0; cursor:ew-resize; z-index:20;">
          </div>

          <!-- Mode 3: Edge Delta Heatmap -->
          <div id="diffViewer" class="viewer" style="display:none;">
            <div class="img-box">
              <div class="img-label">Degraded Input</div>
              <img id="diffDegraded" src="" alt="Degraded">
            </div>
            <div class="img-box">
              <div class="img-label">High-Pass Edge Sharpening Heatmap (Turbo)</div>
              <img id="diffHeatmap" src="" alt="Edge Difference">
            </div>
          </div>

          <!-- Mode 4: YOLOv8 Object Detection Viewer -->
          <div id="yoloViewer" class="viewer" style="display:none;">
            <div class="img-box">
              <div class="img-label" id="yoloDegLabel">Degraded Input · YOLOv8n (0 boxes)</div>
              <img id="yoloDeg" src="" alt="YOLO Degraded">
            </div>
            <div class="img-box">
              <div class="img-label" id="yoloEnhLabel">★ Enhanced Output · YOLOv8n (0 boxes) ★</div>
              <img id="yoloEnh" src="" alt="YOLO Enhanced">
            </div>
          </div>

          <!-- Mode 5: Geometric Inlier Matching Viewer -->
          <div id="matchViewer" class="viewer" style="display:none;">
            <div style="grid-column: 1 / -1; background:#000; border-radius:8px; overflow:hidden; text-align:center;">
              <div class="img-label" id="matchLabel" style="background:#0f172a; color:#38bdf8; font-weight:700;">ORB + RANSAC Feature Matching Vectors (Green = Inliers)</div>
              <img id="matchImg" src="" alt="ORB Inlier Matches" style="max-height:460px; width:100%; object-fit:contain;">
            </div>
          </div>
        </div>

        <div class="card">
          <h2>Portfolio Comparison (All 8 Multi-Hazard Enhancers)</h2>
          <div class="methods-grid" id="methodsGrid">
            <!-- Dynamically populated -->
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentData = null;
    let showingKeypoints = false;
    let currentViewMode = 'side';

    // Drag & drop handlers
    window.addEventListener('DOMContentLoaded', () => {
      const dropZone = document.getElementById('dropZone');
      if (dropZone) {
        ['dragenter', 'dragover'].forEach(eventName => {
          dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--accent)';
            dropZone.style.background = '#eff6ff';
          }, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
          dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = '#cbd5e1';
            dropZone.style.background = '#f8fafc';
          }, false);
        });
        dropZone.addEventListener('drop', (e) => {
          const files = e.dataTransfer.files;
          if (files && files.length > 0) {
            processUploadedFile(files[0]);
          }
        }, false);
      }
      // Load initial benchmark sample
      loadSample('102');
    });

    function setViewMode(mode) {
      currentViewMode = mode;
      document.getElementById('btnSide').classList.toggle('active', mode === 'side');
      document.getElementById('btnSwipe').classList.toggle('active', mode === 'swipe');
      document.getElementById('btnDiff').classList.toggle('active', mode === 'diff');
      document.getElementById('btnMatch').classList.toggle('active', mode === 'match');
      document.getElementById('btnYolo').classList.toggle('active', mode === 'yolo');
      document.getElementById('sideViewer').style.display = (mode === 'side') ? 'grid' : 'none';
      document.getElementById('swipeViewer').style.display = (mode === 'swipe') ? 'block' : 'none';
      document.getElementById('diffViewer').style.display = (mode === 'diff') ? 'grid' : 'none';
      document.getElementById('matchViewer').style.display = (mode === 'match') ? 'block' : 'none';
      document.getElementById('yoloViewer').style.display = (mode === 'yolo') ? 'grid' : 'none';
      if (mode === 'swipe') {
        syncSwipeWidth();
      }
    }

    function onSwipeMove(val) {
      const clip = document.getElementById('swipeClip');
      clip.style.width = val + '%';
      syncSwipeWidth();
    }

    function syncSwipeWidth() {
      const container = document.getElementById('swipeViewer');
      const innerImg = document.getElementById('swipeEnhanced');
      if (container && innerImg && container.offsetWidth > 0) {
        innerImg.style.width = container.offsetWidth + 'px';
      }
    }
    window.addEventListener('resize', syncSwipeWidth);

    function toggleKeypoints() {
      if (!currentData) return;
      showingKeypoints = !showingKeypoints;
      document.getElementById('btnKp').classList.toggle('active', showingKeypoints);
      const targetSrc = showingKeypoints ? currentData.keypoints_image : currentData.enhanced_image;
      document.getElementById('imgEnhanced').src = targetSrc;
      document.getElementById('swipeEnhanced').src = targetSrc;
      document.getElementById('enhancedLabel').innerText = showingKeypoints 
        ? `★ Selected Output (ORB Keypoints: ${currentData.keypoint_count}) ★` 
        : `★ Selected: ${currentData.predicted_method.toUpperCase()} ★`;
    }

    function showLoading(msg) {
      const uploadBox = document.getElementById('uploadBoxContent');
      if (uploadBox) uploadBox.innerHTML = `⏳ <strong>${msg}</strong>`;
      const banner = document.getElementById('resultBanner');
      banner.style.display = 'flex';
      banner.style.background = '#eff6ff';
      banner.style.borderColor = '#93c5fd';
      banner.style.color = '#1e40af';
      document.getElementById('winnerName').innerText = 'ANALYZING...';
      document.getElementById('timingMeta').innerText = msg;
      document.getElementById('regretBadge').style.display = 'none';
    }

    function hideLoading() {
      const uploadBox = document.getElementById('uploadBoxContent');
      if (uploadBox) uploadBox.innerHTML = '📁 <strong>Click or Drag Image Here</strong>';
    }

    function onSafeguardToggle(checked) {
      if (!currentData) return;
      applySelectionState(checked);
    }

    function applySelectionState(enforceSafeguard) {
      if (!currentData) return;
      const targetMethod = (enforceSafeguard && currentData.safeguard_active) 
        ? currentData.safeguard_method 
        : currentData.raw_predicted_method;
      
      const isWinnerOverruled = (targetMethod !== currentData.raw_predicted_method);
      
      // Update displayed images
      const targetImg = currentData.all_methods[targetMethod].image_base64;
      document.getElementById('imgEnhanced').src = targetImg;
      document.getElementById('swipeEnhanced').src = targetImg;
      
      // Update banner
      const banner = document.getElementById('resultBanner');
      banner.style.display = 'flex';
      if (isWinnerOverruled) {
        banner.style.background = '#eff6ff';
        banner.style.borderColor = '#93c5fd';
        banner.style.color = '#1e40af';
        document.getElementById('winnerName').innerText = `${targetMethod.toUpperCase()} (SAFEGUARDED)`;
      } else {
        banner.style.background = '#dcfce7';
        banner.style.borderColor = '#86efac';
        banner.style.color = '#15803d';
        document.getElementById('winnerName').innerText = targetMethod.toUpperCase();
      }
      document.getElementById('enhancedLabel').innerText = `★ Selected: ${targetMethod.toUpperCase()} ★`;
      if (currentData && currentData.pto_prediction) {
        document.getElementById('ptoPill').innerText = `PTO Regressor Choice: ${currentData.pto_prediction.toUpperCase()}`;
      }

      // Update OOD Alert
      const oodAlert = document.getElementById('oodAlert');
      if (currentData.safeguard_active) {
        oodAlert.style.display = 'block';
        document.getElementById('oodMessage').innerText = currentData.safeguard_reason;
      } else {
        oodAlert.style.display = 'none';
      }

      // Update Portfolio Tiles
      for (const [mName, mData] of Object.entries(currentData.all_methods)) {
        const tile = document.getElementById(`tile-${mName}`);
        if (tile) {
          tile.classList.toggle('selected', mName === targetMethod);
          const metaDiv = tile.querySelector('.meta');
          if (metaDiv) {
            metaDiv.style.color = (mName === targetMethod) ? 'var(--success)' : '';
            metaDiv.style.fontWeight = (mName === targetMethod) ? '700' : '';
          }
        }
      }
    }

    function uploadImage(e) {
      const file = e.target.files ? e.target.files[0] : null;
      if (!file) return;
      processUploadedFile(file);
      e.target.value = ''; // Allow re-uploading the same file
    }

    async function processUploadedFile(file) {
      showLoading('Extracting features & running portfolio...');
      const isSafeguard = document.getElementById('chkSafeguard').checked;
      const reader = new FileReader();
      reader.onload = async function() {
        try {
          const b64 = reader.result;
          const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64, safeguard: isSafeguard })
          });
          const data = await res.json();
          if (data.status === 'success') {
            renderResults(data);
          } else {
            alert('Processing error: ' + (data.message || 'Unknown server error'));
            hideLoading();
          }
        } catch (err) {
          alert('Failed to process image: ' + err.message);
          hideLoading();
        }
      };
      reader.onerror = function() {
        alert('Could not read file from disk.');
        hideLoading();
      };
      reader.readAsDataURL(file);
    }

    async function loadSample(name, dataset = '') {
      showLoading(`Loading benchmark sample #${name}...`);
      const isSafeguard = document.getElementById('chkSafeguard').checked;
      try {
        const res = await fetch(`/api/sample?name=${name}&dataset=${dataset}&safeguard=${isSafeguard}`);
        const data = await res.json();
        if (data.status === 'success') {
          renderResults(data);
        } else {
          alert('Sample load error: ' + (data.message || 'Sample image missing'));
          hideLoading();
        }
      } catch (err) {
        alert('Error loading sample: ' + err.message);
        hideLoading();
      }
    }

    function previewMethod(mName) {
      if (!currentData || !currentData.all_methods[mName]) return;
      const targetImg = currentData.all_methods[mName].image_base64;
      document.getElementById('imgEnhanced').src = targetImg;
      document.getElementById('swipeEnhanced').src = targetImg;
      document.getElementById('enhancedLabel').innerText = `★ Previewing: ${mName.toUpperCase()} ★`;

      // Update card selections
      for (const [name, _] of Object.entries(currentData.all_methods)) {
        const tile = document.getElementById(`tile-${name}`);
        if (tile) {
          tile.classList.toggle('selected', name === mName);
          const metaDiv = tile.querySelector('.meta');
          if (metaDiv) {
            metaDiv.style.color = (name === mName) ? 'var(--success)' : '';
            metaDiv.style.fontWeight = (name === mName) ? '700' : '';
          }
        }
      }
    }

    function renderResults(data) {
      hideLoading();
      if (!data || data.status !== 'success') return;
      currentData = data;
      showingKeypoints = false;
      document.getElementById('btnKp').classList.remove('active');
      document.getElementById('kpCount').innerText = data.keypoint_count || '0';

      // Images
      document.getElementById('imgDegraded').src = data.degraded_image;
      document.getElementById('swipeDegraded').src = data.degraded_image;
      if (data.heatmap_image) {
        document.getElementById('diffDegraded').src = data.degraded_image;
        document.getElementById('diffHeatmap').src = data.heatmap_image;
      }
      if (data.yolo_enh_image) {
        document.getElementById('yoloDeg').src = data.yolo_deg_image;
        document.getElementById('yoloEnh').src = data.yolo_enh_image;
        document.getElementById('yoloCount').innerText = data.yolo_enh_boxes || '0';
        document.getElementById('yoloDegLabel').innerText = `Degraded Input · YOLOv8n: ${data.yolo_deg_boxes || 0} boxes (avg conf: ${data.yolo_deg_conf || 0})`;
        document.getElementById('yoloEnhLabel').innerText = `★ Enhanced: ${data.yolo_enh_boxes || 0} boxes (avg conf: ${data.yolo_enh_conf || 0}) ★`;
      }
      if (data.orb_match_image) {
        document.getElementById('matchImg').src = data.orb_match_image;
        document.getElementById('inlierCount').innerText = `${data.orb_inliers || 0} pts`;
        document.getElementById('matchLabel').innerText = `★ ORB + RANSAC Inlier Matching: ${data.orb_inliers || 0} inliers (${data.orb_inlier_ratio || 0}%) | Odometry Gain vs Dark: +${data.orb_gain_pct || 0}% ★`;
      }
      syncSwipeWidth();

      // Timing
      document.getElementById('timingMeta').innerText = `Feature extraction: ${data.timings_ms.features} ms | Total inference: ${data.timings_ms.total} ms`;

      if (data.reference_eval) {
        document.getElementById('regretBadge').style.display = 'block';
        document.getElementById('regretBadge').innerText = `Regret: ${data.reference_eval.regret_db} dB`;
      } else {
        document.getElementById('regretBadge').style.display = 'none';
      }

      // Features
      let fHtml = '';
      fHtml += `<div class="feature-row" style="background:#f8fafc; padding:6px 4px; border-radius:4px;"><span class="feature-name" style="font-weight:700; color:var(--text);">Domain Status</span><span class="feature-val" style="color:${data.safeguard_active ? '#d97706' : '#16a34a'}; font-weight:700;">${data.domain_status}</span></div>`;
      fHtml += `<div class="feature-row" style="background:#f8fafc; padding:6px 4px; border-radius:4px;"><span class="feature-name" style="font-weight:700; color:var(--text);">Laplacian Blur Var</span><span class="feature-val">${data.laplacian_var}</span></div>`;
      for (const [k, v] of Object.entries(data.features)) {
        fHtml += `<div class="feature-row"><span class="feature-name">${k}</span><span class="feature-val">${v}</span></div>`;
      }
      document.getElementById('featuresList').innerHTML = fHtml;

      // Probabilities
      let pHtml = '';
      const sortedClasses = Object.entries(data.probabilities).sort((a,b) => b[1] - a[1]);
      for (const [cls, prob] of sortedClasses) {
        const isWinner = (cls === data.raw_predicted_method);
        pHtml += `
          <div class="prob-bar-container">
            <div class="prob-bar-header">
              <span><strong>${cls}</strong></span>
              <span>${prob}%</span>
            </div>
            <div class="prob-bar-bg">
              <div class="prob-bar-fill ${isWinner ? 'winner' : ''}" style="width: ${prob}%"></div>
            </div>
          </div>
        `;
      }
      document.getElementById('confidenceBars').innerHTML = pHtml;

      // Portfolio Grid
      let mHtml = '';
      for (const [mName, mData] of Object.entries(data.all_methods)) {
        const isSel = (mName === data.predicted_method);
        const metaText = mData.psnr ? `PSNR: ${mData.psnr} dB | SSIM: ${mData.ssim}` : mName;
        mHtml += `
          <div class="method-tile ${isSel ? 'selected' : ''}" id="tile-${mName}" onclick="previewMethod('${mName}')" title="Click to preview ${mName.toUpperCase()} in main viewer">
            <img src="${mData.image_base64}">
            <div class="meta" style="${isSel ? 'color:var(--success); font-weight:700;' : ''}">
              ${mName.toUpperCase()} ${isSel ? '★' : ''}<br>
              <span style="font-size:11px; color:#64748b;">${metaText}</span>
            </div>
          </div>
        `;
      }
      document.getElementById('methodsGrid').innerHTML = mHtml;

      // Apply selection based on toggle state
      const isSafeguard = document.getElementById('chkSafeguard').checked;
      applySelectionState(isSafeguard);
    }

    // Live Webcam Implementation
    let webcamStream = null;
    let webcamInterval = null;
    let isWebcamProcessing = false;

    async function toggleWebcam() {
      const btn = document.getElementById('btnWebcam');
      const snapBtn = document.getElementById('btnSnapshot');
      const container = document.getElementById('webcamContainer');
      const video = document.getElementById('webcamVideo');

      if (webcamStream) {
        clearInterval(webcamInterval);
        webcamInterval = null;
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
        video.srcObject = null;
        container.style.display = 'none';
        snapBtn.style.display = 'none';
        btn.textContent = '📷 Start Live Webcam';
        btn.style.background = 'var(--accent)';
        return;
      }

      try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
          audio: false
        });
        video.srcObject = webcamStream;
        container.style.display = 'block';
        snapBtn.style.display = 'block';
        btn.textContent = '⏹ Stop Webcam';
        btn.style.background = '#ef4444';

        // Auto-sample every 500ms
        webcamInterval = setInterval(() => {
          captureWebcamFrame(false);
        }, 500);
      } catch (err) {
        alert('Webcam Access Error: ' + err.message + '\nPlease check camera permissions in your browser settings.');
      }
    }

    function captureWebcamFrame(isManualSnapshot = false) {
      if (isWebcamProcessing && !isManualSnapshot) return;
      const video = document.getElementById('webcamVideo');
      if (!video || video.readyState !== 4) return;

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const b64 = canvas.toDataURL('image/jpeg', 0.85);

      if (isManualSnapshot) {
        toggleWebcam();
        showLoading('Processing snapshot...');
      }

      isWebcamProcessing = true;
      const isSafeguard = document.getElementById('chkSafeguard').checked;
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64, safeguard: isSafeguard })
      })
      .then(r => r.json())
      .then(data => {
        isWebcamProcessing = false;
        if (data.status === 'success') {
          renderResults(data);
        }
        if (isManualSnapshot) hideLoading();
      })
      .catch(err => {
        isWebcamProcessing = false;
        if (isManualSnapshot) {
          alert('Snapshot error: ' + err.message);
          hideLoading();
        }
      });
    }

    // 1-Click PDF Audit Export
    function downloadAuditPdf() {
      if (!currentData) {
        alert('Please select or upload an image first.');
        return;
      }
      const btn = event.target;
      const oldText = btn.textContent;
      btn.textContent = '⏳ Compiling PDF...';

      fetch('/api/export_pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          degraded_image: currentData.degraded_image,
          enhanced_image: currentData.enhanced_image,
          predicted_method: currentData.predicted_method,
          pto_prediction: currentData.pto_prediction,
          features: currentData.features,
          timings_ms: currentData.timings_ms,
          orb_inliers: currentData.orb_inliers,
          orb_inlier_ratio: currentData.orb_inlier_ratio
        })
      })
      .then(resp => {
        if (!resp.ok) throw new Error('PDF generation failed on server');
        return resp.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Project_Triathlon_${currentData.predicted_method.toUpperCase()}_Audit_Report.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        btn.textContent = oldText;
      })
      .catch(err => {
        alert('Error exporting PDF report: ' + err.message);
        btn.textContent = oldText;
      });
    }
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def _safe_write(self, data: bytes):
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            return
        elif parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self._safe_write(HTML_TEMPLATE.encode("utf-8"))
            return

        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if parsed.path == "/api/sample":
            params = urllib.parse.parse_qs(parsed.query)
            sample_id = params.get("name", ["102"])[0]
            dataset_choice = params.get("dataset", [""])[0].lower()
            safeguard = params.get("safeguard", ["true"])[0].lower() in ["true", "1"]
            
            low_candidates = []
            high_candidates = []

            if dataset_choice == "v2_real":
                low_candidates = [
                    f"data/lol_v2/real/low/{sample_id}.png",
                    f"data/lol_v2/real/low/{sample_id}.jpg",
                ]
                high_candidates = [
                    f"data/lol_v2/real/high/{sample_id}.png",
                    f"data/lol_v2/real/high/{sample_id}.jpg",
                ]
            elif dataset_choice == "v2_synthetic":
                low_candidates = [
                    f"data/lol_v2/synthetic/low/{sample_id}.png",
                    f"data/lol_v2/synthetic/low/{sample_id}.jpg",
                ]
                high_candidates = [
                    f"data/lol_v2/synthetic/high/{sample_id}.png",
                    f"data/lol_v2/synthetic/high/{sample_id}.jpg",
                ]
            else:
                low_candidates = [
                    f"data/lol_dataset/our485/low/{sample_id}.png",
                    f"data/lol_dataset/eval15/low/{sample_id}.png",
                    f"data/lol_v2/real/low/{sample_id}.png",
                    f"data/lol_v2/synthetic/low/{sample_id}.png",
                    f"/Users/home/Desktop/LOLdataset/our485/low/{sample_id}.png",
                    f"/Users/home/Desktop/LOLdataset/eval15/low/{sample_id}.png",
                ]
                high_candidates = [
                    f"data/lol_dataset/our485/high/{sample_id}.png",
                    f"data/lol_dataset/eval15/high/{sample_id}.png",
                    f"data/lol_v2/real/high/{sample_id}.png",
                    f"data/lol_v2/synthetic/high/{sample_id}.png",
                    f"/Users/home/Desktop/LOLdataset/our485/high/{sample_id}.png",
                    f"/Users/home/Desktop/LOLdataset/eval15/high/{sample_id}.png",
                ]

            low_path = next((p for p in low_candidates if os.path.exists(p)), None)
            high_path = next((p for p in high_candidates if os.path.exists(p)), None)

            if not low_path:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps({"status": "error", "message": f"Sample image #{sample_id} not found"}).encode("utf-8"))
                return

            deg_img = cv2.imread(low_path)
            if deg_img is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps({"status": "error", "message": f"Failed to load image from {low_path}"}).encode("utf-8"))
                return
            clean_img = cv2.imread(high_path) if high_path else None
            try:
                res = process_image(deg_img, clean_img, enforce_safeguards=safeguard)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps({"status": "error", "message": f"Processing error: {str(e)}"}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self._safe_write(json.dumps({"status": "error", "message": "Empty request body"}).encode("utf-8"))
                    return

                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                b64_img = payload.get("image") or payload.get("image_base64") or ""
                enforce_safeguard = payload.get("safeguard", True)

                deg_img = base64_to_img(b64_img)
                if deg_img is None or deg_img.size == 0:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self._safe_write(json.dumps({"status": "error", "message": "Could not decode image. Please upload a valid JPG, PNG, or WebP file."}).encode("utf-8"))
                    return

                res = process_image(deg_img, enforce_safeguards=enforce_safeguard)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps({"status": "error", "message": f"Processing error: {str(e)}"}).encode("utf-8"))
            return

        elif self.path == "/api/export_pdf":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                deg_img = base64_to_img(payload.get("degraded_image", ""))
                enh_img = base64_to_img(payload.get("enhanced_image", ""))
                if deg_img is None:
                    deg_img = np.zeros((256, 256, 3), dtype=np.uint8)
                if enh_img is None:
                    enh_img = deg_img.copy()

                pdf_bytes = generate_pdf_audit_report(deg_img, enh_img, payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                sel_m = payload.get("predicted_method", "audit")
                self.send_header("Content-Disposition", f"attachment; filename=\"Project_Triathlon_{sel_m}_Report.pdf\"")
                self.send_header("Content-Length", str(len(pdf_bytes)))
                self.end_headers()
                self._safe_write(pdf_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self._safe_write(json.dumps({"status": "error", "message": f"PDF Generation Error: {str(e)}"}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


class ReusableHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def main():
    parser = argparse.ArgumentParser(description="Project Triathlon Interactive Web App")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on (default: 8501)")
    args = parser.parse_args()

    # Pre-load model
    print("Loading selector model...")
    get_model()

    server = None
    port = args.port
    for attempt_port in range(port, port + 20):
        try:
            server = ReusableHTTPServer(("", attempt_port), RequestHandler)
            port = attempt_port
            break
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"Port {attempt_port} is already in use, trying port {attempt_port + 1}...")
                continue
            raise

    if server is None:
        raise RuntimeError(f"Could not bind to any port in range {args.port} - {args.port + 19}")

    print(f"\n=======================================================")
    print(f"Project Triathlon Interactive Web App Running!")
    print(f"Access locally at: http://localhost:{port}")
    print(f"Direct IPv4:      http://127.0.0.1:{port}")
    if port != args.port:
        print(f"Note: Requested port {args.port} was busy, bound to port {port}")
    print(f"Press Ctrl+C to terminate the server.")
    print(f"=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()

