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
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
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

PORTFOLIO_METHODS = {
    **METHODS,
    "high_boost_sharpen": high_boost_sharpen,
    "dcp_dehaze": dcp_dehaze,
}

MODEL_PATH = "models/rf_selector.joblib"
MODEL = None
YOLO_MODEL = None


def get_yolo_detections(img_bgr: np.ndarray, conf_thresh: float = 0.20):
    """Executes YOLOv8n object detection and renders bounding boxes."""
    global YOLO_MODEL
    try:
        if YOLO_MODEL is None:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO("yolov8n.pt")
        res = YOLO_MODEL.predict(img_bgr, conf=conf_thresh, verbose=False)[0]
        annotated = res.plot()
        n_boxes = len(res.boxes)
        mean_conf = float(res.boxes.conf.mean().item()) if n_boxes > 0 else 0.0
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


def base64_to_img(b64_str: str) -> np.ndarray:
    """Decodes base64 string to BGR image."""
    try:
        if not b64_str:
            return None
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        b64_clean = b64_str.strip().replace("\n", "").replace("\r", "")
        data = base64.b64decode(b64_clean)
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Base64 decoding error: {e}")
        return None


def process_image(img_bgr: np.ndarray, ref_bgr: np.ndarray = None, enforce_safeguards: bool = True):
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

    is_well_exposed = bool(feats["brightness"] > 55.0 or (feats["brightness"] > 45.0 and feats["dark_channel"] > 25.0))
    is_soft_or_blurry = bool(lap_var < 300.0 or (feats["edge_p90"] < 85.0 and feats["contrast"] > 40.0))

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
    orb = cv2.ORB_create(nfeatures=400)
    kp = orb.detect(enhanced_bgr, None)
    kp_img = cv2.drawKeypoints(enhanced_bgr, kp, None, color=(0, 255, 0), flags=0)
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
        cand_dict = {"image_base64": f"data:image/png;base64,{cand_b64}"}
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

    return {
        "status": "success",
        "predicted_method": pred_method,
        "raw_predicted_method": raw_pred_method,
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

          <div style="margin-top:14px; font-size:13px; font-weight:600;">Or Pick LOL v1 Benchmark Sample:</div>
          <div class="samples">
            <button class="btn-sample" onclick="loadSample('102')">102 (Gamma)</button>
            <button class="btn-sample" onclick="loadSample('101')">101 (HistEq)</button>
            <button class="btn-sample" onclick="loadSample('109')">109 (Retinex)</button>
            <button class="btn-sample" onclick="loadSample('1')">1 (eval15)</button>
            <button class="btn-sample" onclick="loadSample('22')">22 (eval15)</button>
            <button class="btn-sample" onclick="loadSample('79')">79 (eval15)</button>
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
            <div style="font-size:12px; margin-top:2px;" id="timingMeta">Computed in 18.2 ms</div>
          </div>
          <div id="regretBadge" style="background:#15803d; color:white; padding:4px 10px; border-radius:6px; font-size:12px; font-weight:700;">Zero Regret</div>
        </div>

        <div class="card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
            <h2>Restoration Result</h2>
            <div style="display:flex; gap:6px; flex-wrap:wrap;">
              <button id="btnSide" class="badge-btn active" onclick="setViewMode('side')">Side-by-Side</button>
              <button id="btnSwipe" class="badge-btn" onclick="setViewMode('swipe')">Interactive Slider</button>
              <button id="btnDiff" class="badge-btn" onclick="setViewMode('diff')">Edge Delta Heatmap</button>
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
      document.getElementById('btnYolo').classList.toggle('active', mode === 'yolo');
      document.getElementById('sideViewer').style.display = (mode === 'side') ? 'grid' : 'none';
      document.getElementById('swipeViewer').style.display = (mode === 'swipe') ? 'block' : 'none';
      document.getElementById('diffViewer').style.display = (mode === 'diff') ? 'grid' : 'none';
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

    async function loadSample(name) {
      showLoading(`Loading benchmark sample #${name}...`);
      const isSafeguard = document.getElementById('chkSafeguard').checked;
      try {
        const res = await fetch(`/api/sample?name=${name}&safeguard=${isSafeguard}`);
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
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if parsed.path == "/api/sample":
            params = urllib.parse.parse_qs(parsed.query)
            sample_id = params.get("name", ["102"])[0]
            safeguard = params.get("safeguard", ["true"])[0].lower() in ["true", "1"]
            
            # Check eval15 or our485
            low_candidates = [
                f"/Users/home/Desktop/LOLdataset/our485/low/{sample_id}.png",
                f"/Users/home/Desktop/LOLdataset/eval15/low/{sample_id}.png",
                f"data/lol_dataset/our485/low/{sample_id}.png",
                f"data/lol_dataset/eval15/low/{sample_id}.png",
            ]
            high_candidates = [
                f"/Users/home/Desktop/LOLdataset/our485/high/{sample_id}.png",
                f"/Users/home/Desktop/LOLdataset/eval15/high/{sample_id}.png",
                f"data/lol_dataset/our485/high/{sample_id}.png",
                f"data/lol_dataset/eval15/high/{sample_id}.png",
            ]

            low_path = next((p for p in low_candidates if os.path.exists(p)), None)
            high_path = next((p for p in high_candidates if os.path.exists(p)), None)

            if not low_path:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Sample image #{sample_id} not found"}).encode("utf-8"))
                return

            deg_img = cv2.imread(low_path)
            if deg_img is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Failed to load image from {low_path}"}).encode("utf-8"))
                return
            clean_img = cv2.imread(high_path) if high_path else None
            try:
                res = process_image(deg_img, clean_img, enforce_safeguards=safeguard)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Processing error: {str(e)}"}).encode("utf-8"))
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
                    self.wfile.write(json.dumps({"status": "error", "message": "Empty request body"}).encode("utf-8"))
                    return

                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                b64_img = payload.get("image", "")
                enforce_safeguard = payload.get("safeguard", True)

                deg_img = base64_to_img(b64_img)
                if deg_img is None or deg_img.size == 0:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Could not decode image. Please upload a valid JPG, PNG, or WebP file."}).encode("utf-8"))
                    return

                res = process_image(deg_img, enforce_safeguards=enforce_safeguard)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Processing error: {str(e)}"}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        # Quiet standard HTTP access logs
        return


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


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
            server = ReusableHTTPServer(("0.0.0.0", attempt_port), RequestHandler)
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

