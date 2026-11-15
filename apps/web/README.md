# Project Triathlon: Interactive Web Application

A lightweight, zero-external-dependency interactive web interface for low-light image enhancement, per-image algorithm selection, downstream machine vision preview, and side-by-side quality analysis.

**Target Hardware**: Real-time CPU execution (`device="cpu"`).  
**Dependencies**: Uses standard Python library (`http.server`), OpenCV (`cv2`), NumPy, and scikit-learn. Zero framework lock-in.

---

## Features

1. **Drag-and-Drop Image Enhancement**:
   - Upload any degraded image or select from pre-loaded benchmark scenes.
   - Computes 8 handcrafted physical descriptors in real time (< 10 ms).
   - Runs winning Logistic Regression selector to predict optimal enhancer.
2. **Side-by-Side Quality Comparison**:
   - Visualizes degraded input, ground truth reference (if provided), and all candidate enhanced images.
   - Computes full-reference metrics (PSNR, SSIM, Decision Regret).
3. **Downstream Feature Matching Preview**:
   - Live ORB keypoint extraction and RANSAC geometric matching visualization.
4. **Zero-Configuration Server**:
   - Built directly on Python's built-in `http.server` with an asynchronous REST API backend.

---

## How to Run (CPU)

```bash
# Launch web server on port 8501 (or custom port)
python3 apps/web/app.py --port 8501

# Alternatively, from root:
python3 app.py --port 8501
```

Then open your browser to `http://localhost:8501`.

---
*Maintained under Project Triathlon Applications Architecture.*
