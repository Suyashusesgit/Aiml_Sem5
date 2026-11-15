# Downstream Computer Vision Utilities (YOLOv8 & ORB/RANSAC)

This extension evaluates whether fidelity-driven algorithm selection directly translates into superior downstream machine vision performance (object detection and geometric feature matching) under low-light conditions.

**Target Hardware**: CPU only (`device="cpu"`, no GPU required).  
**Dependencies**: Listed in `extensions/requirements-extensions.txt` (`ultralytics`, `torch`).

---

## Components

1. **`yolov8n.pt`**: Pretrained YOLOv8 Nano weights for lightweight CPU object detection.
2. **`downstream_yolo_detection.py`**:
   - Runs object detection on raw low-light images vs. selector-enhanced images.
   - Computes mAP, class recall gains, and detection confidence shifts.
3. **`downstream_matching.py`**:
   - Computes ORB keypoints (1,000 per image) and descriptors.
   - Performs RANSAC geometric verification to count true inlier matches against clean reference pairs.
4. **`downstream_detection.py`**:
   - Automated benchmark across evaluation splits logging keypoint repeatability rates.
5. **`downstream_semantic_eval.py`**:
   - Evaluates object detection consistency across varying lighting conditions.

---

## How to Run (CPU)

```bash
# 1. Install extension dependencies (isolated from core)
pip install -r extensions/requirements-extensions.txt

# 2. Run ORB/RANSAC geometric matching benchmark (CPU)
python3 extensions/downstream/downstream_matching.py

# 3. Run YOLOv8 object detection benchmark (CPU)
python3 extensions/downstream/downstream_yolo_detection.py \
    --weights extensions/downstream/yolov8n.pt \
    --data data/lol_dataset/eval15/low \
    --device cpu

# 4. Run downstream keypoint recovery evaluation (CPU)
python3 extensions/downstream/downstream_detection.py --split eval15
```

---
*Maintained under Project Triathlon Extensions Architecture.*
