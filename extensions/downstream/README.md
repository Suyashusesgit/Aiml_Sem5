# Downstream Computer Vision Utilities (ORB / RANSAC Geometric Matching)

This extension evaluates whether fidelity-driven algorithm selection directly translates into superior downstream machine vision performance (geometric feature matching and keypoint repeatability) under low-light conditions.

**Target Hardware**: CPU only (`device="cpu"`, no GPU required).  
**Dependencies**: OpenCV and NumPy standard dependencies.

---

## Components

1. **`downstream_matching.py`**:
   - Computes ORB keypoints (1,000 per image) and descriptors.
   - Performs RANSAC geometric verification to count true inlier matches against clean reference pairs.
2. **`downstream_detection.py`**:
   - Automated benchmark across evaluation splits logging keypoint repeatability rates.
3. **`downstream_semantic_eval.py`**:
   - Evaluates keypoint and edge consistency across varying lighting conditions.

---

## How to Run (CPU)

```bash
# 1. Run ORB/RANSAC geometric matching benchmark (CPU)
python3 src/downstream_matching.py

# 2. Run downstream keypoint recovery evaluation (CPU)
python3 src/downstream_detection.py --split eval15
```

---
*Maintained under Project Triathlon Extensions Architecture (Pure Classical Machine Vision).*
