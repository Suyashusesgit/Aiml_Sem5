"""
Downstream Geometric Feature Matching & Visual Odometry Inlier Tracker (Phase 0 Deferred Milestone)
Authors: Project Triathlon Engineering Team
Supervisor: Prof. Kishor P. Upla (Senior Member, IEEE)

Implements Page 9 Deferred Future Scope:
"Downstream decision validation using ORB/AKAZE feature matching."
"""

import time
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np


def compute_orb_inlier_matches(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    max_features: int = 500,
    ransac_thresh: float = 4.0
) -> Dict[str, Any]:
    """
    Computes robust geometric keypoint correspondences and RANSAC inliers
    between two image frames using ORB descriptors and Epipolar / Homography constraints.

    Returns:
        Dict containing total_matches, inliers, inlier_ratio, and rendered visualization canvas.
    """
    if img1_bgr is None or img2_bgr is None:
        raise ValueError("Input images cannot be None")

    t0 = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)

    kp1, des1 = orb.detectAndCompute(img1_bgr, None)
    kp2, des2 = orb.detectAndCompute(img2_bgr, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # Fallback empty visualization
        h = max(img1_bgr.shape[0], img2_bgr.shape[0])
        w = img1_bgr.shape[1] + img2_bgr.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:img1_bgr.shape[0], :img1_bgr.shape[1]] = img1_bgr
        canvas[:img2_bgr.shape[0], img1_bgr.shape[1]:] = img2_bgr
        return {
            "total_matches": 0,
            "inlier_count": 0,
            "inlier_ratio_pct": 0.0,
            "kp1_count": len(kp1) if kp1 else 0,
            "kp2_count": len(kp2) if kp2 else 0,
            "canvas_bgr": canvas,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Match descriptors using Hamming distance with cross-check
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        canvas = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, matches, None, flags=2)
        return {
            "total_matches": len(matches),
            "inlier_count": 0,
            "inlier_ratio_pct": 0.0,
            "kp1_count": len(kp1),
            "kp2_count": len(kp2),
            "canvas_bgr": canvas,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # RANSAC homography estimation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = round((inliers / max(1, len(matches))) * 100.0, 1)

    # Render matching visualization: green = inliers, red = outliers
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask.ravel().tolist() if mask is not None else None,
        flags=2
    )
    canvas = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, matches, None, **draw_params)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "total_matches": len(matches),
        "inlier_count": inliers,
        "inlier_ratio_pct": inlier_ratio,
        "kp1_count": len(kp1),
        "kp2_count": len(kp2),
        "canvas_bgr": canvas,
        "latency_ms": round(latency_ms, 2),
    }


def evaluate_enhancement_matching_gain(
    deg_bgr: np.ndarray,
    enh_bgr: np.ndarray,
    ref_bgr: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates the downstream visual odometry dividend:
    Compares feature repeatability of Degraded vs Enhanced.
    If ref_bgr is provided, matches both against clean ground truth.
    If not, applies a calibrated affine/perspective transform to quantify self-tracking stability.
    """
    h, w = deg_bgr.shape[:2]
    if ref_bgr is None:
        # Create a calibrated synthetic camera motion view (affine rotation + shift)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 0.98)
        target_deg = cv2.warpAffine(deg_bgr, M, (w, h))
        target_enh = cv2.warpAffine(enh_bgr, M, (w, h))
        baseline_res = compute_orb_inlier_matches(deg_bgr, target_deg)
        enhanced_res = compute_orb_inlier_matches(enh_bgr, target_enh)
    else:
        baseline_res = compute_orb_inlier_matches(deg_bgr, ref_bgr)
        enhanced_res = compute_orb_inlier_matches(enh_bgr, ref_bgr)

    gain_inliers = enhanced_res["inlier_count"] - baseline_res["inlier_count"]
    gain_pct = round(
        ((enhanced_res["inlier_count"] - baseline_res["inlier_count"]) / max(1, baseline_res["inlier_count"])) * 100.0,
        1
    )

    return {
        "baseline_inliers": baseline_res["inlier_count"],
        "baseline_ratio": baseline_res["inlier_ratio_pct"],
        "enhanced_inliers": enhanced_res["inlier_count"],
        "enhanced_ratio": enhanced_res["inlier_ratio_pct"],
        "inlier_gain": gain_inliers,
        "inlier_gain_pct": gain_pct,
        "canvas_bgr": enhanced_res["canvas_bgr"],
        "latency_ms": round(baseline_res["latency_ms"] + enhanced_res["latency_ms"], 2),
    }
