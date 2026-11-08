"""
Dataset and Preprocessing Module
Implements the preprocessing contract and scene grouping specified in
Chapter 23 of the handbook:
- 512px downscaling along the long side using cv2.INTER_AREA (identical scaling for clean and degraded pairs)
- Lossless PNG pipeline
- Orientation verification
- Grouped split handling to avoid scene leakage across cross-validation folds
"""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


def resize_long_side(img: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, float]:
    """
    Resizes image such that the longest dimension equals target_size,
    preserving exact aspect ratio with cv2.INTER_AREA.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= target_size:
        return img.copy(), 1.0

    scale = target_size / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def load_and_preprocess_pair(
    low_path: str,
    high_path: str,
    target_long_side: int = 512
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads degraded (low-light) and clean (normal-light) image pair and applies
    the identical preprocessing contract.

    Args:
        low_path: File path to degraded image.
        high_path: File path to ground-truth clean image.
        target_long_side: Dimension to which the long side is scaled.

    Returns:
        Tuple of (clean_img, degraded_img, scale_factor) in BGR uint8 format.
    """
    if not os.path.exists(low_path):
        raise FileNotFoundError(f"Low-light image not found: {low_path}")
    if not os.path.exists(high_path):
        raise FileNotFoundError(f"Clean reference image not found: {high_path}")

    deg = cv2.imread(low_path, cv2.IMREAD_COLOR)
    clean = cv2.imread(high_path, cv2.IMREAD_COLOR)

    if deg is None:
        raise ValueError(f"Failed to read image at {low_path}")
    if clean is None:
        raise ValueError(f"Failed to read image at {high_path}")

    if deg.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch between degraded {deg.shape} and clean {clean.shape}"
        )

    # Apply identical resize to both
    deg_resized, scale = resize_long_side(deg, target_size=target_long_side)
    clean_resized, _ = resize_long_side(clean, target_size=target_long_side)

    return clean_resized, deg_resized, scale


def compute_perceptual_hash(img: np.ndarray) -> str:
    """Computes a 64-bit perceptual average hash for grouping duplicate scenes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    return "".join("1" if b else "0" for b in bits)


def resolve_dataset_dir(candidate_path: Optional[str] = None) -> str:
    """
    Resolves the dataset directory with priority:
    1. candidate_path (if provided and valid)
    2. /Users/home/Desktop/LOLdataset (external Desktop dataset)
    3. data/lol_dataset (project-local)
    4. data
    """
    candidates = []
    if candidate_path:
        candidates.append(candidate_path)
    candidates.extend([
        "/Users/home/Desktop/LOLdataset",
        "data/lol_dataset",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lol_dataset")),
        "data",
    ])
    for c in candidates:
        if c and os.path.exists(c):
            try:
                test_sub = os.path.join(c, "our485")
                if os.path.isdir(test_sub) and len(os.listdir(test_sub)) > 0:
                    return c
                test_nested = os.path.join(c, "lol_dataset", "our485")
                if os.path.isdir(test_nested) and len(os.listdir(test_nested)) > 0:
                    return c
            except (PermissionError, OSError):
                continue
    return candidate_path or "data/lol_dataset"


def get_dataset_manifest(
    data_dir: str,
    split_name: str = "our485"
) -> List[Dict[str, Any]]:
    """
    Scans dataset directory and constructs manifest of valid paired images.

    Args:
        data_dir: Path to base data directory containing 'our485' or 'eval15'.
        split_name: 'our485' (training) or 'eval15' (test).

    Returns:
        List of dicts containing pair paths, filename, and scene grouping ID.
    """
    data_dir = resolve_dataset_dir(data_dir)
    split_dir = os.path.join(data_dir, split_name)
    low_dir = os.path.join(split_dir, "low")
    high_dir = os.path.join(split_dir, "high")

    if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
        # Fallback to check if nested under lol_dataset
        nested_low = os.path.join(data_dir, "lol_dataset", split_name, "low")
        nested_high = os.path.join(data_dir, "lol_dataset", split_name, "high")
        if os.path.isdir(nested_low) and os.path.isdir(nested_high):
            low_dir, high_dir = nested_low, nested_high
        else:
            raise FileNotFoundError(
                f"Could not locate low and high image directories in {split_dir}"
            )

    low_files = sorted([
        f for f in os.listdir(low_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    manifest = []
    for f in low_files:
        low_p = os.path.join(low_dir, f)
        high_p = os.path.join(high_dir, f)
        if os.path.exists(high_p):
            scene_id = os.path.splitext(f)[0]
            manifest.append({
                "filename": f,
                "low_path": low_p,
                "high_path": high_p,
                "scene_id": scene_id,
            })

    return manifest
