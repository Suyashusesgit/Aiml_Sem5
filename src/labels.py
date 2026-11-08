"""
Labels and Scoring Module
Implements PSNR, SSIM, and Oracle Label Computation as specified in
Chapters 10, 11, 13, and 22 of the handbook.

- PSNR: Logarithmic fidelity metric in decibels (dB)
- SSIM: Structural Similarity Index (Gaussian window 11x11, sigma=1.5)
- Oracle Decision Rule: argmax PSNR with SSIM tie-breaking within 0.05 dB
- Regret: Non-negative PSNR loss compared to the oracle: s_{a*}(x) - s_{a}(x)
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from src.methods import METHODS


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error across all pixels and channels."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return 100.0  # Cap at 100 dB for identical images
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
    sigma: float = 1.5
) -> float:
    """
    Computes Structural Similarity Index (SSIM) based on Wang et al. (2004).
    Evaluated on greyscale luminance channel with 11x11 Gaussian window.
    """
    if len(img1.shape) == 3:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Gaussian weighting
    mu1 = gaussian_filter(g1, sigma=sigma, mode="reflect")
    mu2 = gaussian_filter(g2, sigma=sigma, mode="reflect")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(g1 * g1, sigma=sigma, mode="reflect") - mu1_sq
    sigma2_sq = gaussian_filter(g2 * g2, sigma=sigma, mode="reflect") - mu2_sq
    sigma12 = gaussian_filter(g1 * g2, sigma=sigma, mode="reflect") - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)

    return float(np.mean(ssim_map))


def score_pair(
    clean: np.ndarray,
    degraded: np.ndarray,
    tie_threshold_db: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates all enhancement methods in the bank against the clean reference.

    Determines:
    - PSNR and SSIM for every candidate method
    - Oracle label (best method by PSNR, with SSIM tie-breaking within 0.05 dB)
    - Oracle PSNR and SSIM

    Args:
        clean: Ground truth normal-light image.
        degraded: Degraded/low-light input image.
        tie_threshold_db: PSNR threshold in dB within which SSIM breaks ties.

    Returns:
        Dictionary with per-method PSNR, SSIM, oracle label, and oracle score.
    """
    row: Dict[str, Any] = {}
    enhanced_outputs = {}

    for name, method_fn in METHODS.items():
        out = method_fn(degraded)
        enhanced_outputs[name] = out
        p = compute_psnr(clean, out)
        s = compute_ssim(clean, out)
        row[f"psnr_{name}"] = p
        row[f"ssim_{name}"] = s

    # Find highest PSNR
    max_psnr = max(row[f"psnr_{name}"] for name in METHODS)

    # Check for near-ties within tie_threshold_db
    candidates = [
        name for name in METHODS
        if (max_psnr - row[f"psnr_{name}"]) <= tie_threshold_db
    ]

    if len(candidates) == 1:
        best_method = candidates[0]
    else:
        # Break tie using SSIM
        best_method = max(candidates, key=lambda n: row[f"ssim_{n}"])

    row["label"] = best_method
    row["oracle_psnr"] = row[f"psnr_{best_method}"]
    row["oracle_ssim"] = row[f"ssim_{best_method}"]

    return row
