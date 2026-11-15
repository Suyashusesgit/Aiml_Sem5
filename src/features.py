"""
Feature Extraction Module
Implements the seven handcrafted physical features described in Chapter 15 & 22 of the handbook:
1. Brightness (mean greyscale intensity mu)
2. Contrast (standard deviation sigma)
3. Entropy (Shannon entropy of 256-bin intensity histogram)
4. Noise estimator (Laplacian MAD in flattest decile: 1.4826 * median(|L(x)|))
5. Dark Channel Prior (mean of local patch-eroded minimum across color channels)
6. Edge Statistics (Sobel gradient magnitude mean and 90th percentile)
7. Colourfulness (Hasler-Suesstrunk metric across opponent color channels)

Additional engineered interaction features for ablation studies:
- Contrast-to-brightness ratio: sigma / (mu + eps)
- Entropy-noise product: H * sigma_n

CRITICAL RULE:
All features are computed STRICTLY on the degraded input image without any access
to ground truth or candidate enhanced outputs.
"""

from typing import Dict, Any, List
import cv2
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "brightness",
    "contrast",
    "entropy",
    "noise",
    "dark_channel",
    "edge_mean",
    "edge_p90",
    "colourfulness",
]

ENGINEERED_FEATURE_NAMES = [
    "contrast_brightness_ratio",
    "entropy_noise_product",
]

# The 10 canonical physical descriptors (He et al., Hasler-Suesstrunk, Shannon)
PHYSICAL_10_DESCRIPTORS = [
    "brightness",
    "contrast",
    "entropy",
    "noise",
    "dark_channel",
    "air_light_transmission",
    "dynamic_range_ratio",
    "edge_mean",
    "edge_p90",
    "colourfulness",
]


# Forbidden column substrings per Appendix A
FORBIDDEN_COLUMN_SUBSTRINGS = [
    "psnr", "ssim", "label", "oracle", "uid", "name", "path", "regret"
]


def validate_feature_dataframe(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Enforces anti-leakage rule: refuses feature columns containing any forbidden substring.
    """
    for col in feature_cols:
        col_lower = col.lower()
        for forbidden in FORBIDDEN_COLUMN_SUBSTRINGS:
            if forbidden in col_lower:
                raise ValueError(
                    f"CRITICAL LEAKAGE DETECTED: feature column '{col}' contains forbidden substring '{forbidden}'"
                )


def compute_immerkaer_noise(gray: np.ndarray, patch_size: int = 16, flat_fraction: float = 0.1) -> float:
    """
    Immerkaer (1996) fast noise estimator computed across the flattest 10% of 16x16 patches.
    Per Appendix A: noise (Immerkaer mask on the flattest 10% of 16x16 patches).
    """
    h, w = gray.shape
    H = (h // patch_size) * patch_size
    W = (w // patch_size) * patch_size
    if H < patch_size or W < patch_size:
        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
        resp = cv2.filter2D(gray, -1, kernel)
        return float(np.mean(np.abs(resp[1:-1, 1:-1])) * np.sqrt(np.pi / 2.0) / 6.0)

    cropped = gray[:H, :W].astype(np.float32)
    patches = cropped.reshape(H // patch_size, patch_size, W // patch_size, patch_size).swapaxes(1, 2).reshape(-1, patch_size, patch_size)
    variances = np.var(patches, axis=(1, 2))
    
    n_flat = max(1, int(len(variances) * flat_fraction))
    flat_indices = np.argsort(variances)[:n_flat]
    flat_patches = patches[flat_indices]

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    laplacians = []
    for p in flat_patches:
        resp = cv2.filter2D(p, -1, kernel)
        laplacians.append(np.mean(np.abs(resp[1:-1, 1:-1])))

    sigma_n = float(np.mean(laplacians) * np.sqrt(np.pi / 2.0) / 6.0)
    return sigma_n


def air_light_transmission(img: np.ndarray, patch_size: int = 15, percentile: float = 99.9) -> float:
    """
    Estimates atmospheric transmission t from Dark Channel Prior (He et al., 2009).
    t = 1.0 - 0.95 * (dark / A), clipped to [0, 1].
    """
    img_f = img.astype(np.float32) / 255.0
    dark = np.min(img_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(dark, kernel)
    A = float(np.percentile(dark, percentile))
    t = 1.0 - 0.95 * (dark / (A + 1e-6))
    return float(np.clip(t, 0.0, 1.0).mean())


def dynamic_range_ratio(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> float:
    """
    Dynamic range ratio: p99 / (p1 + eps). Captures intensity stretching across dynamic range.
    """
    low_val = float(np.percentile(gray, low_pct))
    high_val = float(np.percentile(gray, high_pct))
    return float(high_val / (low_val + 1e-4))


def extract_features(
    img: np.ndarray,
    include_engineered: bool = True,
    include_all_physical: bool = False
) -> Dict[str, float]:
    """
    Computes handcrafted physical degradation descriptors for a single degraded image.

    Args:
        img: Input image in BGR format (uint8, HxWx3).
        include_engineered: If True, also computes interaction features (sigma/mu, H*sigma_n).
        include_all_physical: If True, also computes extended physical descriptors (air_light_transmission, dynamic_range_ratio).

    Returns:
        Dictionary mapping feature names to scalar float values.
    """
    if img is None:
        raise ValueError("Input image is None")

    # Ensure 3-channel BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 1 & 2: Greyscale intensity conversion, Mean (Brightness) & Std (Contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # 3: Shannon Entropy (0-8 bits for 8-bit image)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-12)
    nonzero_p = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p)))

    # 4: Noise estimation (Immerkaer mask on flattest 10% of 16x16 patches per Appendix A)
    noise = float(compute_immerkaer_noise(gray, patch_size=16, flat_fraction=0.10))

    # 5: Dark Channel Prior (He et al. 2011)
    # Minimum across B, G, R channels eroded with 15x15 rectangular kernel
    min_channel = np.min(img, axis=2)
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel_map = cv2.erode(min_channel, kernel)
    dark_channel = float(dark_channel_map.mean())

    # 6: Edge statistics (Sobel gradient magnitude mean and 90th percentile)
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sob_x ** 2 + sob_y ** 2)
    edge_mean = float(grad_mag.mean())
    edge_p90 = float(np.percentile(grad_mag, 90.0))

    # 7: Colourfulness metric (Hasler & Suesstrunk, 2003)
    # Opponent color spaces: rg = R - G, yb = 0.5*(R + G) - B
    # OpenCV stores images in BGR format
    b_chan = img[:, :, 0].astype(np.float32)
    g_chan = img[:, :, 1].astype(np.float32)
    r_chan = img[:, :, 2].astype(np.float32)

    rg = r_chan - g_chan
    yb = 0.5 * (r_chan + g_chan) - b_chan

    sigma_rg = float(rg.std())
    sigma_yb = float(yb.std())
    mu_rg = float(rg.mean())
    mu_yb = float(yb.mean())

    std_root = np.sqrt(sigma_rg ** 2 + sigma_yb ** 2)
    mean_root = np.sqrt(mu_rg ** 2 + mu_yb ** 2)
    colourfulness = float(std_root + 0.3 * mean_root)

    feat_dict = {
        "brightness": brightness,
        "contrast": contrast,
        "entropy": entropy,
        "noise": noise,
        "dark_channel": dark_channel,
        "edge_mean": edge_mean,
        "edge_p90": edge_p90,
        "colourfulness": colourfulness,
    }

    if include_engineered:
        feat_dict["contrast_brightness_ratio"] = float(contrast / (brightness + 1e-4))
        feat_dict["entropy_noise_product"] = float(entropy * noise)

    if include_all_physical:
        feat_dict["air_light_transmission"] = air_light_transmission(img)
        feat_dict["dynamic_range_ratio"] = dynamic_range_ratio(gray)

    return feat_dict
