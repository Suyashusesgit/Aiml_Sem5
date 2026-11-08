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


def extract_features(img: np.ndarray, include_engineered: bool = True) -> Dict[str, float]:
    """
    Computes handcrafted physical degradation descriptors for a single degraded image.

    Args:
        img: Input image in BGR format (uint8, HxWx3).
        include_engineered: If True, also computes interaction features (sigma/mu, H*sigma_n).

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

    # 4: Noise estimation in flat regions (Laplacian MAD)
    # Gradient magnitude to isolate smooth/flat background
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sob_x ** 2 + sob_y ** 2)

    # Flattest decile (bottom 10% gradient magnitude)
    flat_threshold = np.percentile(grad_mag, 10.0)
    flat_mask = grad_mag <= flat_threshold

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    flat_lap = np.abs(lap[flat_mask])
    if len(flat_lap) > 0:
        noise = float(1.4826 * np.median(flat_lap))
    else:
        noise = float(1.4826 * np.median(np.abs(lap)))

    # 5: Dark Channel Prior (He et al. 2011)
    # Minimum across B, G, R channels eroded with 15x15 rectangular kernel
    min_channel = np.min(img, axis=2)
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel_map = cv2.erode(min_channel, kernel)
    dark_channel = float(dark_channel_map.mean())

    # 6: Edge statistics (Sobel gradient magnitude mean and 90th percentile)
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

    return feat_dict
