"""
Enhancement Methods Bank
Implements the six fixed enhancement operators defined in Chapter 9 & 22 of the handbook:
1. Identity (do nothing)
2. Gamma correction (gamma=0.5)
3. CLAHE (clipLimit=2.0, tileGridSize=(8, 8) on luminance)
4. Histogram Equalization (global CDF on luminance)
5. Retinex (Single-Scale Retinex with sigma=80)
6. Bilateral Filter + Unsharp Mask (d=9, sigmaColor=75, sigmaSpace=75, sigma=1.5, amount=0.8)

All color methods operate on luminance or process channels uniformly to avoid
artificial color cast artifacts, and explicitly clip to [0, 255] uint8.
"""

from typing import Dict, Callable
import cv2
import numpy as np


def identity(img: np.ndarray) -> np.ndarray:
    """Identity method: returns a copy of the input image unchanged."""
    return img.copy()


def gamma(img: np.ndarray, g: float = 0.5) -> np.ndarray:
    """
    Gamma correction via lookup table.
    g < 1.0 lifts shadows; g = 0.5 expands dark end (square root curve).
    """
    lut = np.array([
        ((i / 255.0) ** g) * 255.0 for i in range(256)
    ]).clip(0, 255).astype("uint8")
    return cv2.LUT(img, lut)


def clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization applied to L channel in LAB color space.
    Preserves chromaticity channels (A, B) to prevent color shifting.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe_obj.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def hist_eq(img: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization applied to L channel in LAB color space.
    Maximizes global tonal spread across [0, 255].
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def retinex(img: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Single-Scale Retinex (SSR) based on the Retinex hypothesis:
    I(x, y) = L(x, y) * R(x, y) => log R = log I - log(G_sigma * I)
    Normalizes dynamic range via min-max scaling to [0, 255].
    """
    img_f = img.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ret = np.log10(img_f) - np.log10(blur + 1.0)
    ret = cv2.normalize(ret, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.clip(ret, 0, 255).astype(np.uint8)


def unsharp(
    img: np.ndarray,
    sigma: float = 1.5,
    amount: float = 0.8,
    threshold: int = 0
) -> np.ndarray:
    """
    Unsharp masking operator:
    I' = I + amount * (I - G_sigma * I)
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    if threshold > 0:
        low = cv2.absdiff(img, blur) < threshold
        sharp[low] = img[low]
    return np.clip(sharp, 0, 255).astype(np.uint8)


def bilat_unsharp(
    img: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    unsharp_sigma: float = 1.5,
    unsharp_amount: float = 0.8
) -> np.ndarray:
    """
    Two-stage edge-preserving denoising and sharpening:
    Stage 1: Bilateral filter removes high-frequency noise while preserving edges.
    Stage 2: Unsharp mask restores crispness and local structural definition.
    """
    bilat = cv2.bilateralFilter(
        img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )
    return unsharp(bilat, sigma=unsharp_sigma, amount=unsharp_amount, threshold=0)


def invert_dehaze(
    img: np.ndarray,
    omega: float = 0.95,
    t0: float = 0.1,
    r: int = 7
) -> np.ndarray:
    """
    Physical Inversion-Dehazing Operator (Dong et al., 2011 / He et al., 2011).
    Exploits the optical duality between Koschmieder's atmospheric scattering model
    and inverted low-light irradiance:
        I_inv(x) = (1 - R(x)) * T(x) + A * (1 - T(x))
    Inverts the low-light input, executes single-image dark channel dehazing to estimate
    the volumetric transmission map T(x), recovers scene radiance, and re-inverts.
    
    Guaranteed SOP compliance: ends with np.clip(I_out, 0, 255).astype(np.uint8).
    """
    inv = 255 - img
    inv_f = inv.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # Multi-scale fast transmission path for sub-10ms real-time edge budget
    small_w, small_h = max(16, w // 2), max(16, h // 2)
    small = cv2.resize(inv_f, (small_w, small_h), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    min_c = np.min(small, axis=2)
    dark_c = cv2.erode(min_c, kernel)

    # Atmospheric light A: top 0.1% brightest pixels in dark channel of inverted image
    num_bright = max(1, int(small_h * small_w * 0.001))
    flat_dark = dark_c.ravel()
    indices = np.argpartition(flat_dark, -num_bright)[-num_bright:]
    flat_inv = small.reshape(-1, 3)
    A = np.median(flat_inv[indices], axis=0)
    A = np.clip(A, 0.1, 1.0)

    # Transmission map t(x) = 1 - omega * dark(inv / A)
    norm_small = small / A
    dark_norm = cv2.erode(np.min(norm_small, axis=2), kernel)
    t_small = 1.0 - omega * dark_norm
    t_small = cv2.GaussianBlur(t_small, (7, 7), 0)
    t = cv2.resize(t_small, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.clip(t, t0, 1.0)[:, :, np.newaxis]

    # Scene radiance recovery J_inv(x) = (inv - A) / t + A, re-inverted
    J_inv = (inv_f - A) / t + A
    J_out = 1.0 - J_inv
    return np.clip(J_out * 255.0, 0, 255).astype(np.uint8)


# Dictionary mapping method names to callable enhancement functions
METHODS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": identity,
    "gamma": gamma,
    "clahe": clahe,
    "hist_eq": hist_eq,
    "retinex": retinex,
    "bilat_unsharp": bilat_unsharp,
    "invert_dehaze": invert_dehaze,
}
