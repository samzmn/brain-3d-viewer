"""
Correct order:
    Global normalization (volume-level)
    ↓
    Slab averaging (optional but recommended)
    ↓
    ROI denoise (σ ≈ 0.6)
    ↓
    ROI-CLAHE (low clip, large tiles, blended)
    ↓
    Core-ROI sharpening
    ↓
    Sigmoid remap
    ↓
    Overlays
"""
import numpy as np
from skimage.exposure import equalize_adapthist
from scipy.ndimage import gaussian_filter

def compute_volume_normalization(volume, brainmask, pmin=1, pmax=99):
    data = volume[brainmask > 0]
    lo, hi = np.percentile(data, [pmin, pmax])
    return lo, hi

def apply_volume_normalization(volume, lo, hi):
    volume = np.clip(volume, lo, hi)
    return (volume - lo) / (hi - lo + 1e-8)

def roi_denoise(img, roi_mask, sigma=0.6):
    smooth = gaussian_filter(img, sigma)
    return img * (1 - roi_mask) + smooth * roi_mask

def roi_clahe(
    img,
    roi_mask,
    kernel_size=8,
    clip_limit=0.012,
    blend=0.5
):
    """
    Noise-aware, blended CLAHE for deep nuclei.
    """
    clahe_img = equalize_adapthist(
        img[roi_mask],
        kernel_size=(kernel_size, kernel_size),
        clip_limit=clip_limit
    )

    blended = (1 - blend) * img + blend * clahe_img
    return img * (1 - roi_mask) + blended * roi_mask

def roi_unsharp(img, roi_mask, sigma=1.0, amount=0.6):
    """High-frequency emphasis (ROI-weighted)"""
    blurred = gaussian_filter(img, sigma)
    sharp = img + amount * (img - blurred)
    return img * (1 - roi_mask) + sharp * roi_mask

def sigmoid_remap(img, roi_mask, gain=8.0, cutoff=0.5):
    """Intensity remapping (sigmoid)"""
    remapped = 1 / (1 + np.exp(-gain * (img - cutoff)))
    return img * (1 - roi_mask) + remapped * roi_mask

def fuse_overlay(base, overlay, alpha=0.2):
    """Overlay fusion (FGATIR / PCA / FA)"""
    return np.clip(base + alpha * overlay, 0.0, 1.0)

def enhance_slice(
    base_slice: np.ndarray,
    brainmask: np.ndarray,
    roi_mask: np.ndarray,
    core_roi: np.ndarray,
    fgatir: np.ndarray = None,
    pca: np.ndarray = None,
    fa: np.ndarray = None,
    clahe_kernel: int = 16,
    clahe_clip: float = 0.01,
    background_alpha: float = 0.8  # how much to dim non-brain areas
) -> np.ndarray:
    """
    Assumes:
    - base_slice must already be globally normalized [0,1]
    - All masks are aligned to slice
    - Inputs are float32
    """

    img = base_slice.copy()
    
    # --- ROI-restricted enhancement ---
    # img = roi_denoise(img, roi_mask, sigma=0.6)
    # img = roi_clahe(img, roi_mask)
    # img = roi_unsharp(img, core_roi, sigma=1.0, amount=0.6)
    # img = sigmoid_remap(img, roi_mask, gain=8.0, cutoff=0.5)

    # --- Optional multimodal overlays ---
    if fgatir is not None:
        img = fuse_overlay(img, fgatir, alpha=0.25)
    if pca is not None:
        img = fuse_overlay(img, pca, alpha=0.15)
    if fa is not None:
        img = fuse_overlay(img, fa, alpha=0.10)

    # --- Soft background dimming (visual only) ---
    if brainmask is not None:
        img = img * (brainmask + background_alpha * (1 - brainmask))

    return np.clip(img, 0.0, 1.0)
