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

def slab_average(
    volume: np.ndarray,
    slice_index: int,
    axis: int = 2,
    radius: int = 1,
    sigma: float = None) -> np.ndarray:
    """
    Weighted slab averaging for DBS visualization.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (X, Y, Z) in canonical orientation.
    slice_index : int
        Index of the central slice.
    axis : int
        Slice axis: 0 (sagittal), 1 (coronal), 2 (axial).
    radius : int
        Number of slices on each side (radius=1 -> 3-slice slab).
    sigma : float or None
        Gaussian sigma in slice units. If None, defaults to radius / 2.

    Returns
    -------
    slab_2d : np.ndarray
        Weighted averaged 2D slice.
    """

    if sigma is None:
        sigma = max(radius / 2.0, 0.5)

    # Slice indices (clipped to volume bounds)
    idxs = np.arange(slice_index - radius, slice_index + radius + 1)
    idxs = np.clip(idxs, 0, volume.shape[axis] - 1)

    # Gaussian weights
    offsets = idxs - slice_index
    weights = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()

    # Extract and accumulate
    slab = None
    for i, w in zip(idxs, weights):
        slice_2d = np.take(volume, i, axis=axis)
        if slab is None:
            slab = w * slice_2d.astype(np.float32)
        else:
            slab += w * slice_2d

    return slab

def axial_slab_average(volume:np.ndarray, slice_index:int) -> np.ndarray:
    if 0 < slice_index < volume.shape[2] - 1:
        # weights = np.array([1, 2, 1], dtype=volume.dtype)
        # weights /= weights.sum()
        weights = np.array([0.23899427, 0.52201147, 0.23899427])
        slice2d = weights[0] * volume[:, :, slice_index -1] + weights[1] * volume[:, :, slice_index] + weights[2] * volume[:, :, slice_index +1]
    elif slice_index == 0:
        weights = np.array([0.68594946, 0.31405054])
        slice2d = weights[0] * volume[:, :, 0] + weights[1] * volume[:, :, 1]
    else: # slice_index == volume.shape[2] - 1
        weights = np.array([0.31405054, 0.68594946])
        slice2d = weights[0] * volume[:, :, slice_index -1] + weights[1] * volume[:, :, slice_index]
    return np.clip(slice2d, 0., 1.)

def precompute_slab_volume(volume, axis=2, radius=1, sigma=None) -> np.ndarray:
    """
    Precompute slab-averaged volume along a given axis.
    """
    slab_vol = np.zeros_like(volume, dtype=np.float32)

    for i in range(volume.shape[axis]):
        slab_vol.take(
            indices=i,
            axis=axis,
            out=slab_vol.take(indices=i, axis=axis)
        )
        slab_vol[(slice(None),)*axis + (i,)] = slab_average(
            volume,
            slice_index=i,
            axis=axis,
            radius=radius,
            sigma=sigma
        )

    return slab_vol

def roi_denoise(img, roi_mask, sigma=0.6):
    smooth = gaussian_filter(img, sigma)
    return img * (1 - roi_mask) + smooth * roi_mask

def roi_clahe(
    img,
    roi_mask,
    kernel_size=8,
    clip_limit=0.032,
    blend=0.5
):
    """
    Noise-aware, blended CLAHE for deep nuclei.
    """
    clahe_img = equalize_adapthist(
        img,
        kernel_size=(kernel_size, kernel_size),
        clip_limit=clip_limit
    )

    blended = (1 - blend) * img + blend * clahe_img
    return img * (1 - roi_mask) + blended * roi_mask

def roi_dog_enhance(
    img,
    roi_mask,
    sigma_low=0.6,
    sigma_high=1.2,
    amount=0.4
):
    """
    Structure-preserving enhancement for deep nuclei.
    """

    g1 = gaussian_filter(img, sigma_low)
    g2 = gaussian_filter(img, sigma_high)

    dog = g1 - g2

    out = img.copy()
    out[roi_mask] = img[roi_mask] + amount * dog[roi_mask]

    return out

def roi_window_tighten(
    img: np.ndarray,
    roi_mask: np.ndarray,
    low_pct=20,
    high_pct=80
):
    """
    Gentle contrast tightening inside ROI.
    """
    roi_mask = roi_mask.astype(bool)
    vals = img[roi_mask]
    # print(img.shape, np.min(img), np.max(img), img.dtype)
    # print(roi_mask.shape, np.min(roi_mask), np.max(roi_mask), roi_mask.dtype)
    # print(vals.shape, np.min(vals), np.max(vals), vals.dtype)
    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)
    # print("lo, hi: ", lo, hi)
    out = img.copy()
    out[roi_mask] = np.clip(
        (img[roi_mask] - lo) / (hi - lo + 1e-8),
        0.0,
        1.0
    )
    # print(out.shape, np.min(out), np.max(out), out.dtype)
    return out

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
    clahe_clip: float = 0.012,
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
    img = roi_denoise(img, roi_mask, sigma=0.6)
    img = roi_clahe(img, roi_mask, kernel_size=clahe_kernel, clip_limit=clahe_clip)
    img = roi_dog_enhance(img, roi_mask, amount=0.35)
    img = roi_window_tighten(img, roi_mask)
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
