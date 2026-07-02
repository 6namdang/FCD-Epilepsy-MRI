"""
Slice extraction + normalization helpers for displaying 3D NIfTI
volumes (as numpy arrays) inside Streamlit.
"""
import numpy as np


def normalize(slice_2d: np.ndarray) -> np.ndarray:
    """Scale a 2D slice to 0-255 uint8 for display (1st-99th percentile stretch)."""
    slice_2d = np.nan_to_num(slice_2d)
    lo, hi = np.percentile(slice_2d, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    clipped = np.clip(slice_2d, lo, hi)
    scaled = ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)
    return scaled


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume along the given axis."""
    index = int(np.clip(index, 0, volume.shape[axis] - 1))
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:
        return volume[:, :, index]


def overlay_rgb(base_slice: np.ndarray, overlay_slice: np.ndarray = None,
                 alpha: float = 0.4, overlay_color=(255, 60, 60)) -> np.ndarray:
    """
    Blend a grayscale base slice with a colored overlay (e.g. a lesion mask)
    into an RGB uint8 image. Overlay is treated as a binary label, not an
    intensity image -- thresholded rather than alpha-blended as pixel values.
    """
    base_norm = normalize(base_slice)
    rgb = np.stack([base_norm] * 3, axis=-1).astype(np.float32)

    if overlay_slice is not None and overlay_slice.max() > 0:
        mask = overlay_slice > (overlay_slice.max() * 0.5)
        for c in range(3):
            rgb[..., c] = np.where(
                mask, (1 - alpha) * rgb[..., c] + alpha * overlay_color[c], rgb[..., c]
            )

    return rgb.astype(np.uint8)