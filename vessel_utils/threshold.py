"""Turning a vesselness response into a mask.

The archive applies a single global cut (`vesselness > 230` on a per-image
rescaled response) and then cleans up. A single cut forces one value to serve two
jobs at once: it has to be high enough to reject noise and low enough to keep
faint capillary segments, and no value does both. Hysteresis separates them —
seed on confident vessel, then grow along anything plausibly connected to a seed.
That recovers vessel continuity without admitting isolated noise, and makes the
result far less sensitive to the exact numbers.
"""

import numpy as np
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import (ball, binary_closing, disk, remove_small_holes,
                                remove_small_objects)

__all__ = ["hysteresis_threshold", "otsu_threshold", "clean_mask", "segment"]


def hysteresis_threshold(vesselness, low, high):
    """Keep everything above `low` that connects to something above `high`.

    Args:
        vesselness: response array, any shape.
        low: growing threshold. Voxels above this survive only if connected to a seed.
        high: seeding threshold. Voxels above this are always kept.

    Returns:
        Boolean mask.
    """
    if not low <= high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    return apply_hysteresis_threshold(np.asarray(vesselness), low, high)


def otsu_threshold(vesselness, roi=None):
    """Otsu's cut on the response, optionally restricted to a region of interest.

    Useful as a data-driven starting point for a sweep. Restrict to the brain
    mask: including a large empty background makes Otsu's two-class assumption
    describe tissue-versus-air rather than vessel-versus-parenchyma.
    """
    values = np.asarray(vesselness)
    values = values[np.asarray(roi, dtype=bool)] if roi is not None else values.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0 or values.min() == values.max():
        raise ValueError("cannot compute a threshold from a constant or empty response")
    return float(threshold_otsu(values))


def clean_mask(mask, min_size=10, area_threshold=2000, closing_radius=1):
    """Drop specks, fill pinholes, smooth edges — the archive's post-processing.

    Kept deliberately identical in operation and order to
    `velazquez_rivera_2025.vessels.process_vessels` so that comparisons between
    the two pipelines isolate the filter and threshold rather than confounding
    them with different clean-up. Works in 2D and 3D; the archive was 2D only.
    """
    mask = np.asarray(mask, dtype=bool)
    if min_size:
        mask = remove_small_objects(mask, min_size=min_size)
    if area_threshold:
        mask = remove_small_holes(mask, area_threshold=area_threshold)
    if closing_radius:
        footprint = disk(closing_radius) if mask.ndim == 2 else ball(closing_radius)
        mask = binary_closing(mask, footprint=footprint)
    return mask


def segment(vesselness, low, high, roi=None, min_size=10, area_threshold=2000,
            closing_radius=1):
    """Hysteresis threshold, restrict to a region of interest, then clean up.

    `roi` is applied before clean-up so that objects clipped by the brain mask are
    size-filtered at their clipped size, matching the archive's ordering.
    """
    mask = hysteresis_threshold(vesselness, low, high)
    if roi is not None:
        mask &= np.asarray(roi, dtype=bool)
    return clean_mask(mask, min_size, area_threshold, closing_radius)
