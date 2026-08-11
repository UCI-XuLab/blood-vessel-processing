"""A path-operator vessel filter, in the RORPO family.

This is NOT the reference RORPO. The reference is a C++ library
(https://github.com/path-openings/RORPO) that needs a compiler this sandbox does
not have; scripts/build_rorpo.md sets it up on a machine that does. This module
is a faithful-in-spirit, pure-numpy stand-in so the *comparison* - a
length-based, contrast-preserving operator against a contrast-based Hessian one -
can be run now. Where a number matters, run the reference and compare.

Why a path operator answers the dim-vessel problem
--------------------------------------------------
A Hessian filter scores a vessel by its second derivative, which scales with
contrast, so a dim vessel scores low however vessel-shaped it is. A morphological
opening by a long line structuring element instead keeps a structure if it
*contains a long line* in some orientation, and - being a grey opening - keeps
that structure at its own intensity. A dim but genuinely elongated vessel
therefore survives with a dim-but-nonzero response that connectivity can then
recover, which is exactly what a Hessian filter cannot give.

The reference RORPO adds two things this approximation keeps in spirit:

  - Robust path openings (a bounded number of gaps allowed along the path), so a
    vessel broken by a small dark gap still survives. Approximated here by a
    light closing before the opening.
  - Ranking the orientation responses to suppress blobs: a blob survives the
    opening in every orientation, a vessel in only the few aligned with it. The
    spread across orientations (max - median) is used here as that selector.
"""

import numpy as np
from skimage.morphology import closing, footprint_rectangle, opening

__all__ = ["line_footprint", "path_opening_response", "path_operator_vesselness"]


def line_footprint(length, angle_deg):
    """A 1-pixel-wide line structuring element of exactly `length` pixels.

    Rasterising by stepping `r` along (cos, sin) and rounding collapses distinct
    steps to the same pixel near the diagonals, so a nominally length-21 line at
    45 degrees came out ~15 pixels - meaning diagonal structures needed a third
    less length to survive the opening than axis-aligned ones. Driving the
    rasterisation from the longer axis, so the span on that axis is length-1,
    gives exactly `length` pixels at every angle (a Bresenham line has
    max(|dr|, |dc|) + 1 pixels).
    """
    from skimage.draw import line

    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)
    span = length - 1
    if abs(dx) >= abs(dy):
        dc = span
        dr = int(round(span * (dy / dx))) if dx != 0 else 0
    else:
        dr = span
        dc = int(round(span * (dx / dy))) if dy != 0 else 0
    rr, cc = line(0, 0, dr, dc)
    rr, cc = rr - rr.min(), cc - cc.min()
    fp = np.zeros((rr.max() + 1, cc.max() + 1), dtype=bool)
    fp[rr, cc] = True
    return fp


def path_opening_response(image, length, n_orientations=8, gap=1):
    """Per-orientation openings by a line of `length`, with a small gap tolerance.

    Returns an array of shape (n_orientations, *image.shape). Each plane is the
    grey opening along one orientation: bright where a line of that length and
    orientation fits, and - being an opening - at the original intensity there.
    """
    image = np.asarray(image, dtype=np.float32)
    if gap > 0:
        # Robust-path stand-in: bridge sub-`gap` dark breaks before opening.
        image = closing(image, footprint_rectangle((gap * 2 + 1, gap * 2 + 1)))
    angles = np.linspace(0, 180, n_orientations, endpoint=False)
    return np.stack([opening(image, line_footprint(length, a)) for a in angles])


def path_operator_vesselness(image, lengths, n_orientations=8, gap=1):
    """Length-based, contrast-preserving vessel response over several scales.

    Args:
        image: 2D array, bright vessels on dark background.
        lengths: line lengths in pixels, one per scale. A vessel survives the
            opening at any length up to its own; take several so both short
            capillary segments and long vessels are kept.
        n_orientations: directions sampled between 0 and 180 degrees.
        gap: gap tolerance in pixels for the robust-path stand-in.

    Returns:
        Response normalised to [0, 1]. High where a pixel lies on an elongated
        structure in few orientations; low on blobs, which survive in all.
    """
    image = np.asarray(image, dtype=np.float32)
    best = np.zeros(image.shape, dtype=np.float32)
    for length in lengths:
        responses = path_opening_response(image, int(length), n_orientations, gap)
        aligned = responses.max(axis=0)           # survives in its own orientation
        isotropic = np.median(responses, axis=0)  # a blob survives in all
        # Elongation: bright where the aligned response exceeds the isotropic one.
        best = np.maximum(best, aligned - isotropic)
    # WARNING: this divides by the image's own maximum, so the [0, 1] scale means
    # something different in every section - the very per-image dependence that
    # vessel_utils removed with reference_lambda. It is here only so the response
    # is thresholdable in isolation. For a fair comparison against Jerman under a
    # fixed dataset-wide threshold, replace this with a dataset-wide reference,
    # exactly as jerman_vesselness does; see build_rorpo.md.
    peak = float(best.max())
    return best / peak if peak > 0 else best
