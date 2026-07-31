"""A path-operator vessel filter, in the RORPO family.

This is NOT the reference RORPO. The reference is a C++ library
(https://github.com/path-openings/RORPO) that needs a compiler this sandbox does
not have; scripts/build_rorpo.md sets it up on a machine that does. This module
is a faithful-in-spirit, pure-numpy stand-in so the *comparison* — a
length-based, contrast-preserving operator against a contrast-based Hessian one —
can be run now. Where a number matters, run the reference and compare.

Why a path operator answers the dim-vessel problem
--------------------------------------------------
A Hessian filter scores a vessel by its second derivative, which scales with
contrast, so a dim vessel scores low however vessel-shaped it is. A morphological
opening by a long line structuring element instead keeps a structure if it
*contains a long line* in some orientation, and — being a grey opening — keeps
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
    """A 1-pixel-wide line structuring element of the given length and angle."""
    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle), np.sin(angle)
    coords = [(int(round(r * dy)), int(round(r * dx)))
              for r in np.arange(-(length // 2), length // 2 + 1)]
    coords = sorted(set(coords))
    rows = [c[0] for c in coords]
    cols = [c[1] for c in coords]
    r0, c0 = min(rows), min(cols)
    fp = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in coords:
        fp[r - r0, c - c0] = True
    return fp


def path_opening_response(image, length, n_orientations=8, gap=1):
    """Per-orientation openings by a line of `length`, with a small gap tolerance.

    Returns an array of shape (n_orientations, *image.shape). Each plane is the
    grey opening along one orientation: bright where a line of that length and
    orientation fits, and — being an opening — at the original intensity there.
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
    peak = float(best.max())
    return best / peak if peak > 0 else best
