"""Remove compact blobs from a vessel mask, keeping elongated structures.

The virus channel carries transduced neuron cell bodies that a 2D vesselness
filter cannot reject (in 2D it has no blobness term). They are, however,
geometrically distinct from vessels: a neuron is a compact roundish blob, a
vessel segment is elongated. This drops connected components that look like the
former and keeps the latter.

The discriminator is elongation, measured two ways per component so a single
odd shape does not decide it:

  - eccentricity of the best-fit ellipse: ~0 for a disc, ~1 for a line.
  - a skeleton-length / area ratio: a thin structure has a long skeleton for its
    area; a blob has a short one.

A component is dropped only if it is compact by BOTH measures AND small enough to
be a cell body (neurons are ~10-25 um; anything larger is kept regardless, since
a big round region is more likely a vessel cross-section or a genuine structure
than a neuron). Every threshold is a parameter, and the defaults are stated in
physical units so they can be reasoned about rather than guessed.

This is a shape prior, not ground truth: it will remove a genuinely round vessel
cross-section (a vessel travelling perpendicular to the plane) along with the
neurons, and keep a neuron that happens to be elongated. Report how much area it
removes so that cost is visible, and treat the result as "vessels minus compact
objects", not "vessels".
"""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

__all__ = ["reject_blobs"]


def reject_blobs(mask, um_per_px, max_blob_um=25.0, min_eccentricity=0.9,
                 min_skeleton_ratio=0.10, return_removed=False):
    """Drop compact components; keep elongated ones.

    Args:
        mask: boolean vessel mask.
        um_per_px: pixel size, for the physical size gate.
        max_blob_um: components with an equivalent diameter above this are always
            kept — too big to be a neuron. Only smaller components are eligible
            for removal.
        min_eccentricity: keep a small component if its ellipse eccentricity is at
            least this (elongated). Below it, the component is a removal candidate.
        min_skeleton_ratio: keep a small component if its skeleton length over
            area (per pixel) is at least this. A thin structure clears it; a blob
            does not. Second, independent elongation test.
        return_removed: also return the mask of what was dropped, for inspection.

    Returns:
        Cleaned boolean mask, or (cleaned, removed) if return_removed.

    A component is removed only when it is small AND round by eccentricity AND
    stubby by skeleton ratio — all three — so a structure that is elongated by
    either measure survives.
    """
    mask = np.asarray(mask, dtype=bool)
    max_blob_px = max_blob_um / um_per_px
    labels = label(mask)
    kept = np.zeros_like(mask)
    removed = np.zeros_like(mask)

    for region in regionprops(labels):
        component = labels == region.label
        equiv_diameter = region.equivalent_diameter  # pixels
        big = equiv_diameter > max_blob_px
        elongated_ellipse = region.eccentricity >= min_eccentricity

        skeleton_len = int(skeletonize(component).sum())
        skeleton_ratio = skeleton_len / max(region.area, 1)
        elongated_skeleton = skeleton_ratio >= min_skeleton_ratio

        # Keep unless small AND round AND stubby.
        if big or elongated_ellipse or elongated_skeleton:
            kept |= component
        else:
            removed |= component

    return (kept, removed) if return_removed else kept
