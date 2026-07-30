"""Agreement between two channel masks.

The archive reports eight numbers. Measured on realistic mask statistics (a few
percent vessel area fraction), four of them carry almost no information:

  - `mean_squared_error` and `hamming` are the same quantity on binary masks:
    both reduce to the fraction of differing voxels.
  - `rand_index` on two classes is pixel accuracy, which is dominated by the
    empty background. Two completely unrelated masks still score about 0.91, so
    nearly its whole range is spent on the fact that most of a slice is not vessel.
  - `ssim` is built for continuous perceptual images. On thresholded masks its
    value depends strongly on the spatial structure rather than on agreement:
    measured on unrelated mask pairs it ranges from near 0 for uncorrelated
    speckle to above 0.8 for realistically structured vasculature. Either way it
    fails to separate unrelated masks from related ones, which is the job.

What is kept here is Dice, Jaccard, precision and recall — which do span their
range — plus `cl_dice`, which compares skeletons instead of voxels and so
notices when two channels disagree about *connectivity*. Two masks can reach
Dice 0.9 while disagreeing about whether a capillary bed is joined up, and for
vessels that difference is usually the interesting part.

Asymmetry matters: `precision` and `recall` swap when you swap the arguments,
everything else here is symmetric. Fix an argument order and state it.

All functions binarise their inputs, so numeric masks cannot silently change the
result the way they can in the archive's `iou`.
"""

import numpy as np
from skimage.morphology import skeletonize

__all__ = ["dice", "jaccard", "precision", "recall", "cl_dice", "area_fraction",
           "agreement", "agreement_by_calibre"]

EPSILON = 1e-10


def _binary(mask):
    return np.asarray(mask).astype(bool)


def _pair(a, b):
    a, b = _binary(a), _binary(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def dice(a, b):
    """Overlap as a fraction of mean mask size. Symmetric."""
    a, b = _pair(a, b)
    return float((2.0 * (a & b).sum() + EPSILON) / (a.sum() + b.sum() + EPSILON))


def jaccard(a, b):
    """Intersection over union. Symmetric.

    Named `jaccard` rather than `iou` to make the break with the archive obvious:
    `velazquez_rivera_2025.metrics.iou` computes the union with `+`, which is
    logical OR on boolean masks but arithmetic addition on numeric ones. This one
    binarises first, so it agrees with the archive on boolean input and is simply
    correct on everything else.
    """
    a, b = _pair(a, b)
    return float(((a & b).sum() + EPSILON) / ((a | b).sum() + EPSILON))


def precision(a, b):
    """Fraction of `a` that is also in `b`. NOT symmetric."""
    a, b = _pair(a, b)
    return float(((a & b).sum() + EPSILON) / (a.sum() + EPSILON))


def recall(a, b):
    """Fraction of `b` that is also in `a`. NOT symmetric."""
    a, b = _pair(a, b)
    return float(((a & b).sum() + EPSILON) / (b.sum() + EPSILON))


def cl_dice(a, b):
    """Topology-aware agreement: how much of each skeleton lies inside the other mask.

    The harmonic mean of two centreline scores — the fraction of `a`'s skeleton
    contained in `b`, and the fraction of `b`'s skeleton contained in `a`. Both
    terms swap when the arguments swap, so the result is symmetric, which is what
    you want when neither channel is ground truth.

    Sensitive to breaks and spurious bridges in a way voxel overlap is not: a
    single-voxel gap severing a long vessel barely moves Dice but removes that
    whole branch's contribution here.

    Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
    for Tubular Structure Segmentation", CVPR 2021.
    """
    a, b = _pair(a, b)
    skeleton_a, skeleton_b = skeletonize(a), skeletonize(b)
    if not skeleton_a.any() or not skeleton_b.any():
        return 0.0
    topology_precision = (skeleton_a & b).sum() / skeleton_a.sum()
    topology_sensitivity = (skeleton_b & a).sum() / skeleton_b.sum()
    total = topology_precision + topology_sensitivity
    if total == 0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / total)


def area_fraction(mask, roi=None):
    """Fraction of the region of interest occupied by the mask.

    Report this per channel alongside any agreement score. Two channels can agree
    well while both over- or under-segmenting, and only the raw fractions show it.
    """
    mask = _binary(mask)
    if roi is None:
        return float(mask.mean())
    roi = _binary(roi)
    total = roi.sum()
    if total == 0:
        return float("nan")
    return float((mask & roi).sum() / total)


def agreement(a, b, roi=None, include_topology=True):
    """The full report for one pair of channel masks.

    Args:
        a, b: channel masks. Order matters for precision and recall; `a` is the
            first argument in both, so `precision` asks how much of `a` is in `b`.
        roi: optional brain mask. Applied to both channels before scoring.
        include_topology: compute `cl_dice`, which requires skeletonising both
            masks and dominates the cost on large volumes.

    Returns:
        dict of metric name to value, including each channel's area fraction.
    """
    a, b = _pair(a, b)
    if roi is not None:
        roi = _binary(roi)
        a, b = a & roi, b & roi

    report = {
        "dice": dice(a, b),
        "jaccard": jaccard(a, b),
        "precision_a_in_b": precision(a, b),
        "recall_b_in_a": recall(a, b),
        "area_fraction_a": area_fraction(a, roi),
        "area_fraction_b": area_fraction(b, roi),
        "voxels_a": int(a.sum()),
        "voxels_b": int(b.sum()),
    }
    if include_topology:
        report["cl_dice"] = cl_dice(a, b)
    return report


def agreement_by_calibre(reference, other, edges, spacing=None):
    """Recall broken down by vessel radius.

    Agreement is almost always far worse on capillaries than on penetrating
    vessels, and a single pooled Dice hides that.

    Each voxel is assigned the radius of the vessel it belongs to — the distance
    transform evaluated at the nearest centreline point — rather than its own
    distance to the background. That distinction matters: a thick vessel contains
    voxels at every distance from 1 up to its radius, so binning by each voxel's
    own distance would smear a single thick vessel across every calibre bin.

    Args:
        reference: the mask whose vessels are being described.
        other: the mask being asked whether it found them.
        edges: bin edges for radius, in physical units if `spacing` is given.
        spacing: per-axis voxel size, passed through to the distance transform.

    Returns:
        List of dicts with the bin bounds, voxel count, and recall in that bin.
        Bins containing no reference voxels report a recall of nan.
    """
    import scipy.ndimage as ndi

    reference, other = _pair(reference, other)
    edges = [float(e) for e in edges]
    if len(edges) < 2 or any(y <= x for x, y in zip(edges, edges[1:])):
        raise ValueError("edges must be increasing and contain at least two values")

    distance = ndi.distance_transform_edt(reference, sampling=spacing)
    skeleton = skeletonize(reference)
    if not skeleton.any():
        return [{"radius_low": low, "radius_high": high, "voxels": 0,
                 "recall": float("nan")} for low, high in zip(edges, edges[1:])]

    # Propagate each centreline point's radius outward to the voxels it owns.
    _, indices = ndi.distance_transform_edt(~skeleton, sampling=spacing,
                                            return_indices=True)
    vessel_radius = distance[tuple(indices)]

    radius = vessel_radius[reference]
    captured = other[reference]

    rows = []
    for low, high in zip(edges, edges[1:]):
        in_bin = (radius >= low) & (radius < high)
        count = int(in_bin.sum())
        rows.append({
            "radius_low": low,
            "radius_high": high,
            "voxels": count,
            "recall": float(captured[in_bin].mean()) if count else float("nan"),
        })
    return rows
