"""Combining several segmentations, and measuring where they disagree.

With no hand-annotated ground truth, no single method can be shown to be right.
Running several and looking at where they differ is the honest substitute: it
does not tell you which is correct, but it tells you where the answer is
uncertain — and that map is useful in its own right. Disagreement concentrates on
capillaries and at junctions, which is exactly where a vesselness filter is
weakest and where a follow-up claim is most likely to be contested.

A caution about consensus: agreement between methods is not evidence of accuracy
when the methods share a failure mode. Three Hessian filters at different scales
will agree confidently that a bifurcation is not a vessel, because suppressing
junctions is intrinsic to the approach rather than incidental to any one filter.
Ensemble across *different* families — a Hessian filter, a learned model, an
intensity threshold — or the consensus mostly measures how similar the members
are. `redundancy` reports that similarity so it can be checked rather than assumed.
"""

import numpy as np

__all__ = ["vote_fraction", "consensus", "disagreement_map", "pairwise_agreement",
           "redundancy"]


def _stack(masks):
    if len(masks) < 2:
        raise ValueError(f"need at least two masks, got {len(masks)}")
    arrays = [np.asarray(m).astype(bool) for m in masks]
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f"masks have differing shapes: {sorted(shapes)}")
    return np.stack(arrays, axis=0)


def vote_fraction(masks, weights=None):
    """Fraction of methods calling each voxel vessel, in [0, 1].

    Args:
        masks: sequence of boolean arrays.
        weights: optional per-method weights. Use them when the members are not
            equally trustworthy, but note that choosing weights without ground
            truth is itself a judgement — record how they were chosen.

    Returns:
        Float array of the same shape as one mask.
    """
    stack = _stack(masks)
    if weights is None:
        return stack.mean(axis=0, dtype=np.float64)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (stack.shape[0],):
        raise ValueError(f"expected {stack.shape[0]} weights, got {weights.shape}")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    weights = weights / weights.sum()
    return np.tensordot(weights, stack.astype(np.float64), axes=(0, 0))


def consensus(masks, rule="majority", weights=None):
    """Combine masks into one.

    Args:
        rule: "majority" keeps voxels more than half the methods claim;
            "union" keeps anything any method found, maximising recall at the
            cost of precision; "intersection" keeps only unanimous voxels, the
            reverse trade. "majority" is the default because union and
            intersection both let a single badly-tuned member dominate.
        weights: passed through to `vote_fraction`.

    Returns:
        Boolean mask.
    """
    fraction = vote_fraction(masks, weights)
    if rule == "majority":
        return fraction > 0.5
    if rule == "union":
        return fraction > 0.0
    if rule == "intersection":
        return fraction >= 1.0 - 1e-9
    raise ValueError(f"unknown rule {rule!r}")


def disagreement_map(masks, weights=None):
    """Per-voxel uncertainty, 0 where methods are unanimous and 1 where split.

    Peaks at an even split and falls to zero at unanimity either way, so a voxel
    everyone rejects scores the same as one everyone accepts. Useful as an
    overlay: the regions that light up are where a reviewer should be shown
    evidence rather than a number.
    """
    fraction = vote_fraction(masks, weights)
    return (2.0 * np.minimum(fraction, 1.0 - fraction)).astype(np.float64)


def pairwise_agreement(masks, names=None, metric="dice"):
    """Agreement between every pair of methods.

    Returns:
        (matrix, names). The matrix is symmetric with ones on the diagonal.
    """
    from . import metrics as metric_module

    functions = {"dice": metric_module.dice, "jaccard": metric_module.jaccard,
                 "cl_dice": metric_module.cl_dice}
    if metric not in functions:
        raise ValueError(f"metric must be one of {sorted(functions)}, got {metric!r}")
    score = functions[metric]

    stack = _stack(masks)
    count = stack.shape[0]
    names = list(names) if names is not None else [f"method_{i}" for i in range(count)]
    if len(names) != count:
        raise ValueError(f"expected {count} names, got {len(names)}")

    matrix = np.eye(count, dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            matrix[i, j] = matrix[j, i] = score(stack[i], stack[j])
    return matrix, names


def redundancy(masks, names=None, metric="dice", threshold=0.9):
    """Flag ensemble members that are near-duplicates of each other.

    An ensemble of highly correlated members gives a falsely confident consensus:
    the vote is nearly unanimous everywhere, the disagreement map is empty, and
    none of it is evidence. This reports the mean off-diagonal agreement and names
    the pairs above `threshold`, so the ensemble's diversity is something you
    check rather than assume.
    """
    matrix, names = pairwise_agreement(masks, names, metric)
    count = len(names)
    off_diagonal = matrix[~np.eye(count, dtype=bool)]

    duplicates = [
        (names[i], names[j], float(matrix[i, j]))
        for i in range(count) for j in range(i + 1, count)
        if matrix[i, j] >= threshold
    ]
    return {
        "metric": metric,
        "mean_agreement": float(off_diagonal.mean()),
        "min_agreement": float(off_diagonal.min()),
        "threshold": threshold,
        "near_duplicate_pairs": duplicates,
        "diverse": not duplicates,
    }
