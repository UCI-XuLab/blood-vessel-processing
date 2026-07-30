"""Measuring accuracy on real data without dense annotation.

Tracing vessels through volumes costs weeks. Classifying a few hundred single
points costs an afternoon, and — done with the right sampling design — yields
unbiased precision and recall with confidence intervals. That is the difference
between "our pipeline is more principled" and "precision 0.94, 95% CI 0.91-0.96".

The design matters more than the effort. Vessels occupy a few percent of a brain
volume, so sampling points uniformly spends 97% of the clicks confirming that
empty parenchyma is empty, and produces an estimate of recall with almost no
information in it. Sampling separately inside and outside the predicted mask, and
reweighting by how large each region actually is, puts the clicks where the
uncertainty is.

Stratified estimation, with N+ voxels predicted vessel and N- predicted background:

    precision = p_pos
    recall    = N+ p_pos / (N+ p_pos + N- p_neg)

where p_pos and p_neg are the fractions of sampled points that are truly vessel
in each stratum. Intervals come from bootstrapping the two strata independently,
so the recall interval correctly widens when few false negatives were seen.
"""

import numpy as np

__all__ = ["stratified_sample", "extract_crops", "estimate_accuracy",
           "agreement_by_depth", "depth_invariance"]


def stratified_sample(predicted, n_points=300, positive_fraction=0.5, roi=None,
                      seed=0):
    """Draw review points from inside and outside the predicted mask.

    Args:
        predicted: the boolean mask being validated.
        n_points: total points to review. 300 is about an afternoon and gives a
            precision interval of roughly +/-0.05 for a segmentation around 90%
            precise. Recall is estimated less tightly than precision at the same
            point count, because it depends on the rarer negative-stratum rate:
            against a phantom at 4000 points the estimator recovers precision to
            about +/-0.05 but recall only to about +/-0.10.
        positive_fraction: share drawn from predicted-vessel voxels. Half and
            half is a reasonable default; raise it if precision matters more than
            recall for your claim.
        roi: restrict sampling, e.g. to the brain mask.

    Returns:
        dict with the sampled coordinates per stratum and the stratum sizes,
        which `estimate_accuracy` needs for the reweighting.
    """
    predicted = np.asarray(predicted).astype(bool)
    if roi is not None:
        roi = np.asarray(roi).astype(bool)
        if roi.shape != predicted.shape:
            raise ValueError(f"roi shape {roi.shape} != mask shape {predicted.shape}")
        inside, outside = predicted & roi, (~predicted) & roi
    else:
        inside, outside = predicted, ~predicted

    n_inside, n_outside = int(inside.sum()), int(outside.sum())
    if n_inside == 0:
        raise ValueError("predicted mask is empty; nothing to validate")
    if n_outside == 0:
        raise ValueError("predicted mask covers everything; nothing to validate")

    rng = np.random.default_rng(seed)
    want_positive = min(int(round(n_points * positive_fraction)), n_inside)
    want_negative = min(n_points - want_positive, n_outside)

    def draw(mask, count):
        coordinates = np.argwhere(mask)
        chosen = rng.choice(len(coordinates), size=count, replace=False)
        return coordinates[chosen]

    return {
        "positive_points": draw(inside, want_positive),
        "negative_points": draw(outside, want_negative),
        "n_positive_voxels": n_inside,
        "n_negative_voxels": n_outside,
    }


def extract_crops(volume, points, size=(9, 33, 33)):
    """Local views around each sample point, for a human to classify.

    Give the reviewer context: a single voxel's intensity is not judgeable, but a
    small neighbourhood showing whether it sits on a tubular structure is. The
    default is deliberately thin in z, matching anisotropic sampling.

    Returns:
        (crops, centres) where crops[i] is the neighbourhood of points[i] and
        centres[i] is the point's index within its crop, since crops near a
        volume edge are clipped and the centre is not at the middle.
    """
    volume = np.asarray(volume)
    points = np.atleast_2d(np.asarray(points, dtype=int))
    if points.size and points.shape[1] != volume.ndim:
        raise ValueError(f"points have {points.shape[1]} dims, volume has {volume.ndim}")

    half = [s // 2 for s in size]
    crops, centres = [], []
    for point in points:
        low = [max(0, p - h) for p, h in zip(point, half)]
        high = [min(n, p + h + 1) for p, h, n in zip(point, half, volume.shape)]
        crops.append(volume[tuple(slice(a, b) for a, b in zip(low, high))])
        centres.append(tuple(int(p - a) for p, a in zip(point, low)))
    return crops, centres


def estimate_accuracy(sample, positive_labels, negative_labels, n_bootstrap=5000,
                      seed=0):
    """Unbiased precision and recall from stratified point review.

    Args:
        sample: the dict from `stratified_sample`.
        positive_labels: for each point drawn from inside the mask, 1 if the
            reviewer judged it truly vessel.
        negative_labels: likewise for points drawn from outside.

    Returns:
        dict of point estimates and 95% intervals. `recall` reweights by the
        stratum sizes, so it reflects the whole volume rather than the sample.
    """
    positive_labels = np.asarray(positive_labels, dtype=float)
    negative_labels = np.asarray(negative_labels, dtype=float)
    for labels, key in ((positive_labels, "positive_points"),
                        (negative_labels, "negative_points")):
        expected = len(sample[key])
        if labels.size != expected:
            raise ValueError(f"expected {expected} labels for {key}, got {labels.size}")
    if positive_labels.size == 0:
        raise ValueError("no reviewed points in the positive stratum")

    n_pos = sample["n_positive_voxels"]
    n_neg = sample["n_negative_voxels"]

    def statistics(pos, neg):
        p_pos = pos.mean()
        p_neg = neg.mean() if neg.size else 0.0
        true_positive = n_pos * p_pos
        false_negative = n_neg * p_neg
        precision = p_pos
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative > 0 else float("nan"))
        dice = (2 * true_positive / (n_pos + true_positive + false_negative)
                if n_pos + true_positive + false_negative > 0 else float("nan"))
        return precision, recall, dice

    precision, recall, dice = statistics(positive_labels, negative_labels)

    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 3))
    for index in range(n_bootstrap):
        pos = rng.choice(positive_labels, size=positive_labels.size, replace=True)
        neg = (rng.choice(negative_labels, size=negative_labels.size, replace=True)
               if negative_labels.size else negative_labels)
        draws[index] = statistics(pos, neg)

    def interval(column):
        values = draws[:, column]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return (float("nan"), float("nan"))
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))

    low_p, high_p = interval(0)
    low_r, high_r = interval(1)
    low_d, high_d = interval(2)
    return {
        "precision": float(precision), "precision_ci": (low_p, high_p),
        "recall": float(recall), "recall_ci": (low_r, high_r),
        "dice": float(dice), "dice_ci": (low_d, high_d),
        "n_reviewed": int(positive_labels.size + negative_labels.size),
        "n_positive_reviewed": int(positive_labels.size),
        "n_negative_reviewed": int(negative_labels.size),
        "estimated_true_voxels": float(n_pos * positive_labels.mean()
                                       + n_neg * (negative_labels.mean()
                                                  if negative_labels.size else 0.0)),
    }


def agreement_by_depth(mask_a, mask_b, n_bins=10, axis=0, roi=None, metric="dice"):
    """Channel agreement as a function of depth.

    The single most informative quality check for a two-channel comparison, and
    it needs no ground truth. Agreement should be flat: the biology does not know
    how deep it is. A downward trend means attenuation, or the correction for it,
    is still leaking into the result — and since wavelengths attenuate at
    different rates, that leak is channel-asymmetric and enters the comparison as
    if it were signal.
    """
    from . import metrics as metric_module

    functions = {"dice": metric_module.dice, "jaccard": metric_module.jaccard,
                 "cl_dice": metric_module.cl_dice}
    if metric not in functions:
        raise ValueError(f"metric must be one of {sorted(functions)}")
    score = functions[metric]

    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    if roi is not None:
        roi = np.asarray(roi).astype(bool)
        mask_a, mask_b = mask_a & roi, mask_b & roi

    depth = mask_a.shape[axis]
    edges = np.linspace(0, depth, n_bins + 1).astype(int)
    rows = []
    for start, stop in zip(edges, edges[1:]):
        if stop <= start:
            continue
        slab = [slice(None)] * mask_a.ndim
        slab[axis] = slice(start, stop)
        slab = tuple(slab)
        a_slab, b_slab = mask_a[slab], mask_b[slab]
        rows.append({
            "start": int(start), "stop": int(stop),
            "centre": float((start + stop) / 2),
            metric: score(a_slab, b_slab) if (a_slab.any() or b_slab.any()) else float("nan"),
            "area_fraction_a": float(a_slab.mean()),
            "area_fraction_b": float(b_slab.mean()),
        })
    return rows


def depth_invariance(rows, metric="dice", tolerance=0.1):
    """Summarise whether agreement drifts with depth, and by how much.

    Returns the fitted trend across the volume. A slope whose total change over
    the full depth exceeds `tolerance` should be treated as an artefact to fix
    rather than a result to report.
    """
    if not rows:
        raise ValueError("no depth bins to analyse")
    centres = np.array([row["centre"] for row in rows], dtype=float)
    values = np.array([row[metric] for row in rows], dtype=float)
    usable = np.isfinite(values)
    if usable.sum() < 2:
        raise ValueError("need at least two usable depth bins")

    slope, intercept = np.polyfit(centres[usable], values[usable], 1)
    total_change = float(slope * (centres[usable].max() - centres[usable].min()))
    return {
        "metric": metric,
        "slope_per_voxel": float(slope),
        "total_change": total_change,
        "value_shallow": float(values[usable][0]),
        "value_deep": float(values[usable][-1]),
        "spread": float(np.nanmax(values[usable]) - np.nanmin(values[usable])),
        "tolerance": tolerance,
        "flat": bool(abs(total_change) <= tolerance),
        "verdict": (
            "flat - agreement does not depend on depth"
            if abs(total_change) <= tolerance else
            f"agreement drifts by {total_change:+.3f} across the volume; suspect "
            f"residual attenuation, which is channel-asymmetric and will bias the "
            f"comparison"
        ),
    }
