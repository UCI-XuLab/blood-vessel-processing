"""Threshold sensitivity analysis.

The question a reviewer will ask about a channel-agreement result is whether it
survives the choice of threshold. Answering it costs one sweep: segment both
channels across a range of cuts, score each, and look at whether the conclusion
moves.

A conclusion that holds across a broad plateau is a finding. One that only holds
at a particular threshold is a tuning artefact, and it is much better to discover
that here than in review.
"""

import csv

import numpy as np

from .metrics import agreement
from .threshold import segment

__all__ = ["threshold_sweep", "stability", "write_csv"]


def threshold_sweep(vesselness_a, vesselness_b, thresholds, roi=None, ratio=0.5,
                    min_size=10, area_threshold=2000, closing_radius=1,
                    include_topology=True, progress=None):
    """Score channel agreement across a range of seeding thresholds.

    Args:
        vesselness_a, vesselness_b: responses for the two channels. Use the same
            filter and parameters for both; asymmetric preprocessing shows up as
            an asymmetry between precision and recall that has nothing to do with
            the biology.
        thresholds: seeding ("high") thresholds to try.
        roi: optional brain mask applied to both channels.
        ratio: growing threshold as a fraction of the seeding threshold, so the
            hysteresis pair scales together across the sweep. 0.5 is a reasonable
            starting point; set it to 1.0 to reduce to a single global cut and
            reproduce the archive's behaviour.
        progress: optional callable taking (index, total), for a progress bar.

    Returns:
        List of dicts, one per threshold, each carrying the threshold pair and
        every metric from `agreement`.
    """
    vesselness_a = np.asarray(vesselness_a)
    vesselness_b = np.asarray(vesselness_b)
    if vesselness_a.shape != vesselness_b.shape:
        raise ValueError(f"shape mismatch: {vesselness_a.shape} vs {vesselness_b.shape}")
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must lie in (0, 1], got {ratio}")

    thresholds = [float(t) for t in thresholds]
    rows = []
    for index, high in enumerate(thresholds):
        low = high * ratio
        mask_a = segment(vesselness_a, low, high, roi, min_size, area_threshold,
                         closing_radius)
        mask_b = segment(vesselness_b, low, high, roi, min_size, area_threshold,
                         closing_radius)
        row = {"threshold_high": high, "threshold_low": low}
        row.update(agreement(mask_a, mask_b, roi, include_topology))
        rows.append(row)
        if progress is not None:
            progress(index + 1, len(thresholds))
    return rows


def stability(rows, key="dice", tolerance=0.05):
    """Find the widest run of thresholds over which `key` barely moves.

    Args:
        rows: output of `threshold_sweep`.
        key: metric to examine.
        tolerance: how much the metric may vary within a plateau, in absolute units.

    Returns:
        dict describing the plateau: its threshold bounds, the metric's range
        there, and what fraction of the swept thresholds it covers. A plateau
        spanning most of the sweep means the conclusion does not hinge on the
        threshold; a narrow one means it does.
    """
    if not rows:
        raise ValueError("no sweep rows to analyse")
    values = [row[key] for row in rows]
    thresholds = [row["threshold_high"] for row in rows]

    best = (0, 0)
    start = 0
    for end in range(len(values)):
        while max(values[start:end + 1]) - min(values[start:end + 1]) > tolerance:
            start += 1
        if end - start > best[1] - best[0]:
            best = (start, end)

    low_index, high_index = best
    window = values[low_index:high_index + 1]
    return {
        "metric": key,
        "tolerance": tolerance,
        "threshold_low": thresholds[low_index],
        "threshold_high": thresholds[high_index],
        "value_min": min(window),
        "value_max": max(window),
        "value_median": float(np.median(window)),
        "coverage": (high_index - low_index + 1) / len(values),
    }


def write_csv(rows, path):
    """Write sweep rows to CSV, one row per threshold."""
    if not rows:
        raise ValueError("no sweep rows to write")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
