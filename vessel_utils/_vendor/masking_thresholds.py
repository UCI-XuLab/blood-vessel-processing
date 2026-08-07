# Vendored verbatim from UCI-XuLab-RegTools @ 9f2e5fb
# (regtools/utils/masking_thresholds.py). Do not edit here; re-sync by
# re-copying. Ships as a pair with grabcut.py. See _vendor/README.md.
"""Shared threshold helpers for tissue/background masking.

Two consumers derive a tissue threshold from intensity and must not drift apart: the 2D
registration foreground (``registration_2d._clean_foreground_mask``) and the standalone
tissue-masking tool (``tissue_masking.core.masking.compute_tissue_mask``). They live in
different packages, so the alternative to sharing is duplicating - and duplicated masking
logic drifting apart is exactly the failure this module exists to prevent.

Both steps are load-bearing and were measured on the benchmark corpus:

* **Percentile-clip before thresholding.** A threshold anchored to the histogram's tail
  tracks bright specks, whose survival depends on the downsample factor. On B0039 STPT
  sections the image maximum ran 5,452 -> 16,564 -> 24,767 across an 8x resolution range,
  dragging a Triangle threshold from 53 to 421 - above most of the tissue - and collapsing
  the mask to 0.0003 of the frame. Clipping at the 99.5th percentile removes the dependence.
* **Multi-Otsu rather than Triangle, at four classes.** Normalization alone leaves Triangle
  free to sit *on* a dominant background spike and claim ~85% of the frame. Multi-Otsu
  optimises class variance over the whole histogram and has no degenerate case there. Four
  classes rather than three: three under-segments Ctl1 DAPI badly (foreground 0.2559 against
  0.7565). Measured over 24 STPT configurations - raw Triangle 14 degenerate, normalized
  Triangle 2, normalized Multi-Otsu-4 zero.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LOW_PERCENTILE = 1.0
DEFAULT_HIGH_PERCENTILE = 99.5
DEFAULT_CLASSES = 4

# A tissue threshold below this percentile rank has fallen INTO the background rather than
# sitting above it. Reusing the rank the entropy-GrabCut method's definite-background seed
# claims outright, so the two rules cannot drift.
#
# Measured on the benchmark corpus: the two runaway configurations (B0039 sections 220 and
# 260, where the threshold drops into the 15% zero-padding spike and the mask claims ~85% of
# the frame) sit at percentile 15.21 and 15.23, while the lowest *good* four-class threshold
# anywhere is 35.46 - a 20-point margin with no false positives.
BACKGROUND_PERCENTILE = 20.0


def percentile_normalize(values, low=DEFAULT_LOW_PERCENTILE, high=DEFAULT_HIGH_PERCENTILE):
    """Percentile-clip to [0, 1]. Non-finite input is treated as the low end.

    Returns all-zeros for a degenerate (flat) image rather than raising, so callers can
    treat "nothing to threshold" as an empty mask instead of an error.
    """
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((np.nan_to_num(array, nan=lo) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _percentile_rank(score, value):
    """Percentage of pixels at or below ``value``."""
    array = np.asarray(score)
    return float(np.count_nonzero(array <= value) * 100.0 / array.size)


def multiotsu_low_threshold(
    score, classes=DEFAULT_CLASSES, background_guard=BACKGROUND_PERCENTILE
):
    """Lowest Multi-Otsu threshold, stepping the class count down where it must.

    Two independent reasons to step down, and both matter:

    1. **Too few discretized levels.** Multi-Otsu needs at least as many as classes, and a
       near-binary image has fewer. Stepping down beats raising: a caller that gets an
       exception has no threshold at all and produces an empty mask, which is worse than one
       class fewer. Only that specific failure is degraded past - any other ValueError is a
       real bug and must surface.
    2. **The threshold fell into the background.** With enough classes, Multi-Otsu will spend
       one on a dominant background spike and put its lowest boundary *just above* it, so
       everything non-background becomes foreground. On a B0039 STPT section that takes the
       mask from 0.15 to 0.85 of the frame; raising the class count from 3 to 4 in the
       tissue-masking tool reproduced it at 0.9959 before this guard existed. A threshold
       below ``background_guard`` percent of the pixels is proposing as tissue the very
       pixels a background rule would claim, so the class count steps down until it clears.

    Pass ``background_guard=None`` to disable the second check.

    Returns ``(threshold, classes_used)``; ``classes_used`` is 0 when no class count worked
    and the result came from plain Otsu. A degraded count materially changes the threshold,
    so it is reported rather than silently absorbed.
    """
    from skimage.filters import threshold_multiotsu, threshold_otsu

    requested = int(classes)
    if requested < 2:
        raise ValueError(f"classes must be at least 2; got {classes}")
    fallback = None
    for candidate in range(requested, 1, -1):
        try:
            value = float(threshold_multiotsu(score, classes=candidate)[0])
        except ValueError as exc:
            if "different values" not in str(exc):
                raise
            continue
        if fallback is None:
            fallback = (value, candidate)
        if background_guard is None or _percentile_rank(score, value) >= background_guard:
            return value, candidate
    if fallback is not None:
        # Every count landed in the background. Keep the highest-class result rather than
        # inventing one - the caller sees a real threshold and the mask stays inspectable.
        return fallback
    return float(threshold_otsu(score)), 0
