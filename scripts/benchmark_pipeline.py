"""Reproduce the accuracy numbers quoted for the active pipeline.

    python scripts/benchmark_pipeline.py

Every figure in the commit messages and the README comes from this script, so
they can be re-derived rather than taken on trust. It is deterministic: all
randomness is seeded, so a rerun on the same code gives the same numbers, and a
change to the numbers means a change to the pipeline.

What it measures, in order:
  1. accuracy against phantom ground truth, with per-calibre recall
  2. hysteresis versus a single global cut, the archive's approach
  3. the operating point: how much the hysteresis ratio matters, and whether
     the ratio that maximises Dice also measures vessel volume correctly
  4. attribution: which acquisition degradation actually costs recall

Read every number here as performance on the phantom. It is evidence about the
pipeline, not proof about tissue, and only as good as the degradation model in
`vessel_utils.synth`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vessel_utils import benchmark                              # noqa: E402
from vessel_utils.synth import phantom                          # noqa: E402
from vessel_utils.threshold import segment                      # noqa: E402
from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue  # noqa: E402

SIGMAS = [1.5, 3.0, 6.0]           # micrometres, bracketing the phantom's radii
SPACING = (3.0, 0.75, 0.75)        # z, y, x micrometres
SHAPE = (48, 96, 96)
N_PHANTOMS = 5
CALIBRATION_SEED = 99
HIGH = 0.99                        # seeding threshold
RATIO = 0.5                        # growing threshold as a fraction of seeding
CLEANUP = dict(min_size=8, area_threshold=50, closing_radius=1)


def reference_lambda():
    """Pin the tau regularisation once, as the pipeline is meant to be used."""
    volume, _mask, _segments = phantom(shape=SHAPE, spacing=SPACING,
                                       seed=CALIBRATION_SEED)
    return max_eigenvalue(volume, SIGMAS, SPACING)


def make_segmenter(reference, high=HIGH, ratio=RATIO):
    def segmenter(volume, spacing):
        response = jerman_vesselness(volume, SIGMAS, spacing,
                                     reference_lambda=reference)
        return segment(response, low=high * ratio, high=high, **CLEANUP)
    return segmenter


def responses(reference, n=4):
    """Vesselness for several phantoms, computed once and reused per threshold."""
    out = []
    for seed in range(n):
        volume, truth, _ = phantom(shape=SHAPE, spacing=SPACING, seed=seed)
        out.append((jerman_vesselness(volume, SIGMAS, SPACING,
                                      reference_lambda=reference), truth))
    return out


def main():
    reference = reference_lambda()
    print(f"reference_lambda (pinned once, seed {CALIBRATION_SEED}): {reference:.1f}\n")

    print("1. accuracy against phantom ground truth")
    rows = benchmark.run_benchmark(
        make_segmenter(reference), n_phantoms=N_PHANTOMS, shape=SHAPE,
        spacing=SPACING, seed=0, calibre_edges=[0.0, 2.0, 3.5, 20.0])
    summary = benchmark.summarise(rows)
    for key in ("dice", "cl_dice", "precision", "recall"):
        print(f"   {key:10s} {summary[f'{key}_mean']:.3f}"
              + (f"  (95% CI {summary[f'{key}_ci_low']:.3f}-{summary[f'{key}_ci_high']:.3f})"
                 if f"{key}_ci_low" in summary else ""))
    bins = {}
    for row in rows:
        for entry in row["by_calibre"]:
            bins.setdefault((entry["radius_low"], entry["radius_high"]), []).append(
                entry["recall"])
    print("   recall by true vessel radius (um):")
    for (low, high), values in sorted(bins.items()):
        values = [v for v in values if np.isfinite(v)]
        if values:
            print(f"     {low:4.1f}-{high:4.1f}: {np.mean(values):.3f}")

    print("\n2. hysteresis versus a single global cut")
    precomputed = responses(reference)
    for label, ratio in (("single cut", 1.0), ("hysteresis", RATIO)):
        scored = [benchmark.score_segmentation(
            segment(response, low=HIGH * ratio, high=HIGH, **CLEANUP), truth)
            for response, truth in precomputed]
        stats = benchmark.summarise(scored)
        print(f"   {label:12s} dice {stats['dice_mean']:.3f}   "
              f"recall {stats['recall_mean']:.3f}   precision {stats['precision_mean']:.3f}")

    print("\n3. the operating-point trade")
    print(f"   {'ratio':>6} {'dice':>7} {'prec':>7} {'recall':>7} {'predicted/true area':>20}")
    for ratio in (0.5, 0.7, 0.9, 1.0):
        scored = [benchmark.score_segmentation(
            segment(response, low=HIGH * ratio, high=HIGH, **CLEANUP), truth)
            for response, truth in precomputed]
        stats = benchmark.summarise(scored)
        area = np.mean([s["area_fraction_predicted"] / s["area_fraction_truth"]
                        for s in scored])
        print(f"   {ratio:6.1f} {stats['dice_mean']:7.3f} {stats['precision_mean']:7.3f} "
              f"{stats['recall_mean']:7.3f} {area:20.2f}")
    print("   tune the ratio: it moves dice further than any other single knob.")
    print("   check the area ratio alongside dice - only the area ratio tells you")
    print("   whether absolute vessel density is right, and dice can be high while")
    print("   the mask is systematically too fat or too thin.")

    print("\n4. attribution: which degradation costs recall")
    segmenter = make_segmenter(reference)
    for parameter, values in (("gain", [0.05, 0.5, 5.0]),
                              ("psf_sigma", [(1.0, 0.4, 0.4), (3.0, 0.6, 0.6),
                                             (6.0, 1.2, 1.2)])):
        print(f"   {parameter}:")
        for row in benchmark.sweep_condition(segmenter, parameter, values,
                                             n_phantoms=3, shape=SHAPE,
                                             spacing=SPACING):
            print(f"     {str(row[parameter]):22s} dice {row['dice_mean']:.3f}   "
                  f"recall {row['recall_mean']:.3f}")


if __name__ == "__main__":
    main()
