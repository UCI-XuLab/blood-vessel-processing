"""Is the regional pattern a finding, or a consequence of where the mask was set?

    python scripts/sweep_spinal_cord.py

Why this exists rather than another point estimate
--------------------------------------------------
Tuning on this data moved the segmented vessel area fraction across 0.004, 0.085
and 0.35 depending on choices — the normalisation, the reference, the threshold —
that are individually defensible and jointly under-determined. At that point
another single number is not evidence, because the number can be moved to taste.

So the operating point is swept, and the question is not "what is the enrichment"
but "does the ordering between regions hold across the whole plausible range". An
ordering stable across an order of magnitude of vessel area fraction is a
property of the data; one that appears at a single setting is a property of the
setting.

Which knob to sweep, and why it is this one
-------------------------------------------
In 2D, Jerman saturates everything above `tau * reference_lambda / 2` to exactly
1, so the reference and the threshold are coupled and neither can be tuned alone.
How much authority each has depends on the data. On a bimodal synthetic almost
everything saturates and the threshold is inert; on these sections the graded
band held 21-37% of the tissue, so the threshold does most of the work — provided
the reference is not set so low that the whole section saturates.

The reference is therefore fixed at a value where the response is graded rather
than saturated, and the threshold is swept against it. That also makes the sweep
cheap: the response is computed once per section and every threshold reuses it.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vessel_utils.threshold import segment                       # noqa: E402
from vessel_utils.vesselness import jerman_vesselness            # noqa: E402

from analyse_spinal_cord import (NAME, SIGMAS, UM_PER_PX, curated_paths,  # noqa: E402
                                 normalise_for_segmentation, tissue_mask)

SPACING = (UM_PER_PX, UM_PER_PX)
REFERENCE = 2.5           # graded, not saturated, on these sections
THRESHOLDS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
MIN_VESSEL_PX = int(round(6.0 / UM_PER_PX ** 2))
REGIONS = ("C", "T", "L")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar"}
PLAUSIBLE = (0.01, 0.10)  # CNS vessel area fraction in a section


def prepare(path):
    """Per-section work that does not depend on the threshold."""
    stack = tifffile.imread(path)
    green = stack[0].astype(np.float32)
    cd31 = stack[1].astype(np.float32)
    tissue = tissue_mask(green, cd31)
    response = jerman_vesselness(normalise_for_segmentation(cd31, tissue),
                                 SIGMAS, SPACING, reference_lambda=REFERENCE)
    return green, tissue, response


def main():
    paths = [p for p in curated_paths(pilot_mice=2, slices_per_region=3)
             if NAME.match(p.name).group(4).startswith("SYFP2")]
    print(f"{len(paths)} SYFP2 sections, 2 mice x 3 regions x 3 slices")
    print(f"reference_lambda fixed at {REFERENCE}; sweeping the threshold\n")

    prepared = {}
    for index, path in enumerate(paths, 1):
        _, mouse, region, _, slice_id = NAME.match(path.name).groups()
        prepared[(mouse, region, slice_id)] = prepare(path)
        print(f"  [{index:2d}/{len(paths)}] {mouse} {region} slice {slice_id}")

    graded = np.mean([np.mean((r[2][r[1]] > 0.01) & (r[2][r[1]] < 0.99))
                      for r in prepared.values()])
    print(f"\ngraded fraction of the response inside tissue: {graded:.3f} "
          f"({'threshold has authority' if graded > 0.05 else 'SATURATED - reference too low'})")

    print(f"\n{'thresh':>7}{'vessel_af':>11}{'':3}{'cerv':>7}{'thor':>7}{'lumb':>7}"
          f"   ordering                 mice agreeing")
    rows = []
    for high in THRESHOLDS:
        per_mouse = defaultdict(lambda: defaultdict(list))
        area_fractions = []
        for (mouse, region, _), (green, tissue, response) in prepared.items():
            mask = segment(response, low=high * 0.5, high=high, roi=tissue,
                           min_size=MIN_VESSEL_PX, area_threshold=0,
                           closing_radius=1)
            parenchyma = tissue & ~mask
            if mask.sum() == 0 or parenchyma.sum() == 0:
                continue
            area_fractions.append(mask.sum() / tissue.sum())
            per_mouse[mouse][region].append(
                float(green[mask].mean() / green[parenchyma].mean()))

        mice = sorted(per_mouse)
        if not mice or not all(per_mouse[m][r] for m in mice for r in REGIONS):
            print(f"{high:7.2f}   (some region empty at this threshold)")
            continue

        # Mouse means first: three slices are three views of one animal.
        means = {r: float(np.mean([np.mean(per_mouse[m][r]) for m in mice]))
                 for r in REGIONS}
        order = tuple(sorted(REGIONS, key=lambda r: -means[r]))
        agreeing = sum(1 for m in mice
                       if tuple(sorted(REGIONS,
                                       key=lambda r: -np.mean(per_mouse[m][r]))) == order)
        area = float(np.mean(area_fractions))
        rows.append((high, area, means, order, agreeing, len(mice)))
        print(f"{high:7.2f}{area:11.4f}{'':3}"
              + "".join(f"{means[r]:7.3f}" for r in REGIONS)
              + f"   {' > '.join(LONG[r][:4] for r in order):22s} {agreeing}/{len(mice)}")

    print("\n" + "=" * 78)
    plausible = [r for r in rows if PLAUSIBLE[0] <= r[1] <= PLAUSIBLE[1]]
    print(f"thresholds giving a plausible vessel area fraction "
          f"({PLAUSIBLE[0]:.0%}-{PLAUSIBLE[1]:.0%}): "
          f"{[f'{r[0]:.2f}' for r in plausible] or 'NONE'}")

    if plausible:
        orderings = {r[3] for r in plausible}
        for ordering in orderings:
            at = [f"{r[0]:.2f}" for r in plausible if r[3] == ordering]
            agree = [f"{r[4]}/{r[5]}" for r in plausible if r[3] == ordering]
            print(f"  {' > '.join(LONG[r][:4] for r in ordering):22s} "
                  f"at {', '.join(at)}   mice agreeing: {', '.join(agree)}")
        if len(orderings) == 1:
            print("\nSame ordering everywhere the segmentation is plausible: that is a")
            print("property of the data, not of the operating point.")
        else:
            print("\nThe ordering changes within the plausible range. It cannot be")
            print("claimed without independently justifying one operating point.")

    print(f"\nacross the full threshold range: {len({r[3] for r in rows})} distinct ordering(s)")
    print("\nThis tests robustness to the operating point, not correctness. Every")
    print("setting here could be wrong in the same direction; only ground truth -")
    print("stereological point sampling on a few sections - can settle that.")


if __name__ == "__main__":
    main()
