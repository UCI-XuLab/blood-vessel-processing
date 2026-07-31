"""Is the regional pattern a finding, or a consequence of where the mask was set?

    python scripts/sweep_spinal_cord.py

Why this exists rather than another point estimate
--------------------------------------------------
Tuning on this data moved the segmented vessel area fraction across 0.004, 0.085
and 0.35 depending on choices — the normalisation, the reference percentile, the
seeding threshold — that are individually defensible and jointly
under-determined. At that point another single number is not evidence, because
the number can be moved to taste.

So the operating point is swept, and the question is not "what is the enrichment"
but "does the ordering between regions hold across the whole plausible range". An
ordering stable across an order of magnitude of vessel area fraction is a
property of the data. One that appears at one setting is a property of the setting.

What is swept, and why it is not the threshold
----------------------------------------------
In 2D, Jerman sets lambda_3 := lambda_2, so its saturation condition
`lambda_2 >= lambda_rho/2` reduces to `lambda_2 >= lambda_2/2` and holds
everywhere lambda_2 is positive. The response is therefore near-binary and the
threshold is nearly inert: on these sections moving it from 0.10 to 0.90 changed
the area fraction from 0.534 to 0.493. The operating point in 2D is
`reference_lambda`, which sets the switch at `tau * reference / 2`.

That equivalence also makes the sweep cheap. The eigenvalues are computed once
per section, and each reference is then a comparison against a scalar rather than
a re-run of the filter — which is what lets this cover a 100x range.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vessel_utils.threshold import clean_mask                     # noqa: E402
from vessel_utils.vesselness import hessian_eigenvalues           # noqa: E402

from analyse_spinal_cord import (NAME, SIGMAS, UM_PER_PX, curated_paths,  # noqa: E402
                                 normalise_for_segmentation, tissue_mask)

SPACING = (UM_PER_PX, UM_PER_PX)
REFERENCES = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
TAU = 0.75
MIN_VESSEL_PX = int(round(6.0 / UM_PER_PX ** 2))
REGIONS = ("C", "T", "L")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar"}
PLAUSIBLE = (0.01, 0.10)      # CNS vessel area fraction in section


def prepare(path):
    """Per-section work that does not depend on the operating point."""
    stack = tifffile.imread(path)
    green = stack[0].astype(np.float32)
    cd31 = stack[1].astype(np.float32)
    tissue = tissue_mask(green, cd31)
    normalised = normalise_for_segmentation(cd31, tissue)
    # Largest cross-sectional eigenvalue over scales, sign-flipped for bright
    # vessels. This is the quantity the 2D response thresholds internally.
    lambda_2 = np.max([-hessian_eigenvalues(normalised, s, SPACING)[..., 1]
                       for s in SIGMAS], axis=0)
    return green, tissue, lambda_2


def main():
    paths = [p for p in curated_paths(pilot_mice=2, slices_per_region=3)
             if NAME.match(p.name).group(4).startswith("SYFP2")]
    print(f"{len(paths)} SYFP2 sections, 2 mice x 3 regions x 3 slices\n")

    prepared = {}
    for index, path in enumerate(paths, 1):
        _, mouse, region, _, slice_id = NAME.match(path.name).groups()
        prepared[(mouse, region, slice_id)] = prepare(path)
        print(f"  [{index:2d}/{len(paths)}] {mouse} {region} slice {slice_id}")

    print(f"\n{'reference':>10}{'vessel_af':>11}{'':3}"
          f"{'cerv':>7}{'thor':>7}{'lumb':>7}   ordering (top region agreement)")
    rows = []
    for reference in REFERENCES:
        per_mouse = defaultdict(lambda: defaultdict(list))
        area_fractions = []
        for (mouse, region, _), (green, tissue, lambda_2) in prepared.items():
            mask = clean_mask((lambda_2 >= TAU * reference / 2) & tissue,
                              min_size=MIN_VESSEL_PX, area_threshold=0,
                              closing_radius=1)
            parenchyma = tissue & ~mask
            if mask.sum() == 0 or parenchyma.sum() == 0:
                continue
            area_fractions.append(mask.sum() / tissue.sum())
            per_mouse[mouse][region].append(
                float(green[mask].mean() / green[parenchyma].mean()))

        mice = sorted(per_mouse)
        if not mice:
            print(f"{reference:10.1f}   (mask empty at this reference)")
            continue
        means = {r: float(np.mean([np.mean(per_mouse[m][r]) for m in mice
                                   if per_mouse[m][r]]))
                 if any(per_mouse[m][r] for m in mice) else np.nan
                 for r in REGIONS}
        order = sorted((r for r in REGIONS if not np.isnan(means[r])),
                       key=lambda r: -means[r])
        top = order[0]
        agree = sum(1 for m in mice
                    if per_mouse[m][top] and np.mean(per_mouse[m][top]) ==
                    max(np.mean(per_mouse[m][r]) for r in REGIONS if per_mouse[m][r]))
        area = float(np.mean(area_fractions))
        rows.append((reference, area, means, tuple(order), agree, len(mice)))
        print(f"{reference:10.1f}{area:11.4f}{'':3}"
              + "".join(f"{means[r]:7.3f}" for r in REGIONS)
              + f"   {' > '.join(LONG[r][:4] for r in order)}  ({agree}/{len(mice)})")

    print("\n" + "=" * 78)
    plausible = [r for r in rows if PLAUSIBLE[0] <= r[1] <= PLAUSIBLE[1]]
    print(f"references giving a plausible vessel area fraction "
          f"({PLAUSIBLE[0]:.0%}-{PLAUSIBLE[1]:.0%}): "
          f"{[f'{r[0]:.1f}' for r in plausible] or 'NONE'}")

    if plausible:
        orderings = {r[3] for r in plausible}
        print(f"orderings within that range: {len(orderings)}")
        for ordering in orderings:
            at = [f"{r[0]:.1f}" for r in plausible if r[3] == ordering]
            print(f"  {' > '.join(LONG[r][:4] for r in ordering):22s} at reference {', '.join(at)}")
        if len(orderings) == 1:
            print("\nThe ordering is the same everywhere the segmentation is")
            print("biologically plausible. That is a property of the data, not the")
            print("operating point.")
        else:
            print("\nThe ordering changes within the plausible range, so it cannot be")
            print("claimed without independently justifying one operating point.")

    everywhere = {r[3] for r in rows}
    print(f"\nacross the full {REFERENCES[0]:.0f}-{REFERENCES[-1]:.0f} reference range: "
          f"{len(everywhere)} distinct ordering(s)")
    print("\nNote: this tests robustness to the operating point, not correctness.")
    print("Every setting here could be consistently wrong. Only ground truth -")
    print("stereological point sampling on a few sections - can settle that.")


if __name__ == "__main__":
    main()
