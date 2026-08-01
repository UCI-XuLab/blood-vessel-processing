"""Agreement between the virus and CD31 vessel masks, per section.

    python scripts/dice_between_channels.py            # pilot: 2 mice/reporter
    python scripts/dice_between_channels.py --full      # every clean mouse

This is the paper-1 style comparison: segment vessels in BOTH channels and score
how much they overlap. Both channels go through the identical pipeline —
`analyse_spinal_cord.vessel_mask` (self-normalise, Jerman, wide-gap hysteresis,
one dataset-wide reference) — so the filter choice is symmetric and largely
cancels; what remains is the biology.

The virus channel is deliberately NOT cleaned of neurons or neuropil. That signal
is the vector's off-target labelling, and for an agreement score it is real: a low
Dice *because* the virus mask spills off the vessels is the non-specificity the
experiment measures. Cleaning it would inflate the score and hide the finding.

Read the three numbers together; the directional pair is more interpretable than
Dice alone:

    dice        symmetric overlap of the two masks (the headline "agreement").
    specificity |virus & vessels| / |virus|  — fraction of virus signal on
                vessels. This is the specificity claim.
    coverage    |virus & vessels| / |vessels| — fraction of the vasculature the
                virus reaches.

Aggregation respects the nesting: three slices are three views of one animal, so
slices are averaged within mouse x region before anything else, and figures are
kept separate (Fig 1 and Fig 3 are different constructs; pooling would confound
reporter/construct with region). Reads Z: read-only.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import (NAME, SIGMAS, UM_PER_PX, calibrate_reference,   # noqa: E402
                                 curated_paths, tissue_mask, vessel_mask)
from vessel_utils import metrics                                                 # noqa: E402

# TL included so thoracolumbar sections (e.g. M87) appear in the summary rather
# than being scored into the CSV but silently dropped from the printed tables.
REGIONS = ("C", "T", "L", "TL")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}


def channels(path):
    stack = tifffile.imread(path)
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError(f"expected a 2-channel composite, got shape {stack.shape}")
    return stack[0].astype(np.float32), stack[1].astype(np.float32)   # virus, cd31


def score_section(path, reference):
    virus, cd31 = channels(path)
    tissue = tissue_mask(virus, cd31)
    mask_cd31 = vessel_mask(cd31, tissue, reference)      # same pipeline, both
    mask_virus = vessel_mask(virus, tissue, reference)    # channels — symmetric
    if mask_cd31.sum() == 0 or mask_virus.sum() == 0:
        raise ValueError("a channel produced an empty vessel mask")
    return {
        "dice": metrics.dice(mask_virus, mask_cd31),
        "jaccard": metrics.jaccard(mask_virus, mask_cd31),
        "specificity": metrics.precision(mask_virus, mask_cd31),   # virus on vessels
        "coverage": metrics.recall(mask_virus, mask_cd31),         # vessels labelled
        "virus_af": float(mask_virus.mean()),
        "cd31_af": float(mask_cd31.mean()),
    }


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=3)
    label = "all clean mice" if full else "pilot: 2 mice/reporter, 3 slices/region"
    print(f"{len(paths)} sections ({label})\n")

    reference = calibrate_reference(paths, n_sections=min(6, len(paths)))
    print()

    rows = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        reporter = "SYFP2" if "SYFP2" in reporter else ("tdT" if "tdT" in reporter else reporter)
        try:
            result = score_section(path, reference)
        except Exception as error:                                   # noqa: BLE001
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue
        rows.append({"figure": figure, "reporter": reporter, "mouse": mouse,
                     "region": region, "slice": slice_id or "", **result})
        print(f"[{index:2d}/{len(paths)}] {figure} {reporter:5s} {mouse:7s} {region:2s}  "
              f"dice {result['dice']:.3f}  specificity {result['specificity']:.3f}  "
              f"coverage {result['coverage']:.3f}")

    if not rows:
        sys.exit("no sections scored")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = out / ("dice_between_channels_full.csv" if full
                      else "dice_between_channels_pilot.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # Mouse means first, grouped by (figure, reporter); figures kept separate.
    by_group = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        by_group[(r["figure"], r["reporter"])][r["mouse"]][r["region"]].append(r)

    for (figure, reporter) in sorted(by_group):
        mice = sorted(by_group[(figure, reporter)])
        regions = [x for x in REGIONS
                   if any(by_group[(figure, reporter)][m][x] for m in mice)]
        if not regions:
            continue
        print("\n" + "=" * 76)
        print(f"{figure}  {reporter}   mice: {', '.join(mice)}")
        print("=" * 76)
        for measure in ("dice", "specificity", "coverage"):
            print(f"\n{measure}")
            print("  " + f"{'mouse':8s}" + "".join(f"{LONG[x]:>12s}" for x in regions))
            table = {}
            for mouse in mice:
                cells = []
                for region in regions:
                    vals = [s[measure] for s in by_group[(figure, reporter)][mouse][region]]
                    table[(mouse, region)] = np.mean(vals) if vals else np.nan
                    cells.append("     -  " if np.isnan(table[(mouse, region)])
                                 else f"{table[(mouse, region)]:12.3f}")
                print(f"  {mouse:8s}" + "".join(cells))
            complete = [m for m in mice
                        if all(not np.isnan(table[(m, x)]) for x in regions)]
            if len(complete) >= 1 and len(regions) >= 2:
                print(f"  {'mean':8s}" + "".join(
                    f"{np.mean([table[(m, x)] for m in complete]):12.3f}" for x in regions))

    print("\n" + "=" * 76)
    print("dice/jaccard are symmetric agreement; specificity and coverage are the")
    print("directional, interpretable pair. The virus mask is uncleaned on purpose")
    print("-- its off-vessel signal is the vector's non-specificity, not an error.")
    print("With few mice, read direction-consistency across animals, not p-values.")


if __name__ == "__main__":
    main()
