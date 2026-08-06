"""Agreement between the virus and CD31 vessel masks, per section.

    python scripts/dice_between_channels.py            # pilot: 2 mice/reporter
    python scripts/dice_between_channels.py --full      # every clean mouse

This is the paper-1 style comparison: segment vessels in BOTH channels and score
how much they overlap. Each channel is segmented with a graded Jerman filter
(self-normalise, Jerman at REFERENCE, wide-gap hysteresis, clean-up), but the two
channels get DIFFERENT thresholds on purpose:

  - CD31 is stained vasculature, so a vessel-like cut (CD31_LOW/CD31_HIGH) traces
    the network directly.
  - the virus channel is dominated by non-vascular neuronal expression that is
    structurally vessel-like and cannot be told from vessels by shape. A STRICTER
    cut (VIRUS_LOW/VIRUS_HIGH) keeps only the brightest vesselness responses —
    which are the genuinely vessel-associated virus — and discards the dimmer
    neuronal ridge signal. On a test section this raised specificity against CD31
    from ~0.5 (at CD31's own cut) to ~0.6, without inventing vessels.

Segmenting each channel as well as it can be, then comparing, replaces the older
"same filter on both channels so it cancels" symmetry. That only looked clean
because both masks over-segmented to ~40% of the tissue and overlapped trivially,
inflating Dice; the tuned masks are vessel-like and the agreement is real. Even so
the virus mask cannot fully match CD31 — see enrichment_by_cd31_percentile.py for
the measure that never segments the virus.

Read the three numbers together; the directional pair is more interpretable than
Dice alone:

    dice        symmetric overlap of the two masks (the headline "agreement").
    specificity |virus & cd31| / |virus|  — fraction of the virus vessel mask on
                a CD31 vessel. This is the specificity claim.
    coverage    |virus & cd31| / |cd31| — fraction of the vasculature the virus
                vessel mask reaches.

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

from analyse_spinal_cord import (NAME, SIGMAS, UM_PER_PX, curated_paths,          # noqa: E402
                                 normalise_for_segmentation, tissue_mask)
from vessel_utils import metrics                                                 # noqa: E402
from vessel_utils.threshold import segment                                       # noqa: E402
from vessel_utils.vesselness import jerman_vesselness                            # noqa: E402

# TL included so thoracolumbar sections (e.g. M87) appear in the summary rather
# than being scored into the CSV but silently dropped from the printed tables.
REGIONS = ("C", "T", "L", "TL")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}

# Per-channel Jerman operating points (see module docstring). REFERENCE is graded,
# not saturated, so the hysteresis thresholds actually bite. CD31 gets a vessel-
# like cut; the virus gets a stricter one to keep only vessel-associated signal.
REFERENCE = 2.5
CD31_LOW, CD31_HIGH = 0.04, 0.13
VIRUS_LOW, VIRUS_HIGH = 0.08, 0.22
MIN_VESSEL_PX = int(round(6.0 / UM_PER_PX ** 2))


def channels(path):
    stack = tifffile.imread(path)
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError(f"expected a 2-channel composite, got shape {stack.shape}")
    return stack[0].astype(np.float32), stack[1].astype(np.float32)   # virus, cd31


def _segment(channel, tissue, low, high):
    """Graded-Jerman vessel mask at the given hysteresis thresholds."""
    response = jerman_vesselness(normalise_for_segmentation(channel, tissue),
                                 SIGMAS, (UM_PER_PX, UM_PER_PX), reference_lambda=REFERENCE)
    return segment(response, low=low, high=high, roi=tissue,
                   min_size=MIN_VESSEL_PX, area_threshold=0, closing_radius=1)


def channel_masks(path):
    """Per-channel tuned vessel masks for one section (also used for visualisation)."""
    virus, cd31 = channels(path)
    tissue = tissue_mask(virus, cd31)
    mask_virus = _segment(virus, tissue, VIRUS_LOW, VIRUS_HIGH)
    mask_cd31 = _segment(cd31, tissue, CD31_LOW, CD31_HIGH)
    return virus, cd31, tissue, mask_virus, mask_cd31


def score_section(path):
    _, _, _, mask_virus, mask_cd31 = channel_masks(path)
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
    print(f"{len(paths)} sections ({label})")
    print(f"CD31 thr {CD31_LOW}/{CD31_HIGH}, virus thr {VIRUS_LOW}/{VIRUS_HIGH}, "
          f"reference {REFERENCE}\n")

    rows = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        reporter = "SYFP2" if "SYFP2" in reporter else ("tdT" if "tdT" in reporter else reporter)
        try:
            result = score_section(path)
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
