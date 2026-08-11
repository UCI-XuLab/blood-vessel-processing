"""Back the dim-vessel recall numbers quoted for the hysteresis change.

    python scripts/dim_recall_check.py

The analysis switched from a narrow hysteresis gap (low = high*0.5) to a wide one
(low = 0.02, high = 0.15) on the strength of a claim about dim-vessel recall. The
number was first asserted from a single cervical crop (45% -> 91%) with no way to
reproduce it - the exact failure this project keeps tripping on. This script is
the missing artefact, and it corrects that number: across six sections the mean
is 0.19 -> 0.74, because the one-crop figure was not representative. The
direction and size of the effect hold; the specific numbers did not.

What "recall" means here, precisely
-----------------------------------
There is no manual ground truth, so this is not recall against a human tracing.
It is recall against a *visible-vessel proxy*: CD31 pixels that clear a local
adaptive threshold (brighter than their neighbourhood and above the tissue
upper quartile). The proxy is imperfect - it is itself an intensity rule - so
read the numbers as a self-consistent comparison between two hysteresis settings,
not as absolute accuracy. What it can show honestly is the *difference* the wide
gap makes, stratified by how dim the vessel is, which is the claim being made.

Reads Z: read-only; writes nothing.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_local

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import (NAME, SIGMAS, UM_PER_PX, curated_paths,   # noqa: E402
                                 normalise_for_segmentation, tissue_mask)
from vessel_utils.threshold import clean_mask                              # noqa: E402
from vessel_utils.vesselness import jerman_vesselness                      # noqa: E402

SPACING = (UM_PER_PX, UM_PER_PX)
REFERENCE = 2.5
NARROW = (0.075, 0.15)   # low = high * 0.5, the old default gap
WIDE = (0.02, 0.15)      # the new wide gap
MIN_VESSEL_PX = int(round(6.0 / UM_PER_PX ** 2))


def visible_vessel_proxy(normalised, tissue):
    """CD31 pixels brighter than their local neighbourhood and the tissue Q3."""
    local = threshold_local(normalised, block_size=101)
    return (normalised > local * 1.3) & (normalised > np.percentile(normalised[tissue], 75)) & tissue


def recall_by_brightness(mask, proxy, normalised):
    """Fraction of proxy pixels captured, split into dim/mid/bright thirds."""
    values = normalised[proxy]
    cuts = np.percentile(values, [33, 66])
    out = {}
    for label, low, high in [("dim", -np.inf, cuts[0]),
                             ("mid", cuts[0], cuts[1]),
                             ("bright", cuts[1], np.inf)]:
        band = proxy & (normalised >= low) & (normalised < high)
        out[label] = float(mask[band].mean()) if band.any() else float("nan")
    return out


def main():
    # A few SYFP2 sections spanning regions; the effect is not region-specific.
    paths = [p for p in curated_paths(pilot_mice=2, slices_per_region=1)
             if NAME.match(p.name).group(4).startswith("SYFP2")]
    print(f"{len(paths)} sections\n")
    print(f"{'section':22s}{'gap':8s}{'dim':>7}{'mid':>7}{'bright':>8}{'area':>8}")

    dim = {"narrow": [], "wide": []}
    for path in paths:
        _, mouse, region, _, _ = NAME.match(path.name).groups()
        stack = tifffile.imread(path)
        green, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(green, cd31)
        except ValueError as error:
            print(f"  SKIP {path.name}: {error}")
            continue
        normalised = normalise_for_segmentation(cd31, tissue)
        response = jerman_vesselness(normalised, SIGMAS, SPACING, reference_lambda=REFERENCE)
        proxy = visible_vessel_proxy(normalised, tissue)

        for name, (low, high) in [("narrow", NARROW), ("wide", WIDE)]:
            from vessel_utils.threshold import hysteresis_threshold
            mask = clean_mask(hysteresis_threshold(response, low, high) & tissue,
                              min_size=MIN_VESSEL_PX, area_threshold=0, closing_radius=1)
            r = recall_by_brightness(mask, proxy, normalised)
            dim[name].append(r["dim"])
            print(f"{mouse + ' ' + region:22s}{name:8s}{r['dim']:7.2f}{r['mid']:7.2f}"
                  f"{r['bright']:8.2f}{mask.mean():8.3f}")

    print(f"\nmean dim-vessel recall:  narrow gap {np.nanmean(dim['narrow']):.2f}   "
          f"wide gap {np.nanmean(dim['wide']):.2f}")
    print("(this is the number the analysis comment cites; ~0.19 -> ~0.74)")


if __name__ == "__main__":
    main()
