"""Vascular specificity of the enhancer AAV along the spinal cord axis.

Data: Z:\\Lab\\Eric V\\BEC Spinal Cords\\composites_EV  (read-only; never written to)
Two-channel composites of spinal cord cross-sections, per the dataset README:

    channel 1 = green   = virus reporter (SYFP2 or tdTomato)
    channel 2 = magenta = CD31 endothelial staining, the ground truth

Naming: Fig N_Mouse_Region_reporter_CD31-mag_sliceK.tif, region in {C, T, L, TL}.

Design note, and the main methodological choice here
----------------------------------------------------
The obvious approach — segment vessels in both channels and compare the masks —
is wrong for this question. Running a vesselness filter on the virus channel
pre-filters it to tubular shapes, which silently discards the off-target signal
(transduced neurons in grey matter) that should count *against* specificity. It
would measure "of the virus signal that looks like a vessel, how much is on a
vessel", which is close to a tautology and flatters the vector.

So vessels are defined from CD31 alone, and the virus channel is measured inside
and outside that mask without any shape assumption:

    enrichment  mean virus intensity in vessels / in non-vessel tissue.
                Threshold-free, so it does not depend on a tuning choice. This is
                the primary measure.
    coverage    fraction of CD31 vessel area that is virus-positive — how much of
                the vasculature the vector reaches.
    off_target  fraction of virus-positive area lying outside vessels — leak into
                parenchyma. Coverage and off_target move independently, and a
                vector can be good at one and bad at the other.

The virus threshold is set per image from the non-vessel tissue itself
(median + k*MAD), because the README notes brightness was adjusted per image; a
fixed absolute cut would confound acquisition settings with biology.

Artefact handling, all from the dataset README:
  - a 40 um rim is eroded off the tissue mask, dropping the edge staining artefacts
  - the tissue mask excludes the black background and the rotated-composite corners
  - grey-matter background is *not* removed: it is the leak being measured, and
    it is why off_target is reported separately rather than folded into a score
"""

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.filters import threshold_triangle
from skimage.morphology import binary_erosion, disk, remove_small_holes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vessel_utils import metrics                                   # noqa: E402
from vessel_utils.threshold import segment                         # noqa: E402
from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue  # noqa: E402

DATA = Path(r"Z:\Lab\Eric V\BEC Spinal Cords\composites_EV")
UM_PER_PX = 0.650193          # from the calibrated files; identical in all 34 of them
SIGMAS = [1.5, 3.0, 6.0, 12.0]   # um, capillary through venule radius
VESSEL_HIGH = 0.25            # Jerman seeding threshold on CD31
VESSEL_RATIO = 0.5            # hysteresis growing threshold, as a fraction
RIM_UM = 40.0                 # eroded off the tissue edge
MIN_VESSEL_UM2 = 6.0
VIRUS_K = 3.0                 # virus-positive = background median + k * MAD

NAME = re.compile(r"^(Fig \d\w*)_([\w\-]+)_(C|T|L|TL)_(.+?)_CD31-mag(?:_slice(\d+))?\.tif$")
REGION_NAME = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}


def tissue_mask(green, cd31):
    """Section outline with the edge rim removed.

    Thresholding the channel sum keeps tissue that is bright in either channel.
    The largest connected component drops debris; the erosion drops the bright
    edge staining the README warns about, which would otherwise read as signal.
    """
    total = green + cd31
    mask = total > threshold_triangle(total)
    mask = remove_small_holes(mask, area_threshold=200_000)
    labels, count = ndi.label(mask)
    if count:
        mask = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return binary_erosion(mask, disk(int(round(RIM_UM / UM_PER_PX))))


def vessel_mask(cd31, tissue):
    """Vessels from CD31 alone — the ground-truth channel."""
    reference = max_eigenvalue(cd31, SIGMAS, (UM_PER_PX, UM_PER_PX))
    response = jerman_vesselness(cd31, SIGMAS, (UM_PER_PX, UM_PER_PX),
                                 reference_lambda=reference)
    return segment(response, low=VESSEL_HIGH * VESSEL_RATIO, high=VESSEL_HIGH,
                   roi=tissue, min_size=int(round(MIN_VESSEL_UM2 / UM_PER_PX ** 2)),
                   area_threshold=0, closing_radius=1), reference


def analyse(path):
    stack = tifffile.imread(path)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"expected a 2-channel composite, got shape {stack.shape}")
    green = stack[0].astype(np.float32)
    cd31 = stack[1].astype(np.float32)

    tissue = tissue_mask(green, cd31)
    vessels, reference = vessel_mask(cd31, tissue)
    parenchyma = tissue & ~vessels

    if vessels.sum() == 0 or parenchyma.sum() == 0:
        raise ValueError("empty vessel or parenchyma mask")

    in_vessel = green[vessels]
    outside = green[parenchyma]

    # Threshold-free: how much brighter is the virus on vessels than off them.
    enrichment = float(in_vessel.mean() / outside.mean())

    # Virus-positive, calibrated against this image's own parenchyma.
    background = float(np.median(outside))
    mad = float(1.4826 * np.median(np.abs(outside - background)))
    cut = background + VIRUS_K * mad
    virus_positive = (green > cut) & tissue

    coverage = float((virus_positive & vessels).sum() / vessels.sum())
    off_target = float((virus_positive & ~vessels).sum() / max(virus_positive.sum(), 1))

    px_um2 = UM_PER_PX ** 2
    return {
        "tissue_mm2": tissue.sum() * px_um2 / 1e6,
        "vessel_area_fraction": float(vessels.sum() / tissue.sum()),
        "enrichment": enrichment,
        "coverage": coverage,
        "off_target": off_target,
        "dice_virus_vs_cd31": metrics.dice(virus_positive & tissue, vessels),
        "cl_dice_virus_vs_cd31": metrics.cl_dice(virus_positive & tissue, vessels),
        "virus_area_fraction": float(virus_positive.sum() / tissue.sum()),
        "virus_cut": cut,
        "parenchyma_median": background,
        "reference_lambda": reference,
    }


def main():
    paths = sorted(p for p in DATA.glob("Fig*.tif")
                   if NAME.match(p.name) and NAME.match(p.name).group(1).startswith(
                       ("Fig 1", "Fig 2")))
    print(f"{len(paths)} curated images (Fig 1 and Fig 2)\n")

    rows = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        try:
            result = analyse(path)
        except Exception as error:                       # noqa: BLE001
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue
        row = {"figure": figure, "mouse": mouse, "region": region,
               "region_name": REGION_NAME[region], "reporter": reporter,
               "slice": slice_id or "", **result}
        rows.append(row)
        print(f"[{index:2d}/{len(paths)}] {mouse:6s} {region:2s} "
              f"enrich {result['enrichment']:5.2f}  cover {result['coverage']:5.3f}  "
              f"offtgt {result['off_target']:5.3f}  vessel_af {result['vessel_area_fraction']:.4f}")

    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = out / "spinal_cord_specificity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    print("\n=== by region (mean +/- sd across sections) ===")
    print(f"{'region':14s} {'n':>3s} {'enrichment':>18s} {'coverage':>16s} {'off_target':>16s}")
    for code in ("C", "T", "L", "TL"):
        group = [r for r in rows if r["region"] == code]
        if not group:
            continue
        def stat(key):
            values = np.array([r[key] for r in group])
            return f"{values.mean():.3f} +/- {values.std(ddof=1) if len(values) > 1 else 0:.3f}"
        print(f"{REGION_NAME[code]:14s} {len(group):3d} {stat('enrichment'):>18s} "
              f"{stat('coverage'):>16s} {stat('off_target'):>16s}")

    print("\n=== by mouse and region (enrichment) ===")
    mice = sorted({r["mouse"] for r in rows})
    print(f"{'mouse':8s} " + "".join(f"{REGION_NAME[c]:>16s}" for c in ("C", "T", "L")))
    for mouse in mice:
        cells = []
        for code in ("C", "T", "L"):
            group = [r["enrichment"] for r in rows
                     if r["mouse"] == mouse and r["region"] == code]
            cells.append(f"{np.mean(group):.2f} (n={len(group)})" if group else "-")
        print(f"{mouse:8s} " + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
