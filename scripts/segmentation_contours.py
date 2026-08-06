"""Full-section segmentation contours for BOTH channels, Jerman vs top-q%, as JPG.

    python scripts/segmentation_contours.py            # pilot subset
    python scripts/segmentation_contours.py --full      # all sections

For each section, one JPG with a 2x2 grid so the vessel-tracing quality of the two
segmentation methods can be judged on both channels at full resolution:

               Jerman (cyan)              top-q% percentile (amber)
    CD31   CD31 + Jerman-CD31 contour    CD31 + top-q%-CD31 contour
    virus  virus + Jerman-virus contour  virus + top-q%-virus contour

The Jerman cuts are the tuned per-channel ones from dice_between_channels
(CD31_LOW/HIGH, VIRUS_LOW/HIGH); the percentile mask is the speck-cleaned top-q%
of that channel's own intensity (VIS_Q). The virus row shows plainly what the
whole project keeps finding: neither method traces vessels on the virus channel,
because much of the bright virus is non-vascular (neuronal). Reads Z: read-only;
writes only under results/ (gitignored).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import NAME, REGION_NAME, curated_paths, tissue_mask   # noqa: E402
from dice_between_channels import (CD31_HIGH, CD31_LOW, VIRUS_HIGH,              # noqa: E402
                                   VIRUS_LOW, _segment)
from visualize_sections import VIS_Q, _clip, top_q_mask                         # noqa: E402

JERMAN_COLOUR = "#00e5ff"
PERCENTILE_COLOUR = "#ffb300"
PANEL_INCHES = 7.5
DPI = 170
PANELS = [("CD31", "Jerman"), ("CD31", "top"), ("virus", "Jerman"), ("virus", "top")]


def _draw(ax, grey, mask, colour, title):
    ax.imshow(grey, cmap="gray")
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=0.6)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=1)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "segmentation_contours"
    out.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        reporter = ("SYFP2" if "SYFP2" in reporter
                    else "tdT" if "tdT" in reporter else reporter)
        stack = tifffile.imread(path)
        if stack.ndim != 3 or stack.shape[0] != 2:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: not 2-channel")
            continue
        virus, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(virus, cd31)
        except ValueError as error:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue

        cell = {
            ("CD31", "Jerman"): (cd31, _segment(cd31, tissue, CD31_LOW, CD31_HIGH),
                                 JERMAN_COLOUR, f"{CD31_LOW}/{CD31_HIGH}"),
            ("CD31", "top"): (cd31, top_q_mask(cd31, tissue, VIS_Q),
                              PERCENTILE_COLOUR, f"top-{VIS_Q}%"),
            ("virus", "Jerman"): (virus, _segment(virus, tissue, VIRUS_LOW, VIRUS_HIGH),
                                  JERMAN_COLOUR, f"{VIRUS_LOW}/{VIRUS_HIGH}"),
            ("virus", "top"): (virus, top_q_mask(virus, tissue, VIS_Q),
                               PERCENTILE_COLOUR, f"top-{VIS_Q}%"),
        }
        height, width = cd31.shape
        fig, axes = plt.subplots(2, 2, figsize=(2 * PANEL_INCHES,
                                                2 * PANEL_INCHES * height / width))
        for (channel_name, method), ax in zip(PANELS, axes.ravel()):
            channel, mask, colour, tag = cell[(channel_name, method)]
            _draw(ax, _clip(channel, tissue), mask, colour,
                  f"{channel_name} {method} {tag}   (area {mask.mean():.3f})")

        label = f"{figure} {mouse} {REGION_NAME[region]}" + (f" s{slice_id}" if slice_id else "")
        fig.suptitle(f"{label}  ({reporter}) — segmentation contours: "
                     "Jerman (cyan) vs top-q% (amber); CD31 (top) / virus (bottom)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        jpg = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.jpg")
        fig.savefig(jpg, dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"[{index:2d}/{len(paths)}] {jpg.name}")

    print(f"\nwrote segmentation contour JPGs to {out}")


if __name__ == "__main__":
    main()
