"""Full-section Jerman vessel contours for both channels, as JPG.

    python scripts/segmentation_contours.py            # pilot subset
    python scripts/segmentation_contours.py --full      # all sections

For each section, one JPG with two panels so the vessel-tracing quality can be
judged on both channels at full resolution:

    CD31   CD31 + Jerman-CD31 contour   (magenta = the ground-truth channel)
    virus  virus + Jerman-virus contour (green)

The Jerman cuts are the tuned per-channel ones from dice_between_channels
(CD31_LOW/HIGH, VIRUS_LOW/HIGH). The virus panel shows plainly what the whole
project keeps finding: Jerman does not cleanly trace vessels on the virus channel,
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
from visualize_sections import _clip                                            # noqa: E402

CD31_COLOUR = "#ff36ff"    # CD31 is the ground truth -> magenta
VIRUS_COLOUR = "#39ff14"   # virus reporter -> green
PANEL_INCHES = 7.5
DPI = 170


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

        panels = [
            ("CD31 (ground truth)", cd31,
             _segment(cd31, tissue, CD31_LOW, CD31_HIGH), CD31_COLOUR,
             f"{CD31_LOW}/{CD31_HIGH}"),
            ("virus", virus,
             _segment(virus, tissue, VIRUS_LOW, VIRUS_HIGH), VIRUS_COLOUR,
             f"{VIRUS_LOW}/{VIRUS_HIGH}"),
        ]
        height, width = cd31.shape
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_INCHES,
                                                PANEL_INCHES * height / width))
        for (name, channel, mask, colour, tag), ax in zip(panels, axes):
            _draw(ax, _clip(channel, tissue), mask, colour,
                  f"{name} - Jerman {tag}   (area {mask.mean():.3f})")

        label = f"{figure} {mouse} {REGION_NAME[region]}" + (f" s{slice_id}" if slice_id else "")
        fig.suptitle(f"{label}  ({reporter}) - Jerman vessel contours: "
                     "CD31 ground truth (magenta), virus (green)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        jpg = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.jpg")
        fig.savefig(jpg, dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"[{index:2d}/{len(paths)}] {jpg.name}")

    print(f"\nwrote Jerman contour JPGs to {out}")


if __name__ == "__main__":
    main()
