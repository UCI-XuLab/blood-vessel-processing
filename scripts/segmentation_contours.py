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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import load_sections, section_paths                    # noqa: E402
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
    paths = section_paths(full)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "segmentation_contours"
    out.mkdir(parents=True, exist_ok=True)

    for s in load_sections(paths):
        panels = [
            ("CD31 (ground truth)", s.cd31,
             _segment(s.cd31, s.tissue, CD31_LOW, CD31_HIGH), CD31_COLOUR,
             f"{CD31_LOW}/{CD31_HIGH}"),
            ("virus", s.virus,
             _segment(s.virus, s.tissue, VIRUS_LOW, VIRUS_HIGH), VIRUS_COLOUR,
             f"{VIRUS_LOW}/{VIRUS_HIGH}"),
        ]
        height, width = s.cd31.shape
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_INCHES,
                                                PANEL_INCHES * height / width))
        for (name, channel, mask, colour, tag), ax in zip(panels, axes):
            _draw(ax, _clip(channel, s.tissue), mask, colour,
                  f"{name} - Jerman {tag}   (area {mask.mean():.3f})")

        fig.suptitle(f"{s.label}  ({s.reporter}) - Jerman vessel contours: "
                     "CD31 ground truth (magenta), virus (green)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        jpg = out / f"{s.stem}.jpg"
        fig.savefig(jpg, dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"{s.counter} {jpg.name}")

    print(f"\nwrote Jerman contour JPGs to {out}")


if __name__ == "__main__":
    main()
