"""Full-section CD31 segmentation contours, Jerman vs top-q%, exported as JPG.

    python scripts/cd31_method_contours.py            # pilot subset
    python scripts/cd31_method_contours.py --full      # all sections

For each section, one JPG with two full-resolution panels of the CD31 channel,
each with a segmentation contour drawn on top so the vessel-tracing quality of the
two methods can be judged side by side:

  left   Jerman vesselness mask (the graded Jerman cut used in
         dice_between_channels, CD31_LOW/CD31_HIGH) — contour in cyan
  right  top-q% CD31 intensity (the threshold-free percentile definition used by
         the enrichment measure, VIS_Q) — contour in amber

JPG (not PNG) keeps the files small enough to browse the whole dataset while
staying detailed enough to zoom in on individual vessels. Reads Z: read-only;
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
from dice_between_channels import CD31_HIGH, CD31_LOW, _segment                 # noqa: E402
from visualize_sections import VIS_Q, _clip, top_q_mask                         # noqa: E402

JERMAN_COLOUR = "#00e5ff"
PERCENTILE_COLOUR = "#ffb300"
PANEL_INCHES = 8.0     # per panel; savefig dpi sets the pixel resolution
DPI = 200              # -> ~1600 px per panel


def _draw(ax, grey, mask, colour, title):
    ax.imshow(grey, cmap="gray")
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=colour, linewidths=0.6)
    ax.set_title(title, fontsize=13)
    ax.axis("off")


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=1)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "cd31_contours"
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

        jerman = _segment(cd31, tissue, CD31_LOW, CD31_HIGH)
        percentile = top_q_mask(cd31, tissue, VIS_Q)
        grey = _clip(cd31, tissue)

        height, width = cd31.shape
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_INCHES,
                                                PANEL_INCHES * height / width))
        _draw(axes[0], grey, jerman, JERMAN_COLOUR,
              f"Jerman  {CD31_LOW}/{CD31_HIGH}   (area {jerman.mean():.3f})")
        _draw(axes[1], grey, percentile, PERCENTILE_COLOUR,
              f"top-{VIS_Q}% CD31   (area {percentile.mean():.3f})")
        label = f"{figure} {mouse} {REGION_NAME[region]}" + (f" s{slice_id}" if slice_id else "")
        fig.suptitle(f"{label}  ({reporter}) — CD31 vessel contours: "
                     "Jerman (cyan) vs top-q% (amber)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        jpg = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.jpg")
        fig.savefig(jpg, dpi=DPI, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        print(f"[{index:2d}/{len(paths)}] {jpg.name}   "
              f"Jerman af {jerman.mean():.3f}  top-{VIS_Q}% af {percentile.mean():.3f}")

    print(f"\nwrote CD31 contour JPGs to {out}")


if __name__ == "__main__":
    main()
