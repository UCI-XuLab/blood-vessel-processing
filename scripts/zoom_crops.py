"""Full-resolution zoom crops of the most salient windows in every section.

    python scripts/zoom_crops.py            # pilot subset (fast)
    python scripts/zoom_crops.py --full     # all 38 sections

The whole-section panels (visualize_sections.py) are downsampled 2x, so single
capillaries are not resolved. This finds, per section, the most informative
~500 um window under each of four criteria and renders it at native resolution
so the virus-on-vessel relationship can be inspected by eye. Vessels are the
top-q% of CD31, matching the primary threshold-free measure.

The four criteria (each becomes one crop):
  densest vasculature       most CD31 vessel area — best place to judge whether
                            virus tracks vessels (on-target inspection)
  off-vessel virus (leak)   brightest virus in parenchyma away from vessels —
                            the off-target signal (transduced neurons)
  strongest colocalization  highest local virus-in-vessel / virus-out ratio —
                            the vector at its best
  brightest virus           highest raw virus, agnostic to vessels

Each section produces one figure: a locator (the section with the four crop
boxes drawn) beside four crop strips — virus | CD31 | green/magenta merge with
the vessel outline and a 100 um scale bar. A zoom contact sheet tiles the
densest-vasculature crop of every section.

Reads Z: read-only; writes only under results/ (which is gitignored).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import (NAME, REGION_NAME, UM_PER_PX, curated_paths,   # noqa: E402
                                 tissue_mask)
from visualize_sections import VIS_Q, top_q_mask                                # noqa: E402

CROP_UM = 800.0    # zoomed out from 500 for more surrounding context
CROP_PX = int(round(CROP_UM / UM_PER_PX))
SCALEBAR_UM = 100.0
# (key, display name, box colour) in the order rows are drawn.
CRITERIA = [
    ("densest", "densest vasculature", "#00e5ff"),
    ("leak", "off-vessel virus (leak)", "#ff8c00"),
    ("coloc", "strongest colocalization", "#39ff14"),
    ("bright", "brightest virus", "#ffd400"),
]


def _clip(channel, lo, hi):
    return np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _merge_rgb(virus_clip, cd31_clip):
    """Virus in green, CD31 in magenta; their overlap reads white."""
    rgb = np.zeros((*virus_clip.shape, 3), np.float32)
    rgb[..., 0] = cd31_clip
    rgb[..., 1] = virus_clip
    rgb[..., 2] = cd31_clip
    return rgb


def local_enrichment(green, vessels, tissue):
    parenchyma = tissue & ~vessels
    if vessels.sum() == 0 or parenchyma.sum() == 0:
        return float("nan")
    return float(green[vessels].mean() / green[parenchyma].mean())


def saliency_centers(green, cd31, tissue, vessels):
    """One (row, col) window centre per criterion, or None where undefined.

    Scores are windowed means over a CROP_PX box (uniform_filter is separable, so
    this is O(n) even for a large box). Centres are restricted to windows that lie
    fully inside the image and are mostly tissue, so a crop never runs off the edge
    or sits on background.
    """
    box = lambda x: uniform_filter(x.astype(np.float32), size=CROP_PX, mode="constant")
    eps = 1e-6
    parenchyma = tissue & ~vessels
    tis_f = box(tissue.astype(np.float32))
    ves_f = box(vessels.astype(np.float32))
    par_f = box(parenchyma.astype(np.float32))
    virus_in_tissue = box(green * tissue) / (tis_f + eps)
    virus_in_vessel = box(green * vessels) / (ves_f + eps)
    virus_in_paren = box(green * parenchyma) / (par_f + eps)

    height, width = green.shape
    half = CROP_PX // 2
    valid = np.zeros(green.shape, bool)
    valid[half:height - half, half:width - half] = True
    valid &= tis_f > 0.6
    stable = valid & (ves_f > 0.005) & (par_f > 0.05)

    scores = {
        "densest": np.where(valid, ves_f, -np.inf),
        "leak": np.where(valid & (par_f > 0.05), virus_in_paren, -np.inf),
        "coloc": np.where(stable, virus_in_vessel / (virus_in_paren + eps), -np.inf),
        "bright": np.where(valid, virus_in_tissue, -np.inf),
    }
    centres = {}
    for key, score in scores.items():
        flat = int(np.argmax(score))
        row, col = np.unravel_index(flat, score.shape)
        centres[key] = (int(row), int(col)) if np.isfinite(score[row, col]) else None
    return centres


def _crop_slice(centre):
    half = CROP_PX // 2
    row, col = centre
    return np.s_[row - half:row - half + CROP_PX, col - half:col - half + CROP_PX]


def _add_scalebar(ax):
    length = SCALEBAR_UM / UM_PER_PX
    x0, y0 = CROP_PX * 0.06, CROP_PX * 0.92
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.text(x0, y0 - CROP_PX * 0.03, f"{SCALEBAR_UM:.0f} µm", color="white",
            fontsize=8, va="bottom")


def section_figure(green, cd31, tissue, vessels, centres, title, png_path):
    lo_v, hi_v = np.percentile(green[tissue], [1, 99])
    lo_c, hi_c = np.percentile(cd31[tissue], [1, 99])

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.05, hspace=0.15)

    # Locator: downsampled merge of the whole section with the crop boxes.
    ds = max(1, green.shape[0] // 700)
    ax_loc = fig.add_subplot(gs[:, 0])
    ax_loc.imshow(_merge_rgb(_clip(green, lo_v, hi_v)[::ds, ::ds],
                             _clip(cd31, lo_c, hi_c)[::ds, ::ds]))
    ax_loc.set_title("locator — virus (green) / CD31 (magenta)", fontsize=10)
    ax_loc.set_xticks([]); ax_loc.set_yticks([])

    half = CROP_PX // 2
    for (key, name, colour) in CRITERIA:
        centre = centres.get(key)
        if centre is None:
            continue
        row, col = centre
        rect = mpatches.Rectangle(((col - half) / ds, (row - half) / ds),
                                  CROP_PX / ds, CROP_PX / ds, fill=False,
                                  edgecolor=colour, lw=2)
        ax_loc.add_patch(rect)

    for i, (key, name, colour) in enumerate(CRITERIA):
        centre = centres.get(key)
        axes = [fig.add_subplot(gs[i, j]) for j in (1, 2, 3)]
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        if centre is None:
            axes[0].set_ylabel(f"{name}\n(no valid window)", fontsize=9, color=colour)
            for ax in axes:
                ax.set_facecolor("0.1")
            continue
        sl = _crop_slice(centre)
        gc = _clip(green[sl], lo_v, hi_v)
        cc = _clip(cd31[sl], lo_c, hi_c)
        vc, tc = vessels[sl], tissue[sl]
        enrich = local_enrichment(green[sl], vc, tc)

        axes[0].imshow(gc, cmap="gray")
        axes[1].imshow(cc, cmap="gray")
        axes[2].imshow(_merge_rgb(gc, cc))
        if vc.any():
            axes[2].contour(vc.astype(float), levels=[0.5], colors="#ffea00",
                            linewidths=0.35)
        _add_scalebar(axes[2])
        if i == 0:
            axes[0].set_title("virus", fontsize=10)
            axes[1].set_title("CD31", fontsize=10)
            axes[2].set_title("merge + vessel outline", fontsize=10)
        axes[0].set_ylabel(f"{name}\nlocal enrich {enrich:.2f}", fontsize=9, color=colour)
        # Recolour the crop's frame to match its locator box.
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(colour); spine.set_linewidth(2)

    fig.suptitle(f"{title}   —   {CROP_UM:.0f} µm crops at native resolution",
                 fontsize=13, y=0.995)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _densest_thumb(green, cd31, tissue, centre):
    lo_v, hi_v = np.percentile(green[tissue], [1, 99])
    lo_c, hi_c = np.percentile(cd31[tissue], [1, 99])
    sl = _crop_slice(centre)
    return _merge_rgb(_clip(green[sl], lo_v, hi_v), _clip(cd31[sl], lo_c, hi_c))


def contact_sheet(thumbs, png_path):
    if not thumbs:
        return
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for (label, thumb), ax in zip(thumbs, axes.ravel()):
        ax.imshow(thumb)
        ax.set_title(label, fontsize=8)
    fig.suptitle(f"Densest-vasculature crop of every section "
                 f"({CROP_UM:.0f} µm) — virus (green) / CD31 (magenta)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=1)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'}); "
          f"crop {CROP_UM:.0f} µm = {CROP_PX} px\n")

    out = Path(__file__).resolve().parent.parent / "results" / "zoom_panels"
    out.mkdir(parents=True, exist_ok=True)

    thumbs = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        reporter = ("SYFP2" if "SYFP2" in reporter
                    else "tdT" if "tdT" in reporter else reporter)
        label = f"{figure} {mouse} {REGION_NAME[region]}" + (f" s{slice_id}" if slice_id else "")

        stack = tifffile.imread(path)
        if stack.ndim != 3 or stack.shape[0] != 2:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: not 2-channel")
            continue
        green, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(green, cd31)
        except ValueError as error:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue
        if min(green.shape) < CROP_PX:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: smaller than one crop")
            continue

        vessels = top_q_mask(cd31, tissue, VIS_Q)
        centres = saliency_centers(green, cd31, tissue, vessels)

        png = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.png")
        section_figure(green, cd31, tissue, vessels, centres,
                       f"{label}  ({reporter})", png)
        if centres.get("densest") is not None:
            thumbs.append((label, _densest_thumb(green, cd31, tissue, centres["densest"])))
        print(f"[{index:2d}/{len(paths)}] {png.name}")

    print(f"\nwrote {len(thumbs)} zoom figures to {out}")
    contact_sheet(thumbs, out.parent / ("zoom_contact_sheet_full.png" if full
                                        else "zoom_contact_sheet_pilot.png"))


if __name__ == "__main__":
    main()
