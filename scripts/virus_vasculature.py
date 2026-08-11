"""CD31-guided virus vasculature: which vessels the virus actually labels.

    python scripts/virus_vasculature.py            # pilot subset (fast)
    python scripts/virus_vasculature.py --full     # all sections

The virus channel cannot be segmented into vessels on its own -- its off-vessel
signal is genuine neuronal expression that is structurally vessel-like, so a
vesselness filter fires on it (demonstrated: specificity ~0.17-0.31 against CD31,
and background suppression does not help). So the vessel geometry is taken from
CD31 (top-q% intensity, the threshold-free definition) and the virus only decides
which of those vessels are labelled. That gives a clean "virus-labelled
vasculature" mask and the coverage metric -- the area companion to enrichment.

Per section, four panels:
  virus (raw)              the reporter channel
  CD31 top-q% vessels      the vessel geometry, from CD31 alone
  virus-positive placement on the vessels (green) vs in parenchyma (orange)
  virus-labelled vessels   virus-positive AND on a CD31 vessel -- the clean mask

Title carries coverage (virus-positive fraction of the vessels) and enrichment.
A contact sheet tiles the clean mask of every section.

Reads Z: read-only; writes only under results/ (gitignored).
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
from enrichment_by_cd31_percentile import VIRUS_K, coverage_curve, enrichment_curve  # noqa: E402
from visualize_sections import VIS_Q, _clip, top_q_mask                          # noqa: E402

DS = 2
THUMB_PX = 240


def virus_positive(virus, tissue, vessels):
    """Per-image virus-positive: parenchyma median + VIRUS_K*MAD (the pipeline rule)."""
    parenchyma = tissue & ~vessels
    background = np.median(virus[parenchyma])
    mad = 1.4826 * np.median(np.abs(virus[parenchyma] - background))
    return (virus > background + VIRUS_K * mad) & tissue


def _labelled_rgb(virus, tissue, on_vessel):
    """Dim virus with the virus-labelled vessels lit green by virus intensity."""
    vg = _clip(virus, tissue)
    rgb = np.stack([vg, vg, vg], axis=-1) * 0.4
    rgb[on_vessel, 0] = 0.0
    rgb[on_vessel, 1] = np.clip(vg[on_vessel] * 1.5, 0.0, 1.0)
    rgb[on_vessel, 2] = 0.0
    return rgb


def panel(virus, cd31, tissue, title, png_path):
    vessels = top_q_mask(cd31, tissue, VIS_Q)
    vpos = virus_positive(virus, tissue, vessels)
    on, off = vpos & vessels, vpos & ~vessels
    coverage = on.sum() / max(vessels.sum(), 1)
    enrich = enrichment_curve(virus, cd31, tissue)[VIS_Q]
    d = np.s_[::DS, ::DS]

    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2))
    ax[0].imshow(_clip(virus, tissue)[d], cmap="gray")
    ax[0].set_title("virus (reporter), raw")

    base = _clip(cd31, tissue)[d]
    rgb = np.stack([base, base, base], axis=-1)
    rgb[vessels[d]] = [1.0, 0.2, 0.2]
    ax[1].imshow(rgb)
    ax[1].set_title(f"CD31 top-{VIS_Q}% vessels")

    vg = _clip(virus, tissue)[d] * 0.5
    place = np.stack([vg, vg, vg], axis=-1)
    place[on[d]] = [0.1, 1.0, 0.1]
    place[off[d]] = [1.0, 0.55, 0.0]
    ax[2].imshow(place)
    ax[2].set_title(f"virus⁺: on vessels (green) / parenchyma (orange)")

    ax[3].imshow(_labelled_rgb(virus[d], tissue[d], on[d]))
    ax[3].set_title("virus-labelled vasculature")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}     coverage {coverage:.2f}  (virus reaches this fraction "
                 f"of the vessels)     enrichment {enrich:.2f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return coverage


def _thumbnail(virus, cd31, tissue):
    vessels = top_q_mask(cd31, tissue, VIS_Q)
    on = virus_positive(virus, tissue, vessels) & vessels
    step = max(1, virus.shape[0] // THUMB_PX)
    s = np.s_[::step, ::step]
    return _labelled_rgb(virus[s], tissue[s], on[s])


def contact_sheet(thumbs, png_path):
    if not thumbs:
        return
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for (label, thumb, coverage), ax in zip(thumbs, axes.ravel()):
        ax.imshow(thumb)
        ax.set_title(f"{label}\ncoverage {coverage:.2f}", fontsize=8)
    fig.suptitle(f"Virus-labelled vasculature (virus⁺ on top-{VIS_Q}% CD31 vessels) "
                 "- every section", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=1)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "virus_vasculature"
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
        virus, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(virus, cd31)
        except ValueError as error:
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue
        png = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.png")
        coverage = panel(virus, cd31, tissue, f"{label}  ({reporter})", png)
        thumbs.append((label, _thumbnail(virus, cd31, tissue), coverage))
        print(f"[{index:2d}/{len(paths)}] {png.name}   coverage {coverage:.2f}")

    print(f"\nwrote {len(thumbs)} panels to {out}")
    contact_sheet(thumbs, out.parent / ("virus_vasculature_contact_full.png" if full
                                        else "virus_vasculature_contact_pilot.png"))


if __name__ == "__main__":
    main()
