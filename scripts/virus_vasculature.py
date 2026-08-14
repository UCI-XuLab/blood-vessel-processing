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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import load_sections, section_paths, virus_cut         # noqa: E402
from enrichment_by_cd31_percentile import enrichment_curve                      # noqa: E402
from visualize_sections import VIS_Q, _clip, contact_sheet, top_q_mask          # noqa: E402

DS = 2
THUMB_PX = 240


def virus_positive(virus, tissue, vessels):
    """Per-image virus-positive: parenchyma median + VIRUS_K*MAD (the pipeline rule)."""
    cut, _ = virus_cut(virus[tissue & ~vessels])
    return (virus > cut) & tissue


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


def main():
    full = "--full" in sys.argv
    paths = section_paths(full)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "virus_vasculature"
    out.mkdir(parents=True, exist_ok=True)

    tiles = []
    for s in load_sections(paths):
        png = out / f"{s.stem}.png"
        coverage = panel(s.virus, s.cd31, s.tissue, f"{s.label}  ({s.reporter})", png)
        tiles.append((f"{s.label}\ncoverage {coverage:.2f}",
                      _thumbnail(s.virus, s.cd31, s.tissue)))
        print(f"{s.counter} {png.name}   coverage {coverage:.2f}")

    print(f"\nwrote {len(tiles)} panels to {out}")
    contact_sheet(tiles, out.parent / ("virus_vasculature_contact_full.png" if full
                                       else "virus_vasculature_contact_pilot.png"),
                  f"Virus-labelled vasculature (virus⁺ on top-{VIS_Q}% CD31 vessels) "
                  "- every section")


if __name__ == "__main__":
    main()
