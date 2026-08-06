"""Per-section visualisation of the threshold-free enrichment measure.

    python scripts/visualize_sections.py            # pilot subset (fast)
    python scripts/visualize_sections.py --full     # all 39 sections

Writes one 6-panel PNG per section to results/section_panels/, plus a contact
sheet tiling one result panel per section. The panels show the primary,
threshold-free measure (scripts/enrichment_by_cd31_percentile.py): a "vessel" is
the top-q% of CD31 intensity within tissue, and enrichment is mean virus there
over mean virus in the rest. No vesselness filter, no operating point — the only
choice is q, which is swept, so the panels visualise exactly the merged result.

Panels, left to right, top then bottom:
  virus (raw)        the reporter channel, contrast-clipped to tissue percentiles
  CD31 (raw)         the endothelial ground-truth channel, same clipping
  tissue mask        section outline after the 90 um edge rim is removed
  CD31 intensity     the ranking that defines "vessel"; no threshold yet
  top-q% vessels     the q-th percentile cut — what counts as vessel at this q
  result             virus on the vessels, coloured by its own intensity, so a
                     vessel bed that the virus reaches glows and one it misses
                     stays dark; enrichment at every swept q in the title

Reads Z: read-only; writes only under results/ (which is gitignored).
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

from analyse_spinal_cord import NAME, REGION_NAME, curated_paths, tissue_mask  # noqa: E402
# top_q_mask lives in enrichment_by_cd31_percentile (the metric and the visuals must
# use the same speck-cleaned mask); re-exported here so downstream scripts that do
# `from visualize_sections import top_q_mask` keep working.
from enrichment_by_cd31_percentile import (PERCENTILES, enrichment_curve,       # noqa: E402,F401
                                           top_q_mask)

DS = 2            # display downsample; QC panels do not need full resolution
THUMB_PX = 240    # target height of each contact-sheet tile
VIS_Q = 20        # the percentile whose vessel mask is drawn; all q's in the title
                  # (top-5/10% undersegment — miss dim vessels; top-20% captures them)


def _clip(channel, tissue):
    """Channel scaled to its in-tissue 1st-99th percentiles, for display only."""
    lo, hi = np.percentile(channel[tissue], [1, 99])
    return np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _result_rgb(green, tissue, vessels):
    """Dim grey virus everywhere; vessel pixels tinted green by virus intensity.

    A vessel bed the virus reaches glows green; one it misses stays dark. This is
    the enrichment ratio made visible — no separate virus threshold is imposed,
    matching the threshold-free measure.
    """
    vg = _clip(green, tissue)
    rgb = np.stack([vg, vg, vg], axis=-1) * 0.45
    rgb[vessels, 0] = 0.0
    rgb[vessels, 1] = np.clip(vg[vessels] * 1.4, 0.0, 1.0)
    rgb[vessels, 2] = 0.0
    return rgb


def panel(green, cd31, tissue, curve, title, png_path):
    """The six-up figure for one section."""
    d = np.s_[::DS, ::DS]
    tis = tissue[d].astype(float)
    vessels = top_q_mask(cd31, tissue, VIS_Q)
    area_frac = float(vessels.sum() / tissue.sum())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
    ax = axes.ravel()

    ax[0].imshow(_clip(green, tissue)[d], cmap="gray")
    ax[0].set_title("virus (reporter), raw")

    ax[1].imshow(_clip(cd31, tissue)[d], cmap="gray")
    ax[1].set_title("CD31 (ground truth), raw")

    dim_cd31 = _clip(cd31, tissue)[d] * np.where(tis > 0.5, 1.0, 0.25)
    ax[2].imshow(dim_cd31, cmap="gray")
    if tis.any():
        ax[2].contour(tis, levels=[0.5], colors="#ffd400", linewidths=0.6)
    ax[2].set_title("tissue mask — 90 µm rim removed")

    cd31_rank = np.where(tissue, _clip(cd31, tissue), np.nan)[d]
    ax[3].set_facecolor("black")
    im = ax[3].imshow(cd31_rank, cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.02)
    ax[3].set_title("CD31 intensity (defines vessel rank)")

    base = _clip(cd31, tissue)[d]
    vessel_rgb = np.stack([base, base, base], axis=-1)
    vessel_rgb[vessels[d]] = [1.0, 0.2, 0.2]
    ax[4].imshow(vessel_rgb)
    ax[4].set_title(f"top-{VIS_Q}% CD31 = vessel (area frac {area_frac:.3f})")

    ax[5].imshow(_result_rgb(green[d], tissue[d], vessels[d]))
    ax[5].set_title(f"virus on vessels (green = virus-rich)\n"
                    f"enrichment at top-{VIS_Q}%: {curve[VIS_Q]:.2f}")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    shown = "  ".join(f"q{q}={curve[q]:.2f}" for q in PERCENTILES
                      if not np.isnan(curve[q]))
    fig.suptitle(f"{title}     enrichment  {shown}", fontsize=13.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _thumbnail(green, tissue, cd31):
    """A small result RGB for the contact sheet."""
    step = max(1, green.shape[0] // THUMB_PX)
    s = np.s_[::step, ::step]
    vessels = top_q_mask(cd31, tissue, VIS_Q)
    return _result_rgb(green[s], tissue[s], vessels[s])


def contact_sheet(thumbs, png_path):
    """Tile one result thumbnail per section into a single overview image."""
    if not thumbs:
        return
    cols = min(6, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for (label, thumb, enrich), ax in zip(thumbs, axes.ravel()):
        ax.imshow(thumb)
        title = f"{label}\nenrich {enrich:.2f}" if not np.isnan(enrich) else f"{label}\nenrich n/a"
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"All sections — virus on top-{VIS_Q}% CD31 vessels "
                 "(green = virus-rich)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=1)
    print(f"{len(paths)} sections ({'all' if full else 'pilot'})\n")

    out = Path(__file__).resolve().parent.parent / "results" / "section_panels"
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
        curve = enrichment_curve(green, cd31, tissue)

        png = out / (f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
                     f"_s{slice_id or '0'}.png")
        panel(green, cd31, tissue, curve, f"{label}  ({reporter})", png)
        thumbs.append((label, _thumbnail(green, tissue, cd31), curve[VIS_Q]))
        print(f"[{index:2d}/{len(paths)}] {png.name}   enrich q{VIS_Q}={curve[VIS_Q]:.2f}")

    print(f"\nwrote {len(thumbs)} panels to {out}")
    contact_sheet(thumbs, out.parent / ("section_contact_sheet_full.png" if full
                                        else "section_contact_sheet_pilot.png"))


if __name__ == "__main__":
    main()
