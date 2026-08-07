"""Assemble a self-contained handoff package of results, figures and segmentations.

    python scripts/export_handoff.py            # -> results/handoff/

Bundles everything a collaborator needs to re-analyse or re-visualise the
spinal-cord vascular-specificity results WITHOUT the raw imaging data (which
stays read-only on Z:) and without this repo's environment:

  metrics/         the two measure CSVs + a column dictionary
  figures/         the publication result plots (PNG + PDF)
  segmentations/
      masks/       per-section 4-channel mask TIFs (tissue, CD31 vessel,
                   virus vessel, CD31 top-20%) for re-analysis in any tool
      contours/    per-section contour JPGs (visual QC of the masks)
  visualization/   section panels, 800 um zoom crops, virus-vasculature
                   panels, and the contact-sheet overviews
  notebooks/       the two executed analysis notebooks (figures embedded)
  README.md        dataset, methods, every parameter, the finding, a file
                   guide, how to load the masks, and the caveats

The masks are recomputed here (not just copied) so the package carries the
actual boolean segmentations, not only their renderings. Reads Z: read-only.
"""

import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import NAME, curated_paths, tissue_mask             # noqa: E402
from dice_between_channels import (CD31_HIGH, CD31_LOW, MIN_VESSEL_PX,        # noqa: E402
                                   REFERENCE, VIRUS_HIGH, VIRUS_LOW, _segment,
                                   channels)
from visualize_sections import VIS_Q, top_q_mask                             # noqa: E402
import matplotlib.pyplot as plt   # after visualize_sections, which sets the Agg backend

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "handoff"

# Mask channels, in the order they are written into each section's TIF.
MASK_CHANNELS = ["tissue", "cd31_vessel", "virus_vessel", "cd31_top20pct"]


def clean_stem(path):
    """Gallery-style stem, e.g. 'Fig1_M131_C_SYFP2_s3' — matches the JPG/PNG names."""
    figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
    reporter = "SYFP2" if "SYFP2" in reporter else ("tdT" if "tdT" in reporter else reporter)
    stem = f"{figure.replace(' ', '')}_{mouse}_{region}_{reporter}"
    return stem + (f"_s{slice_id}" if slice_id else "")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:                                                # noqa: BLE001
        return "unknown"


def render_overlay(virus, cd31, tissue, cd31_ves, virus_ves, cd31_top, title, out_path):
    """Section with its vessel segmentation drawn on top, as a high-quality JPG:
    each channel grayscale with its own vessel-mask contour (thin, so the vessels
    stay visible). Downsampled 2x; JPG (q92) keeps each file well under 1 MB."""
    d = np.s_[::2, ::2]

    def grey(channel):
        lo, hi = np.percentile(channel[tissue], [1, 99])
        return np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1)[d]

    fig, ax = plt.subplots(1, 2, figsize=(17, 8.5))
    ax[0].imshow(grey(cd31), cmap="gray")
    ax[0].contour(cd31_ves[d].astype(float), [0.5], colors="#00e5ff", linewidths=0.5)
    ax[0].contour(cd31_top[d].astype(float), [0.5], colors="#ffb000", linewidths=0.4)
    ax[0].set_title("CD31 + vessel mask (cyan) + top-20% (amber)", fontsize=11)
    ax[1].imshow(grey(virus), cmap="gray")
    ax[1].contour(virus_ves[d].astype(float), [0.5], colors="#39ff14", linewidths=0.5)
    ax[1].set_title("virus + vessel mask (green)", fontsize=11)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


def export_masks(mask_dir, overlay_dir):
    """Per section: write a 4-channel mask TIF and a contour-overlay PNG."""
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    paths = curated_paths()
    written, skipped = 0, []
    for index, path in enumerate(paths, 1):
        stem = clean_stem(path)
        try:
            virus, cd31 = channels(path)
            tissue = tissue_mask(virus, cd31)
            cd31_ves = _segment(cd31, tissue, CD31_LOW, CD31_HIGH)
            virus_ves = _segment(virus, tissue, VIRUS_LOW, VIRUS_HIGH)
            cd31_top = top_q_mask(cd31, tissue, VIS_Q)
        except Exception as error:                                   # noqa: BLE001
            print(f"[{index:2d}/{len(paths)}] SKIP {stem}: {error}")
            skipped.append((stem, str(error)))
            continue
        stack = np.stack([tissue, cd31_ves, virus_ves, cd31_top]).astype(np.uint8) * 255
        tifffile.imwrite(mask_dir / f"{stem}_masks.tif", stack,
                         compression="zlib", metadata={"axes": "CYX"})
        render_overlay(virus, cd31, tissue, cd31_ves, virus_ves, cd31_top, stem,
                       overlay_dir / f"{stem}_overlay.jpg")
        written += 1
        print(f"[{index:2d}/{len(paths)}] {stem}  ({stack.shape[2]}x{stack.shape[1]})")
    return written, skipped


def copy_one(src, dst_dir, rename=None):
    if not src.exists():
        print(f"  (missing, skipped) {src.name}")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / (rename or src.name))
    return 1


def copy_tree(src, dst):
    if not src.exists():
        print(f"  (missing, skipped) {src}/")
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return len(list(dst.iterdir()))


def write_readme(readme_path, n_masks, skipped):
    commit = git_commit()
    text = f"""# Spinal-cord vascular specificity — analysis handoff

Enhancer-AAV vascular targeting along the mouse spinal-cord axis: how strongly a
virus reporter (SYFP2 or tdTomato) is enriched on the vasculature, measured
against a CD31 endothelial ground truth. This package is the derived result set —
metrics, figures, and segmentations — assembled for downstream analysis and
visualisation.

- **Exported:** {date.today().isoformat()}  ·  **repo commit:** `{commit}`
- **Source images:** `Z:\\Lab\\Eric V\\BEC Spinal Cords\\composites_EV` (read-only;
  NOT included here). Two-channel composites, `channel 1 = virus`, `channel 2 =
  CD31 (ground truth)`. Naming: `Fig N_Mouse_Region_reporter_CD31-mag_sliceK.tif`,
  region in {{C, T, L, TL}}.
- **Sections:** {n_masks} (SYFP2: M131/M132/M133; tdT: M63 C/T/L, M87 TL).

## The finding

**Virus vascular enrichment is lowest in lumbar.** For SYFP2 (n=3 mice) the full
ordering **cervical > thoracic > lumbar is stable across the entire percentile
sweep** (q = 2/5/10/20/30) — the strongest form of the result, not an artefact of
one threshold. tdT is n=1 per figure (M63, M87) and shows no stable regional
ordering — more tdT animals is the main gap. See `figures/fig_enrichment_by_region`.

## Two measures (read them together)

1. **Threshold-free CD31-percentile enrichment — PRIMARY.**
   `metrics/enrichment_by_cd31_percentile.csv`. "Vessel" = the top-q% of CD31
   intensity inside tissue; enrichment(q) = mean virus there / mean virus in the
   rest of the tissue. No segmentation of the virus, no operating point — swept
   over q = 2/5/10/20/30. `coverage` = fraction of the top-q% CD31 that is
   virus-positive (parenchyma median + 3·MAD). Headline q = {VIS_Q}%.

2. **Per-channel vessel-mask agreement — SECONDARY.**
   `metrics/dice_between_channels.csv`. Segment vessels in BOTH channels with a
   graded Jerman filter and compare. Dice/Jaccard are symmetric overlap;
   `specificity` = |virus∩CD31|/|virus| (how much of the virus mask sits on a
   vessel); `coverage` = |virus∩CD31|/|CD31| (how much vasculature the virus
   reaches). The virus mask is deliberately uncleaned — its off-vessel signal is
   the vector's non-specificity, not an error.

## Segmentations (`segmentations/`)

`masks/<section>_masks.tif` — one 4-channel `uint8` TIF per section (0 / 255),
same H×W as the source. Channel order (axis 'C'):

| # | channel | definition |
|---|---------|------------|
| 0 | `tissue` | entropy-guided GrabCut silhouette of the section |
| 1 | `cd31_vessel` | CD31 graded-Jerman vessel mask, hysteresis {CD31_LOW}/{CD31_HIGH} |
| 2 | `virus_vessel` | virus graded-Jerman vessel mask, hysteresis {VIRUS_LOW}/{VIRUS_HIGH} |
| 3 | `cd31_top20pct` | CD31 top-{VIS_Q}% percentile mask (the primary-measure "vessel") |

Load in Python:

```python
import tifffile
m = tifffile.imread("segmentations/masks/Fig1_M131_C_SYFP2_s3_masks.tif")  # (4, H, W)
tissue, cd31_vessel, virus_vessel, cd31_top20 = (m > 0)   # booleans
```

Or drag a `_masks.tif` into ImageJ/Fiji (4-slice stack) or napari (4 channels).

Two rendered views of the same masks over the section images:
- `section_overlays/<section>_overlay.jpg` — each channel with its vessel contour
  (CD31 vessel cyan + top-20% amber; virus vessel green).
- `contours/<section>.jpg` — a 2x2 method-QC grid (rows CD31/virus, cols Jerman
  vs top-q%).

## Methods and every parameter

- **Physical scale:** 0.650193 µm/px (from the calibrated files).
- **Tissue mask:** the vendored entropy-guided seeded GrabCut masker
  (`vessel_utils/_vendor`, from UCI-XuLab-RegTools). Hugs the true edge, keeps
  torn fragments (≥1% of the largest), rejects background haze. It keeps the
  bright pial edge (no rim erosion) — accepting edge staining rather than dropping
  real near-edge vessels.
- **Vesselness:** Jerman, bounded [0,1], sigmas 1.5/3.0/6.0/12.0 µm (capillary→
  venule radius), single dataset-wide tau reference (`REFERENCE = {REFERENCE}`) so
  a threshold means the same thing in every section and channel.
- **Vessel hysteresis:** CD31 = {CD31_LOW}/{CD31_HIGH}, virus = {VIRUS_LOW}/{VIRUS_HIGH}
  (virus stricter than CD31; the virus is dominated by non-vascular neuronal
  expression, so a stricter cut keeps only vessel-associated signal). Min vessel
  size {MIN_VESSEL_PX} px (~6 µm²).
- **Aggregation:** slices are averaged within mouse×region first; figures (Fig 1
  SYFP2, Fig 2a/2b tdT) are kept separate — pooling would confound reporter and
  construct with region.

## Caveats

- **The virus channel cannot be cleanly segmented into vessels** — proven, not
  assumed. Its off-vessel signal is genuine neuronal expression that is
  structurally vessel-like. That is why the PRIMARY measure never segments the
  virus; use `dice_between_channels.csv` only as a cross-check.
- **tdT is n=1 per figure** — no regional claim for tdT.
- **No manual / stereological ground-truth validation yet.**
- The older mask-based measure (`spinal_cord_specificity*.csv`, not included) was
  superseded: its calibrated reference over-segmented ~45% of tissue.

## File guide

```
metrics/       enrichment_by_cd31_percentile.csv, dice_between_channels.csv,
               DATA_DICTIONARY.md
figures/       fig_enrichment_by_region, fig_coverage_by_region,
               fig_enrichment_vs_q, fig_method_agreement (each PNG + PDF),
               enrichment_curves.png
segmentations/ masks/ (39 x 4-channel TIF), section_overlays/ (39 x JPG:
               section + vessel contours), contours/ (39 x JPG: 2x2 method QC)
visualization/ section_panels/, zoom_crops/ (800 um), virus_vasculature/,
               contact_sheets/ (three overview PNGs)
```

This package is outputs only — no code. The analysis notebooks and scripts live
in the source repository.
"""
    if skipped:
        text += "\n## Sections skipped during export\n\n" + \
            "\n".join(f"- {stem}: {why}" for stem, why in skipped) + "\n"
    readme_path.write_text(text, encoding="utf-8")


def write_data_dictionary(path):
    path.write_text("""# Data dictionary

Both CSVs carry identifier columns `figure`, `reporter`, `mouse`, `region`
(C/T/L/TL), `slice`. One row per section (slice). Aggregate by averaging within
`mouse` x `region` before comparing regions.

## enrichment_by_cd31_percentile.csv  (PRIMARY)

| column | meaning |
|--------|---------|
| `enrichment_q2` … `enrichment_q30` | mean virus in the top-q% of CD31 / mean virus in the rest of tissue, for q = 2/5/10/20/30. > 1 means virus is enriched on vessels. |
| `coverage_q5`, `coverage_q10` | fraction of the top-q% CD31 area that is virus-positive (virus > parenchyma median + 3·MAD). |

## dice_between_channels.csv  (SECONDARY)

| column | meaning |
|--------|---------|
| `dice`, `jaccard` | symmetric overlap of the virus and CD31 vessel masks. |
| `specificity` | \\|virus ∩ CD31\\| / \\|virus\\| — fraction of the virus vessel mask that sits on a CD31 vessel. |
| `coverage` | \\|virus ∩ CD31\\| / \\|CD31\\| — fraction of the CD31 vasculature the virus mask reaches. |
| `virus_af`, `cd31_af` | area fraction of each vessel mask within tissue. |

Directional pair (`specificity`, `coverage`) is more interpretable than `dice`
alone. With few mice, read direction-consistency across animals, not p-values.
""", encoding="utf-8")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print(f"assembling handoff at {OUT}\n")

    print("segmentations/masks + section_overlays (recomputing) ...")
    n_masks, skipped = export_masks(OUT / "segmentations" / "masks",
                                    OUT / "segmentations" / "section_overlays")

    print("\nmetrics ...")
    copy_one(RESULTS / "enrichment_cd31_percentile_full.csv", OUT / "metrics",
             "enrichment_by_cd31_percentile.csv")
    copy_one(RESULTS / "dice_between_channels_full.csv", OUT / "metrics",
             "dice_between_channels.csv")
    write_data_dictionary(OUT / "metrics" / "DATA_DICTIONARY.md")

    print("figures ...")
    for base in ("fig_enrichment_by_region", "fig_coverage_by_region",
                 "fig_enrichment_vs_q", "fig_method_agreement"):
        for ext in (".png", ".pdf"):
            copy_one(RESULTS / f"{base}{ext}", OUT / "figures")
    copy_one(RESULTS / "enrichment_cd31_percentile_full.png", OUT / "figures",
             "enrichment_curves.png")

    print("segmentations/contours ...")
    copy_tree(RESULTS / "segmentation_contours", OUT / "segmentations" / "contours")

    print("visualization ...")
    copy_tree(RESULTS / "section_panels", OUT / "visualization" / "section_panels")
    copy_tree(RESULTS / "zoom_panels", OUT / "visualization" / "zoom_crops")
    copy_tree(RESULTS / "virus_vasculature", OUT / "visualization" / "virus_vasculature")
    for sheet in ("section_contact_sheet_full.png", "zoom_contact_sheet_full.png",
                  "virus_vasculature_contact_full.png"):
        copy_one(RESULTS / sheet, OUT / "visualization" / "contact_sheets")

    print("README + data dictionary ...")
    write_readme(OUT / "README.md", n_masks, skipped)

    total = sum(1 for _ in OUT.rglob("*") if _.is_file())
    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\ndone: {total} files, {size_mb:.0f} MB in {OUT}")
    print(f"      {n_masks} mask sets" + (f", {len(skipped)} skipped" if skipped else ""))


if __name__ == "__main__":
    main()
