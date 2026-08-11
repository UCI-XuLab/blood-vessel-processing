"""Publication figures for the vascular-enrichment result, exported PNG + PDF.

    python scripts/plot_results.py

Reads the CSVs written by the analysis scripts (results/ is gitignored, so run
those first) and writes three figures, each as a raster PNG for review and a
vector PDF for figure assembly:

  fig_enrichment_by_region   the headline: virus enrichment per region, one point
                             per mouse (slices averaged first) over the group mean.
                             Lumbar sits lowest in every mouse.
  fig_enrichment_vs_q        robustness: enrichment against the vessel-percentile
                             q. A region ordering that holds across q is a property
                             of the data, not of where a threshold was put.
  fig_method_agreement       enrichment per region from two independent measures —
                             threshold-free CD31-percentile and the Jerman-mask
                             pipeline — normalised to cervical so the shapes are
                             comparable. Both place lumbar lowest; they disagree on
                             cervical vs thoracic, which is the known ambiguous part.

Aggregation is nested: slices are averaged within a mouse x region before mice are
combined, so three slices of one animal count as one animal, not three.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
PERCENTILE_CSV = RESULTS / "enrichment_cd31_percentile_full.csv"
MASK_CSV = RESULTS / "spinal_cord_specificity.csv"
DICE_CSV = RESULTS / "dice_between_channels_full.csv"

QS = (2, 5, 10, 20, 30)
HEADLINE_Q = 10   # representative percentile for the headline region plots
REGION_ORDER = ("C", "T", "L", "TL")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}
COLOUR = {"C": "#d1495b", "T": "#edae49", "L": "#00798c", "TL": "#6b6b6b"}
REPORTERS = ("SYFP2", "tdT")


def load(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _reporter(row):
    r = row.get("reporter", "")
    return "SYFP2" if "SYFP2" in r else "tdT" if "tdT" in r else r


def _mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def mouse_region(rows, reporter, value_of):
    """{mouse: {region: mean over its slices}} for one reporter.

    value_of(row) pulls the metric from a row (percentile column or mask column),
    returning nan when absent so a missing slice never poisons the mean.
    """
    slices = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if _reporter(row) != reporter:
            continue
        slices[row["mouse"]][row["region"]].append(value_of(row))
    return {mouse: {region: _mean(vals) for region, vals in regions.items()}
            for mouse, regions in slices.items()}


def regions_present(per_mouse):
    present = {region for regions in per_mouse.values() for region in regions}
    return [r for r in REGION_ORDER if r in present]


def _by_region_figure(rows, value_of, ylabel, suptitle, stem, hline=1.0):
    """Per-region metric at the headline q: mouse points over the group mean ± sd.

    Slices are already collapsed inside mouse_region, so each faint line is one
    animal and the diamonds are the across-mouse mean. Shared by the enrichment
    and coverage figures — the same aggregation, a different column.
    """
    fig, axes = plt.subplots(1, len(REPORTERS), figsize=(6.4 * len(REPORTERS), 5.4),
                             squeeze=False)
    for ax, reporter in zip(axes[0], REPORTERS):
        per_mouse = mouse_region(rows, reporter, value_of)
        regions = regions_present(per_mouse)
        if not regions:
            ax.axis("off"); continue
        xs = np.arange(len(regions))
        mice = sorted(per_mouse)
        for i, mouse in enumerate(mice):
            ys = [per_mouse[mouse].get(r, np.nan) for r in regions]
            jitter = (i - (len(mice) - 1) / 2) * 0.04
            ax.plot(xs + jitter, ys, "-o", color="0.6", lw=1, ms=5,
                    alpha=0.7, label=mouse if reporter == REPORTERS[0] else None)
        means = [_mean([per_mouse[m].get(r, np.nan) for m in mice]) for r in regions]
        sds = [np.nanstd([per_mouse[m].get(r, np.nan) for m in mice]) for r in regions]
        ax.errorbar(xs, means, yerr=sds, fmt="D", ms=10, lw=2, capsize=6,
                    color="#2b2b2b", zorder=5, label="mouse mean ± sd")
        if hline is not None:
            ax.axhline(hline, color="0.7", ls=":", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([LONG[r] for r in regions])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{reporter}   (n={len(mice)} mice)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    _save(fig, stem)


def fig_enrichment_by_region(rows):
    """Enrichment per region at the headline q, mouse points over the group mean."""
    _by_region_figure(
        rows, lambda r: float(r[f"enrich_top{HEADLINE_Q}"]),
        f"virus enrichment (top-{HEADLINE_Q}% CD31)",
        "Virus vascular enrichment by spinal-cord region — lumbar is lowest in every mouse",
        "fig_enrichment_by_region")


def fig_coverage_by_region(rows):
    """Coverage (virus-positive fraction of the vessels) per region, the area companion."""
    if not any(f"cover_top{HEADLINE_Q}" in r for r in rows):
        print("   (skipping coverage figure: CSV predates the coverage columns)")
        return
    _by_region_figure(
        rows, lambda r: float(r[f"cover_top{HEADLINE_Q}"]),
        f"virus coverage (fraction of top-{HEADLINE_Q}% CD31 vessels)",
        "Virus coverage of the vasculature by region — SYFP2 lumbar reaches the fewest vessels",
        "fig_coverage_by_region", hline=None)


def fig_enrichment_vs_q(rows):
    """Enrichment vs the vessel percentile q; a stable ordering is data, not threshold."""
    fig, axes = plt.subplots(1, len(REPORTERS), figsize=(6.4 * len(REPORTERS), 5.2),
                             squeeze=False)
    for ax, reporter in zip(axes[0], REPORTERS):
        drawn = False
        for region in REGION_ORDER:
            per_mouse_q = []
            for q in QS:
                pm = mouse_region(rows, reporter, lambda r, q=q: float(r[f"enrich_top{q}"]))
                per_mouse_q.append([pm[m].get(region, np.nan) for m in sorted(pm)])
            region_mean = [_mean(col) for col in per_mouse_q]
            if all(np.isnan(v) for v in region_mean):
                continue
            drawn = True
            ax.plot(QS, region_mean, "-o", color=COLOUR[region], lw=2.4, ms=6,
                    label=LONG[region])
            for series in np.array(per_mouse_q).T:
                ax.plot(QS, series, color=COLOUR[region], lw=0.8, alpha=0.25)
        if not drawn:
            ax.axis("off"); continue
        ax.axhline(1.0, color="0.7", ls=":", lw=1)
        ax.set_xlabel("top q% of CD31 called vessel")
        ax.set_ylabel("virus enrichment")
        ax.set_title(reporter)
        ax.legend(frameon=False)
    fig.suptitle("Enrichment is stable across the vessel-percentile sweep",
                 fontsize=13)
    _save(fig, "fig_enrichment_vs_q")


def fig_method_agreement(pct_rows, mask_rows):
    """Regional ordering from two independent measures, normalised to cervical."""
    if not mask_rows:
        print("   (skipping method-agreement figure: mask CSV not found)")
        return
    fig, axes = plt.subplots(1, len(REPORTERS), figsize=(6.4 * len(REPORTERS), 5.2),
                             squeeze=False)
    methods = [
        ("CD31 percentile (top-5%)", pct_rows,
         lambda r: float(r[f"enrich_top{HEADLINE_Q}"]), "#00798c", "-o"),
        ("Jerman mask", mask_rows, lambda r: float(r["enrichment"]), "#d1495b", "--s"),
    ]
    for ax, reporter in zip(axes[0], REPORTERS):
        regions = None
        for name, rows, value_of, colour, style in methods:
            per_mouse = mouse_region(rows, reporter, value_of)
            regions = regions or regions_present(per_mouse)
            regions = [r for r in regions if r != "TL"] or regions  # C/T/L axis
            mice = sorted(per_mouse)
            means = np.array([_mean([per_mouse[m].get(r, np.nan) for m in mice])
                              for r in regions])
            if np.all(np.isnan(means)) or np.isnan(means[0]) or means[0] == 0:
                continue
            ax.plot(np.arange(len(regions)), means / means[0], style, color=colour,
                    lw=2.2, ms=8, label=name)
        if regions:
            ax.set_xticks(np.arange(len(regions)))
            ax.set_xticklabels([LONG[r] for r in regions])
        ax.axhline(1.0, color="0.7", ls=":", lw=1)
        ax.set_ylabel("enrichment, relative to cervical")
        ax.set_title(reporter)
        ax.legend(frameon=False)
    fig.suptitle("Both measures place lumbar lowest; the cervical–thoracic "
                 "order differs by method", fontsize=13)
    _save(fig, "fig_method_agreement")


def fig_agreement_by_region(dice_rows):
    """Dice / precision / recall between the virus and CD31 Jerman masks, per region."""
    if not dice_rows:
        print("   (skipping agreement figure: dice CSV not found)")
        return
    metrics = [("dice", "dice", "#2b2b2b", "-o"),
               ("precision", "specificity", "#d1495b", "-s"),
               ("recall", "coverage", "#00798c", "-^")]
    fig, axes = plt.subplots(1, len(REPORTERS), figsize=(6.4 * len(REPORTERS), 5.2),
                             squeeze=False)
    for ax, reporter in zip(axes[0], REPORTERS):
        regions = None
        for label, col, colour, style in metrics:
            per_mouse = mouse_region(dice_rows, reporter, lambda r, c=col: float(r[c]))
            if regions is None:
                present = regions_present(per_mouse)
                regions = [r for r in present if r != "TL"] or present
            mice = sorted(per_mouse)
            means = [_mean([per_mouse[m].get(r, np.nan) for m in mice]) for r in regions]
            sds = [np.nanstd([per_mouse[m].get(r, np.nan) for m in mice]) for r in regions]
            ax.errorbar(np.arange(len(regions)), means, yerr=sds, fmt=style, color=colour,
                        lw=2, ms=7, capsize=4, label=label)
        if not regions:
            ax.axis("off"); continue
        ax.set_xticks(np.arange(len(regions)))
        ax.set_xticklabels([LONG[r] for r in regions])
        ax.set_ylim(0, 1)
        ax.set_ylabel("virus vs CD31 vessel-mask agreement")
        ax.set_title(f"{reporter}   (n={len(mice)} mice)")
        ax.legend(frameon=False)
    fig.suptitle("Virus–CD31 vessel-mask agreement by region (Jerman masks): "
                 "dice / precision / recall", fontsize=13)
    _save(fig, "fig_agreement_by_region")


def _save(fig, stem):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        path = RESULTS / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    pct_rows = load(PERCENTILE_CSV)
    mask_rows = load(MASK_CSV)
    if not pct_rows:
        sys.exit(f"no percentile CSV at {PERCENTILE_CSV}; "
                 "run scripts/enrichment_by_cd31_percentile.py --full first")
    print(f"{len(pct_rows)} percentile rows, {len(mask_rows)} mask rows\n")
    fig_enrichment_by_region(pct_rows)
    fig_coverage_by_region(pct_rows)
    fig_enrichment_vs_q(pct_rows)
    fig_method_agreement(pct_rows, mask_rows)
    fig_agreement_by_region(load(DICE_CSV))


if __name__ == "__main__":
    main()
