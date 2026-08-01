"""Vascular enrichment of the virus, without segmenting anything.

    python scripts/enrichment_by_cd31_percentile.py           # pilot: 2 mice/reporter
    python scripts/enrichment_by_cd31_percentile.py --full     # every clean mouse

The mask-based enrichment in analyse_spinal_cord.py is sound, but it depends on
one choice: the CD31 vessel mask's operating point, which decides which pixels
count as "vessel". Across defensible settings that choice moved the enrichment
magnitude and, at the margin, the thoracic-vs-lumbar ordering. This measure
removes the choice entirely.

CD31 is an endothelial stain, so its brightest pixels ARE vessels by definition —
no filter needed. Define "vessel" as the top q% of CD31 intensity within tissue,
and report

    enrichment(q) = mean virus where CD31 is in its top q%
                  / mean virus in the rest of the tissue

Sweep q. An enrichment ordering between regions that holds across q is a property
of the data, not of where a threshold was put. This is the same quantity the
mask-based `enrichment` estimates, with the segmentation operating point replaced
by an explicit, swept percentile — nothing hidden.

What is and is not corrected: the virus channel is used raw (enrichment is a
ratio of virus intensities, so a per-section gain cancels). CD31 is used raw for
ranking pixels; only the rank matters, so CD31 staining brightness does not enter.
Grey/white matter is not separated — that confound is orthogonal and applies to
the mask-based measure too. Reads Z: read-only.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import NAME, curated_paths, tissue_mask   # noqa: E402

PERCENTILES = (2, 5, 10, 20, 30)   # top-q% of CD31 called vessel
REGIONS = ("C", "T", "L", "TL")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}


def enrichment_curve(virus, cd31, tissue):
    """enrichment(q) for each q in PERCENTILES, on one section."""
    virus_t, cd31_t = virus[tissue], cd31[tissue]
    out = {}
    for q in PERCENTILES:
        cut = np.percentile(cd31_t, 100 - q)
        high = virus_t[cd31_t >= cut]
        low = virus_t[cd31_t < cut]
        out[q] = float(high.mean() / low.mean()) if high.size and low.size else float("nan")
    return out


def main():
    full = "--full" in sys.argv
    paths = curated_paths() if full else curated_paths(pilot_mice=2, slices_per_region=3)
    print(f"{len(paths)} sections ({'all clean mice' if full else 'pilot'})\n")

    rows = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        reporter = "SYFP2" if "SYFP2" in reporter else ("tdT" if "tdT" in reporter else reporter)
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
        curve = enrichment_curve(virus, cd31, tissue)
        rows.append({"figure": figure, "reporter": reporter, "mouse": mouse,
                     "region": region, "slice": slice_id or "",
                     **{f"enrich_top{q}": curve[q] for q in PERCENTILES}})
        print(f"[{index:2d}/{len(paths)}] {figure} {reporter:5s} {mouse:7s} {region:2s}  "
              + "  ".join(f"q{q}={curve[q]:.2f}" for q in PERCENTILES))

    if not rows:
        sys.exit("no sections scored")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = out / ("enrichment_cd31_percentile_full.csv" if full
                      else "enrichment_cd31_percentile_pilot.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    # sections[(figure, reporter, mouse, region, q)] -> list of per-slice values.
    sections = defaultdict(list)
    for r in rows:
        for q in PERCENTILES:
            sections[(r["figure"], r["reporter"], r["mouse"], r["region"], q)].append(
                r[f"enrich_top{q}"])

    groups = sorted({(f, rep) for (f, rep, *_ ) in sections})
    for figure, reporter in groups:
        mice = sorted({m for (f, rep, m, *_ ) in sections if (f, rep) == (figure, reporter)})
        regions = [x for x in ("C", "T", "L")
                   if any((figure, reporter, m, x, PERCENTILES[0]) in sections for m in mice)]
        if len(regions) < 2:
            continue

        def mouse_mean(mouse, region, q):
            """Average slices for one mouse x region at percentile q."""
            vals = sections.get((figure, reporter, mouse, region, q), [])
            return float(np.mean(vals)) if vals else float("nan")

        print("\n" + "=" * 74)
        print(f"{figure}  {reporter}   mice: {', '.join(mice)}")
        print("=" * 74)
        print(f"{'top-q%':>7}" + "".join(f"{LONG[x]:>11s}" for x in regions)
              + "   ordering            mice agree")
        orderings = set()
        for q in PERCENTILES:
            region_mean = {x: float(np.nanmean([mouse_mean(m, x, q) for m in mice]))
                           for x in regions}
            order = tuple(sorted(regions, key=lambda x: -region_mean[x]))
            agree = sum(1 for m in mice
                        if tuple(sorted(regions, key=lambda x: -mouse_mean(m, x, q))) == order)
            orderings.add(order)
            print(f"{q:6d}%" + "".join(f"{region_mean[x]:11.3f}" for x in regions)
                  + f"   {' > '.join(LONG[x][:4] for x in order):20s} {agree}/{len(mice)}")
        print(f"\n  distinct orderings across the q-sweep: {len(orderings)}")
        if len(orderings) == 1:
            print("  -> stable: the regional ordering is a property of the data, not the")
            print("     percentile. This is the strongest form of the result.")
        else:
            lowest = {order[-1] for order in orderings}
            if len(lowest) == 1:
                print(f"  -> top/middle depends on q, but {LONG[next(iter(lowest))]} is lowest at"
                      " every q -- that part is safe to claim.")
            else:
                print("  -> ordering not stable across q; no safe regional claim.")

    print("\n" + "=" * 74)
    print("No segmentation, no filter, no operating point: virus intensity ranked by")
    print("CD31 intensity. This is the same enrichment the mask-based measure")
    print("estimates, with the vessel-mask threshold replaced by an explicit sweep.")

    plot_curves(sections, out / csv_path.name.replace(".csv", ".png"))


def plot_curves(sections, png_path):
    """Enrichment vs q, one panel per (figure, reporter): the result, visualised.

    Each region is a line (mean across mice); individual mouse-means are faint
    points, so both the regional separation and the between-animal spread are
    visible. A line that stays below the others across the whole q-axis is the
    robust part of the result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(f, rep) for (f, rep, *_ ) in sections})
    groups = [g for g in groups
              if len([x for x in ("C", "T", "L")
                      if any((*g, m, x, PERCENTILES[0]) in sections
                             for (_f, _r, m, *_ ) in sections if (_f, _r) == g)]) >= 2]
    if not groups:
        return
    colours = {"C": "#d1495b", "T": "#edae49", "L": "#00798c"}
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 5.2), squeeze=False)
    for ax, (figure, reporter) in zip(axes[0], groups):
        mice = sorted({m for (f, r, m, *_ ) in sections if (f, r) == (figure, reporter)})
        for region in ("C", "T", "L"):
            per_mouse = []
            for mouse in mice:
                vals = [np.mean(sections.get((figure, reporter, mouse, region, q), [np.nan]))
                        for q in PERCENTILES]
                if not np.all(np.isnan(vals)):
                    per_mouse.append(vals)
                    ax.plot(PERCENTILES, vals, color=colours[region], alpha=0.25, lw=1)
            if per_mouse:
                mean = np.nanmean(per_mouse, axis=0)
                ax.plot(PERCENTILES, mean, color=colours[region], lw=2.5,
                        marker="o", label=LONG[region])
        ax.axhline(1.0, color="0.6", ls=":", lw=1)
        ax.set_xlabel("top q% of CD31 called vessel")
        ax.set_ylabel("virus enrichment (in-vessel / out)")
        ax.set_title(f"{figure}  {reporter}   (n={len(mice)} mice)")
        ax.legend(frameon=False)
    fig.suptitle("Vascular enrichment without segmentation — lines that stay lowest "
                 "across q are the robust result", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
