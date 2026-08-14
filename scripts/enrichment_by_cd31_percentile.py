"""Vascular enrichment of the virus, without segmenting anything.

    python scripts/enrichment_by_cd31_percentile.py           # pilot: 2 mice/reporter
    python scripts/enrichment_by_cd31_percentile.py --full     # every clean mouse

The mask-based enrichment in analyse_spinal_cord.py is sound, but it depends on
one choice: the CD31 vessel mask's operating point, which decides which pixels
count as "vessel". Across defensible settings that choice moved the enrichment
magnitude and, at the margin, the thoracic-vs-lumbar ordering. This measure
removes the choice entirely.

CD31 is an endothelial stain, so its brightest pixels ARE vessels by definition -
no filter needed. Define "vessel" as the top q% of CD31 intensity within tissue,
and report

    enrichment(q) = mean virus where CD31 is in its top q%
                  / mean virus in the rest of the tissue

Sweep q. An enrichment ordering between regions that holds across q is a property
of the data, not of where a threshold was put. This is the same quantity the
mask-based `enrichment` estimates, with the segmentation operating point replaced
by an explicit, swept percentile - nothing hidden.

What is and is not corrected: the virus channel is used raw (enrichment is a
ratio of virus intensities, so a per-section gain cancels). CD31 is used raw for
ranking pixels; only the rank matters, so CD31 staining brightness does not enter.
Grey/white matter is not separated - that confound is orthogonal and applies to
the mask-based measure too. Reads Z: read-only.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.morphology import remove_small_objects

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_spinal_cord import (REGION_NAME as LONG, UM_PER_PX,             # noqa: E402
                                 load_sections, section_paths, virus_cut)
from vessel_utils.sweep import write_csv                                      # noqa: E402

PERCENTILES = (2, 5, 10, 20, 30)   # top-q% of CD31 called vessel
REGIONS = ("C", "T", "L", "TL")
# Speck cleanup: drop connected components smaller than ~8 µm² (below a capillary
# cross-section), so isolated bright-CD31 noise pixels are not counted as vessel.
# On a clean section this changes the vessel area negligibly (0.1186 -> 0.1182);
# on a noisy-CD31 section it removes thousands of single-pixel specks.
MIN_SPECK_PX = int(round(8.0 / UM_PER_PX ** 2))


def top_q_mask(cd31, tissue, q):
    """The top-q% of CD31 intensity within tissue, speck-cleaned.

    This is THE vessel definition for the percentile method - enrichment,
    coverage and every percentile visual (visualize_sections re-exports it) go
    through it, so the number and the picture always use the same mask.
    """
    cut = np.percentile(cd31[tissue], 100 - q)
    return remove_small_objects((cd31 >= cut) & tissue, min_size=MIN_SPECK_PX)


def enrichment_curve(virus, cd31, tissue):
    """enrichment(q) for each q in PERCENTILES, on one section.

    Guards against ties at the cutoff: a plateau of repeated CD31 values (from
    quantised or saturated intensities) can make the top set select far more than
    q% of pixels, silently collapsing several q's onto the same selection and
    faking "stability across q". Any q whose achieved fraction is more than 1.5x
    the nominal is returned as nan rather than a misleading value.
    """
    total = int(tissue.sum())
    out = {}
    for q in PERCENTILES:
        vessels = top_q_mask(cd31, tissue, q)
        achieved = vessels.sum() / total if total else 0.0
        high, low = virus[vessels], virus[tissue & ~vessels]
        degenerate = achieved > 1.5 * (q / 100)      # a tie plateau blew up the top set
        out[q] = (float(high.mean() / low.mean())
                  if high.size and low.size and not degenerate else float("nan"))
    return out


def coverage_curve(virus, cd31, tissue):
    """coverage(q): fraction of the top-q% CD31 vessel area that is virus-positive.

    The area companion to enrichment, on the SAME threshold-free CD31 vessel
    geometry. Enrichment asks how much brighter the virus is on vessels; coverage
    asks how much of the vasculature it actually reaches. A direct vessel
    segmentation of the virus channel is not achievable (its off-vessel signal is
    genuine neuronal expression, structurally vessel-like), so the vessel geometry
    comes from CD31 and the virus only decides which of those vessels are labelled.

    "Virus-positive" is the pipeline's per-image rule (parenchyma median +
    VIRUS_K*MAD), so a per-section brightness gain does not move it. Reported per q
    because "vessel" is the top-q% of CD31; the companion off-vessel area fraction
    is dominated by the vast parenchyma and is deliberately not returned here.
    """
    out = {}
    for q in PERCENTILES:
        vessels = top_q_mask(cd31, tissue, q)
        parenchyma = tissue & ~vessels
        if not vessels.any() or not parenchyma.any():
            out[q] = float("nan")
            continue
        cut, _ = virus_cut(virus, parenchyma)
        out[q] = float(((virus > cut) & vessels).sum() / vessels.sum())
    return out


def main():
    full = "--full" in sys.argv
    paths = section_paths(full, slices_per_region=3)
    print(f"{len(paths)} sections ({'all clean mice' if full else 'pilot'})\n")

    rows = []
    for s in load_sections(paths):
        curve = enrichment_curve(s.virus, s.cd31, s.tissue)
        cover = coverage_curve(s.virus, s.cd31, s.tissue)
        rows.append({"figure": s.figure, "reporter": s.reporter, "mouse": s.mouse,
                     "region": s.region, "slice": s.slice_id,
                     **{f"enrich_top{q}": curve[q] for q in PERCENTILES},
                     **{f"cover_top{q}": cover[q] for q in PERCENTILES}})
        print(f"{s.counter} {s.figure} {s.reporter:5s} {s.mouse:7s} {s.region:2s}  "
              + "enrich " + " ".join(f"q{q}={curve[q]:.2f}" for q in PERCENTILES)
              + f"   cover@5%={cover[5]:.2f}")

    if not rows:
        sys.exit("no sections scored")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = write_csv(rows, out / ("enrichment_cd31_percentile_full.csv" if full
                                      else "enrichment_cd31_percentile_pilot.csv"))
    print(f"\nwrote {csv_path}")

    # sections[(figure, reporter, mouse, region, q)] -> list of per-slice values.
    sections = defaultdict(list)
    cover_sections = defaultdict(list)
    for r in rows:
        for q in PERCENTILES:
            key = (r["figure"], r["reporter"], r["mouse"], r["region"], q)
            sections[key].append(r[f"enrich_top{q}"])
            cover_sections[key].append(r[f"cover_top{q}"])

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

        def cover_mean(mouse, region, q):
            """Average coverage over slices for one mouse x region at percentile q."""
            vals = cover_sections.get((figure, reporter, mouse, region, q), [])
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
            # Only mice with a value in every compared region vote on the ordering.
            # A mouse missing a region would sort on a nan key, which leaves it in
            # place and can spuriously "agree" with the canonical region order,
            # inflating the count printed as evidence.
            voters = [m for m in mice
                      if all(not np.isnan(mouse_mean(m, x, q)) for x in regions)]
            agree = sum(1 for m in voters
                        if tuple(sorted(regions, key=lambda x: -mouse_mean(m, x, q))) == order)
            orderings.add(order)
            print(f"{q:6d}%" + "".join(f"{region_mean[x]:11.3f}" for x in regions)
                  + f"   {' > '.join(LONG[x][:4] for x in order):20s} {agree}/{len(voters)}")
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

        # Coverage: the area companion. How much of the CD31 vasculature the virus
        # actually reaches (virus-positive), on the same top-q% vessel definition.
        print("\n  coverage (virus-positive fraction of the top-q% CD31 vessels):")
        for q in (5, 10):
            per_region = {x: float(np.nanmean([cover_mean(m, x, q) for m in mice]))
                          for x in regions}
            print(f"    top-{q:2d}%: "
                  + "   ".join(f"{LONG[x]} {per_region[x]:.3f}" for x in regions))

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
    fig.suptitle("Vascular enrichment without segmentation - lines that stay lowest "
                 "across q are the robust result", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
