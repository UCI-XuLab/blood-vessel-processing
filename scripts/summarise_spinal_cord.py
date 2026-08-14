"""Aggregate the per-section measurements to a per-mouse regional comparison.

    python scripts/summarise_spinal_cord.py

The statistical point this script exists to enforce
---------------------------------------------------
There are three slices per mouse per region. They are three views of one animal,
not three independent samples of the population, so pooling them as n=9 per
region treats within-animal measurement noise as biological replication and
overstates significance by roughly the square root of the slice count.

So slices are averaged within each mouse x region first, and every comparison is
made on the mouse-level means. Each mouse contributes all three regions, which
makes the comparison naturally paired - and pairing is what removes the
between-animal variability that the experimenter has already flagged as large.

With three mice a p-value is not worth much: the smallest achievable two-sided
paired p at n=3 is 0.25 by sign alone. What is worth reporting at this n is the
per-mouse effect and whether its direction is consistent across animals. This
script therefore reports each mouse's value, the paired differences, and how many
animals follow the expected ordering, rather than manufacturing a significance
claim the design cannot support.

Reporters are kept apart: SYFP2 (Fig 1) and tdTomato (Fig 2) are different
constructs, and pooling them would confound reporter with mouse.
"""

import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def results_path():
    """CSV to summarise: the one named on the command line, else the newest."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    candidates = sorted(RESULTS_DIR.glob("spinal_cord_specificity*.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"no results in {RESULTS_DIR}; run scripts/analyse_spinal_cord.py first")
    return candidates[0]


MEASURES = ("enrichment", "coverage", "off_target", "vessel_area_fraction")
ORDER = ("C", "T", "L")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}


def load(path):
    if not path.exists():
        sys.exit(f"no results at {path}; run scripts/analyse_spinal_cord.py first")
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in MEASURES:
            row[key] = float(row[key])
    return rows


def mouse_region_means(rows):
    """Average slices within each mouse x region - the unit of replication."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["reporter"], row["mouse"], row["region"])].append(row)
    means = {}
    for key, group in grouped.items():
        means[key] = {m: float(np.mean([r[m] for r in group])) for m in MEASURES}
        means[key]["n_slices"] = len(group)
    return means


def main():
    path = results_path()
    print(f"source: {path.name}\n")
    rows = load(path)
    means = mouse_region_means(rows)
    reporters = sorted({key[0] for key in means})

    for reporter in reporters:
        mice = sorted({key[1] for key in means if key[0] == reporter})
        regions = [r for r in ORDER if any(k[0] == reporter and k[2] == r for k in means)]
        print("=" * 78)
        print(f"reporter: {reporter}    mice: {', '.join(mice)}    "
              f"regions: {', '.join(regions)}")
        print("=" * 78)

        for measure in MEASURES:
            print(f"\n{measure}")
            header = "  " + f"{'mouse':8s}" + "".join(f"{LONG[r]:>13s}" for r in regions)
            print(header)
            table = {}
            for mouse in mice:
                cells = []
                for region in regions:
                    entry = means.get((reporter, mouse, region))
                    value = entry[measure] if entry else np.nan
                    table[(mouse, region)] = value
                    cells.append("     -   " if np.isnan(value) else f"{value:13.4f}")
                print(f"  {mouse:8s}" + "".join(cells))

            complete = [m for m in mice
                        if all(not np.isnan(table[(m, r)]) for r in regions)]
            if len(complete) < 2 or len(regions) < 2:
                continue

            print(f"  {'mean':8s}" + "".join(
                f"{np.mean([table[(m, r)] for m in complete]):13.4f}" for r in regions))
            print(f"\n  paired differences across the {len(complete)} mice with all regions:")
            for a, b in itertools.combinations(regions, 2):
                diffs = np.array([table[(m, a)] - table[(m, b)] for m in complete])
                same = int(np.sum(np.sign(diffs) == np.sign(diffs[0])))
                direction = f"{LONG[a]} > {LONG[b]}" if diffs.mean() > 0 else f"{LONG[b]} > {LONG[a]}"
                print(f"    {LONG[a]:>10s} - {LONG[b]:<10s} "
                      f"mean {diffs.mean():+8.4f}   per mouse "
                      f"[{', '.join(f'{d:+.3f}' for d in diffs)}]   "
                      f"{same}/{len(diffs)} agree -> {direction}")
        print()

    print("=" * 78)
    print("reading these numbers")
    print("=" * 78)
    print("  enrichment is the primary measure: threshold-free, so it cannot be")
    print("  moved by a tuning choice. coverage and off_target both depend on the")
    print("  virus-positive cut and should be read alongside the sensitivity sweep.")
    print()
    print("  with three mice, direction consistency across animals is the evidence.")
    print("  a paired test at n=3 cannot reach p<0.25 two-sided on signs alone, so")
    print("  no p-value is quoted here; more animals is the only fix.")


if __name__ == "__main__":
    main()
