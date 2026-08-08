"""Statistics for the SYFP2 regional enrichment finding (cervical > thoracic > lumbar).

    python scripts/regional_stats.py            # reads results/enrichment_cd31_percentile_full.csv

The finding lives at the mouse level (slices are pseudoreplicates), so the test is
a paired one on per-mouse (lumbar - rostral) deltas: each mouse is one independent
replicate, the contrast is within-mouse. Reported alongside the Friedman test
across all three regions and the raw direction-consistency, because at n=3 no
single test is decisive on its own. tdT is n=1 per figure -> descriptive only.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CSV = Path(__file__).resolve().parent.parent / "results" / "enrichment_cd31_percentile_full.csv"
QS = (2, 5, 10, 20, 30)
HEADLINE_Q = 10
REGIONS = ("C", "T", "L")
LONG = {"C": "cervical", "T": "thoracic", "L": "lumbar"}


def mouse_region_means(rows, reporter, q):
    """{mouse: {region: mean enrichment at top-q%}} for one reporter, C/T/L only."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["reporter"] == reporter and r["region"] in REGIONS:
            acc[r["mouse"]][r["region"]].append(float(r[f"enrich_top{q}"]))
    return {m: {reg: float(np.mean(v)) for reg, v in regs.items()}
            for m, regs in acc.items()}


def paired_stats(by_mouse):
    """Lumbar vs rostral (mean of cervical+thoracic), paired across mice."""
    mice = [m for m, r in by_mouse.items() if REGIONS[0] in r and REGIONS[1] in r and "L" in r]
    deltas = np.array([by_mouse[m]["L"] - (by_mouse[m]["C"] + by_mouse[m]["T"]) / 2 for m in mice])
    rostral = np.array([(by_mouse[m]["C"] + by_mouse[m]["T"]) / 2 for m in mice])
    n = len(deltas)
    sd = float(deltas.std(ddof=1))
    mean, se = float(deltas.mean()), sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t_p = stats.ttest_1samp(deltas, 0.0)
    friedman = stats.friedmanchisquare(*([by_mouse[m][reg] for m in mice] for reg in REGIONS))
    return {
        "mice": mice, "deltas": deltas, "n": n, "mean": mean,
        "ci": (mean - tcrit * se, mean + tcrit * se),
        "pct": 100 * mean / float(rostral.mean()),
        "dz": mean / sd if sd else float("nan"),
        "t_p": float(t_p.pvalue), "neg": int((deltas < 0).sum()),
        "friedman_p": float(friedman.pvalue),
    }


def report(rows):
    out = []
    w = out.append
    by_head = mouse_region_means(rows, "SYFP2", HEADLINE_Q)
    s = paired_stats(by_head)

    w(f"SYFP2 regional enrichment - statistics (n={s['n']} mice: {', '.join(s['mice'])})\n")
    w(f"Per-mouse enrichment (top-{HEADLINE_Q}% CD31):")
    w(f"  {'mouse':8}{'cervical':>10}{'thoracic':>10}{'lumbar':>10}{'L-rostral':>11}")
    for m in s["mice"]:
        r = by_head[m]
        w(f"  {m:8}{r['C']:>10.3f}{r['T']:>10.3f}{r['L']:>10.3f}"
          f"{r['L'] - (r['C'] + r['T']) / 2:>+11.3f}")

    lo, hi = s["ci"]
    w(f"\nLumbar vs rostral (mean of cervical+thoracic), paired across mice:")
    w(f"  per-mouse deltas: {', '.join(f'{d:+.3f}' for d in s['deltas'])}  "
      f"({s['neg']}/{s['n']} negative)")
    w(f"  mean delta {s['mean']:+.3f}  (95% CI {lo:+.3f} .. {hi:+.3f})  "
      f"= {s['pct']:+.1f}% vs rostral")
    w(f"  paired t-test (mouse-level) p = {s['t_p']:.4f}   Cohen's dz = {s['dz']:.2f}")
    w(f"  Friedman across C/T/L        p = {s['friedman_p']:.4f}")

    w(f"\nOrdering consistency across the q-sweep (Friedman across C/T/L per q):")
    w(f"  {'q':>4}{'friedman p':>12}{'L lowest':>11}  ordering")
    for q in QS:
        bm = mouse_region_means(rows, "SYFP2", q)
        mice = list(bm)
        fp = float(stats.friedmanchisquare(*([bm[m][r] for m in mice] for r in REGIONS)).pvalue)
        l_lowest = sum(bm[m]["L"] == min(bm[m][r] for r in REGIONS) for m in mice)
        means = {r: np.mean([bm[m][r] for m in mice]) for r in REGIONS}
        order = " > ".join(LONG[r] for r in sorted(REGIONS, key=lambda r: -means[r]))
        w(f"  {q:>4}{fp:>12.4f}{f'{l_lowest}/{len(mice)}':>11}  {order}")

    w("\nNote: n=3 caps power. The paired mouse-level t-test is the primary test")
    w("(each mouse one replicate, within-mouse contrast); Friedman and the 3/3")
    w("consistency are the nonparametric backups. tdT is n=1 per figure (M63, M87):")
    w("no regional statistics; report descriptively and get more tdT animals.")
    return "\n".join(out)


def _selfcheck():
    """Consistent lumbar-lowest across 3 synthetic mice must test significant-ish."""
    vals = {"a": (1.60, 1.52, 1.34), "b": (1.70, 1.60, 1.38), "c": (1.55, 1.48, 1.31)}
    rows = [{"reporter": "SYFP2", "mouse": m, "region": reg, "enrich_top10": v}
            for m, ctl in vals.items() for reg, v in zip(REGIONS, ctl)]
    s = paired_stats(mouse_region_means(rows, "SYFP2", 10))
    assert s["neg"] == 3 and s["t_p"] < 0.05, s


if __name__ == "__main__":
    _selfcheck()
    if not CSV.exists():
        sys.exit(f"missing {CSV} — run enrichment_by_cd31_percentile.py --full first")
    text = report(list(csv.DictReader(open(CSV, newline="", encoding="utf-8"))))
    print(text)
    dst = CSV.parent / "regional_stats.txt"
    dst.write_text(text + "\n", encoding="utf-8")
    print(f"\nwrote {dst}")
