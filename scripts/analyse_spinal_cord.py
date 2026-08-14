"""Vascular specificity of the enhancer AAV along the spinal cord axis.

Data: Z:\\Lab\\Eric V\\BEC Spinal Cords\\composites_EV  (read-only; never written to)
Two-channel composites of spinal cord cross-sections, per the dataset README:

    channel 1 = green   = virus reporter (SYFP2 or tdTomato)
    channel 2 = magenta = CD31 endothelial staining, the ground truth

Naming: Fig N_Mouse_Region_reporter_CD31-mag_sliceK.tif, region in {C, T, L, TL}.

Design note, and the main methodological choice here
----------------------------------------------------
The obvious approach - segment vessels in both channels and compare the masks -
is wrong for this question. Running a vesselness filter on the virus channel
pre-filters it to tubular shapes, which silently discards the off-target signal
(transduced neurons in grey matter) that should count *against* specificity. It
would measure "of the virus signal that looks like a vessel, how much is on a
vessel", which is close to a tautology and flatters the vector.

So vessels are defined from CD31 alone, and the virus channel is measured inside
and outside that mask without any shape assumption:

    enrichment  mean virus intensity in vessels / in non-vessel tissue.
                Threshold-free, so it does not depend on a tuning choice. This is
                the primary measure.
    coverage    fraction of CD31 vessel area that is virus-positive - how much of
                the vasculature the vector reaches.
    off_target  fraction of virus-positive area lying outside vessels - leak into
                parenchyma. Coverage and off_target move independently, and a
                vector can be good at one and bad at the other.

The virus threshold is set per image from the non-vessel tissue itself
(median + k*MAD), because the README notes brightness was adjusted per image; a
fixed absolute cut would confound acquisition settings with biology.

Artefact handling:
  - the tissue mask (see tissue_mask) is an entropy-guided GrabCut silhouette that
    excludes the black background and the rotated-composite corners
  - the bright pial edge is kept, not eroded: dropping a fixed rim also dropped real
    near-edge vessels, so the edge staining is accepted instead of a rim erosion
  - grey-matter background is *not* removed: it is the leak being measured, and
    it is why off_target is reported separately rather than folded into a score
"""

import hashlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vessel_utils import metrics                                   # noqa: E402
from vessel_utils._vendor import compute_entropy_grabcut, EntropyGrabCutConfig  # noqa: E402
from vessel_utils.sweep import write_csv                           # noqa: E402
from vessel_utils.threshold import segment                         # noqa: E402
from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue  # noqa: E402

DATA = Path(r"Z:\Lab\Eric V\BEC Spinal Cords\composites_EV")
UM_PER_PX = 0.650193          # from the calibrated files; identical in all 34 of them
SIGMAS = [1.5, 3.0, 6.0, 12.0]   # um, capillary through venule radius
# Hysteresis with a WIDE gap. The seed (high) sits on confident vessel; the grow
# (low) sits just above the noise floor. Measured across six SYFP2 sections by
# scripts/dim_recall_check.py: the old narrow gap (low = high*0.5) recovered a
# mean 0.19 of dim vessels, the wide gap 0.74 - because the wide gap follows dim
# vessel segments connected to bright ones. (Recall here is against a
# local-threshold visible-vessel proxy, not manual ground truth; read it as the
# difference between the two settings, not absolute accuracy.) The two thresholds
# do different jobs and are set independently, not as a ratio.
VESSEL_HIGH = 0.15            # seed: confident vessel in the Jerman response
VESSEL_LOW = 0.02            # grow: ~2x the response noise floor
MIN_VESSEL_UM2 = 6.0
VIRUS_K = 3.0                 # virus-positive = background median + k * MAD

NAME = re.compile(r"^(Fig \d\w*)_([\w\-]+)_(C|T|L|TL)_(.+?)_CD31-mag(?:_slice(\d+))?\.tif$")
REGION_NAME = {"C": "cervical", "T": "thoracic", "L": "lumbar", "TL": "thoracolumbar"}


# Tissue mask via the vendored entropy-guided GrabCut masker (see vessel_utils/_vendor).
# The default config is the upstream benchmark: percentile 1-99.5 normalize +
# Multi-Otsu-4 seed. Construct once and reuse.
_GRABCUT = EntropyGrabCutConfig()
_MASK_CACHE = Path(__file__).resolve().parent.parent / "results" / "tissue_masks"
_MASK_CACHE_VERSION = "grabcut-9f2e5fb"   # bump (or delete the dir) to invalidate


def tissue_mask(green, cd31):
    """Section silhouette from the vendored entropy-guided GrabCut masker.

    Tissue is whatever is bright in either channel, so the masker runs on the
    channel sum. Its seed uses a local-entropy channel, so it rejects the flat
    background haze a pure-intensity threshold leaks into, and it keeps every
    component >= 1% of the largest - so a section torn during histology stays whole
    rather than losing every piece but the biggest. It draws the true tissue edge
    INCLUDING the bright pial rim: the earlier 90 um rim erosion is gone by choice,
    since it also dropped real near-edge vessels, so the edge staining is accepted
    in exchange. This also removed the old triangle threshold's failure on sections
    whose bright outliers defeated it (e.g. Fig 2b M87 slice6, previously skipped),
    which GrabCut's percentile-normalised, entropy-guided seed handles.

    GrabCut is deterministic, so masks are cached on disk, content-addressed on the
    channel sum. Delete results/tissue_masks/ (or bump _MASK_CACHE_VERSION) to
    recompute - e.g. after re-syncing the vendored masker.
    """
    total = np.ascontiguousarray(
        np.asarray(green, np.float32) + np.asarray(cd31, np.float32))
    key = hashlib.blake2b(total.tobytes(), digest_size=16).hexdigest()
    cache_file = _MASK_CACHE / f"{_MASK_CACHE_VERSION}_{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    mask = compute_entropy_grabcut(total, polarity="bright", config=_GRABCUT).mask
    if not mask.any():
        raise ValueError(
            "empty tissue mask: GrabCut found no foreground (near-blank section)."
        )
    _MASK_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, mask)
    return mask


def calibrate_reference(paths, n_sections=6):
    """One tau reference for the whole dataset, not one per image.

    This is the point of reference_lambda. Computed per image, the vesselness
    response would be regularised against that image's own maximum eigenvalue,
    so a fixed threshold would mean a different thing in every section - and the
    comparison being made here is precisely across sections, regions and mice.
    A brighter cervical section would then get a systematically different vessel
    criterion from a dimmer thoracic one, and the regional difference under test
    would be partly an artefact of the calibration.

    Sampled across regions so the reference is not set by one anatomy.
    """
    chosen = paths[:: max(1, len(paths) // n_sections)][:n_sections]
    values = []
    for path in chosen:
        stack = tifffile.imread(path)
        green, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(green, cd31)
        except ValueError as error:
            # A section the tissue threshold cannot handle must not take the whole
            # calibration down with it; drop it and calibrate on the rest.
            print(f"   {path.name[:46]:46s} skipped ({error})")
            continue
        # Calibrate on the same normalised input the segmentation will see.
        normalised = normalise_for_segmentation(cd31, tissue)
        # A high quantile inside tissue, not the max: the max is set by one
        # bright structure and varied fourfold between sections of one cord.
        values.append(max_eigenvalue(normalised, SIGMAS, (UM_PER_PX, UM_PER_PX),
                                     percentile=99.9, mask=tissue))
        print(f"   {path.name[:46]:46s} lambda_max {values[-1]:9.1f}")
    if not values:
        raise ValueError("no section in the calibration sample had usable tissue")
    reference = float(np.median(values))
    print(f"   -> dataset reference_lambda = {reference:.1f} "
          f"(median of {len(values)}; spread {min(values):.0f}-{max(values):.0f})")
    return reference


def normalise_for_segmentation(channel, tissue):
    """Put each section's CD31 on a common intensity scale before filtering.

    Necessary, and the pilot showed why. With raw input the vessel area fraction
    ranged 0.0036 to 0.207 across ten sections - a 57x spread, when CNS vascular
    density varies by nothing like that. It tracked CD31 staining contrast
    (p99/p50) almost perfectly: brightly stained sections yielded more "vessels".
    Normalising each section to its own tissue percentiles collapses that spread
    to 1.5x.

    Note what is corrected and what is not. Staining and exposure are properties
    of the acquisition, so equalising them is legitimate. Fixing the *threshold*
    dataset-wide, which `reference_lambda` does, cannot substitute for this: a
    common criterion applied to inputs on different scales is still a different
    criterion. Normalise the input, then fix the threshold.

    Applied to CD31 only. The virus channel is left raw, because `enrichment` is
    a ratio of virus intensities and subtracting a per-section offset from it
    would change that ratio - the measurement would then depend on the
    correction.
    """
    low, high = np.percentile(channel[tissue], [50, 99])
    return np.clip((channel - low) / max(high - low, 1e-6), 0, None).astype(np.float32)


def vessel_mask(cd31, tissue, reference):
    """Vessels from CD31 alone - the ground-truth channel."""
    response = jerman_vesselness(normalise_for_segmentation(cd31, tissue),
                                 SIGMAS, (UM_PER_PX, UM_PER_PX),
                                 reference_lambda=reference)
    return segment(response, low=VESSEL_LOW, high=VESSEL_HIGH,
                   roi=tissue, min_size=int(round(MIN_VESSEL_UM2 / UM_PER_PX ** 2)),
                   area_threshold=0, closing_radius=1)


def analyse(path, reference):
    stack = tifffile.imread(path)
    # `!= 2`, matching load_sections: `< 2` would let a 3-channel file through
    # and guess that stack[0]/stack[1] are virus/CD31, which curated_paths
    # refuses to do precisely because that pairing is not established.
    if stack.ndim != 3 or stack.shape[0] != 2:
        raise ValueError(f"expected a 2-channel composite, got shape {stack.shape}")
    green = stack[0].astype(np.float32)
    cd31 = stack[1].astype(np.float32)

    tissue = tissue_mask(green, cd31)
    vessels = vessel_mask(cd31, tissue, reference)
    parenchyma = tissue & ~vessels

    if vessels.sum() == 0 or parenchyma.sum() == 0:
        raise ValueError("empty vessel or parenchyma mask")

    in_vessel = green[vessels]
    outside = green[parenchyma]

    # Threshold-free: how much brighter is the virus on vessels than off them.
    enrichment = float(in_vessel.mean() / outside.mean())

    # Virus-positive, calibrated against this image's own parenchyma.
    cut, background = virus_cut(outside)
    virus_positive = (green > cut) & tissue

    coverage = float((virus_positive & vessels).sum() / vessels.sum())
    off_target = float((virus_positive & ~vessels).sum() / max(virus_positive.sum(), 1))

    px_um2 = UM_PER_PX ** 2
    return {
        "tissue_mm2": tissue.sum() * px_um2 / 1e6,
        "vessel_area_fraction": float(vessels.sum() / tissue.sum()),
        "enrichment": enrichment,
        "coverage": coverage,
        "off_target": off_target,
        "dice_virus_vs_cd31": metrics.dice(virus_positive & tissue, vessels),
        "virus_area_fraction": float(virus_positive.sum() / tissue.sum()),
        "virus_cut": cut,
        "parenchyma_median": background,
    }


def channel_count(path):
    """Channels, read from the header without decoding the image."""
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        return series.shape[0] if series.axes.startswith("C") else 1


def curated_paths(pilot_mice=None, slices_per_region=None):
    """Two-channel Fig 1 / Fig 2 images, optionally cut down to a pilot subset.

    Files with more than two channels are excluded outright rather than having
    their first two taken: the extra-channel files have not been curated yet, so
    which two are green and CD31 is not established, and guessing would silently
    compare the wrong pair.
    """
    everything = sorted(p for p in DATA.glob("Fig*.tif")
                        if NAME.match(p.name)
                        and NAME.match(p.name).group(1).startswith(("Fig 1", "Fig 2")))

    paths, skipped = [], []
    for path in everything:
        count = channel_count(path)
        (paths if count == 2 else skipped).append((path, count))
    if skipped:
        for path, count in skipped:
            print(f"   excluded (has {count} channels, not 2): {path.name}")
    paths = [p for p, _ in paths]

    if pilot_mice is None and slices_per_region is None:
        return paths

    chosen, seen = [], defaultdict(int)
    mice_by_reporter = defaultdict(list)
    for path in paths:
        _, mouse, _, reporter, _ = NAME.match(path.name).groups()
        if mouse not in mice_by_reporter[reporter]:
            mice_by_reporter[reporter].append(mouse)
    keep = {m for reporter in mice_by_reporter
            for m in mice_by_reporter[reporter][:pilot_mice or len(paths)]}

    for path in paths:
        _, mouse, region, _, _ = NAME.match(path.name).groups()
        if mouse not in keep:
            continue
        if slices_per_region and seen[(mouse, region)] >= slices_per_region:
            continue
        seen[(mouse, region)] += 1
        chosen.append(path)
    return chosen


# --------------------------------------------------------------------------
# shared section loading
#
# Every analysis and figure script needs the same preamble: pick the pilot or
# full path list, parse the filename, check the file really is a two-channel
# composite, cast to float32, build the tissue mask, and skip-with-a-reason
# anything that fails. That block was copied into eight scripts and drifted; it
# lives here now and a script's loop is `for s in load_sections(paths)`.
#
# Seven scripts use it. export_handoff.py deliberately does not: its `clean_stem`
# omits the `_s0` suffix that `Section.stem` always emits, so switching it would
# rename every exported mask TIF that its README references.
# --------------------------------------------------------------------------

def short_reporter(reporter):
    """Filename reporter field to its display form: 'SYFP2-green' -> 'SYFP2'."""
    return "SYFP2" if "SYFP2" in reporter else "tdT" if "tdT" in reporter else reporter


# eq=False, so the generated __eq__/__hash__ over the array fields are not
# built: hashing one would raise on the ndarrays, and comparing two would raise
# "truth value of an array is ambiguous". Identity semantics are what a loaded
# section wants anyway, and they let a Section be used as a dict key.
@dataclass(frozen=True, eq=False)
class Section:
    """One loaded, tissue-masked section, with its identifiers already parsed."""
    path: Path
    index: int
    total: int
    figure: str
    mouse: str
    region: str
    reporter: str        # display form, from short_reporter
    slice_id: str        # "" when the filename carries no slice number
    virus: np.ndarray
    cd31: np.ndarray
    tissue: np.ndarray

    @property
    def label(self):
        """Human-readable, e.g. 'Fig 1 M131 cervical s3'."""
        return (f"{self.figure} {self.mouse} {REGION_NAME[self.region]}"
                + (f" s{self.slice_id}" if self.slice_id else ""))

    @property
    def stem(self):
        """Gallery filename stem, e.g. 'Fig1_M131_C_SYFP2_s3'."""
        return (f"{self.figure.replace(' ', '')}_{self.mouse}_{self.region}"
                f"_{self.reporter}_s{self.slice_id or '0'}")

    @property
    def counter(self):
        """'[ 7/39]' - the progress prefix every script prints."""
        return f"[{self.index:2d}/{self.total}]"


def section_paths(full=False, slices_per_region=1, pilot_mice=2):
    """Every curated section when `full`, else a pilot subset of them.

    **`full=True` ignores `slices_per_region` and `pilot_mice`** - --full means
    every section, and the subset arguments describe only the pilot. Callers
    pass both because `full` is a runtime flag, so read the signature as "the
    pilot is this shape" rather than as a filter applied in both modes.
    """
    return (curated_paths() if full
            else curated_paths(pilot_mice=pilot_mice,
                               slices_per_region=slices_per_region))


def load_sections(paths):
    """Yield each usable section, loaded and tissue-masked.

    A section that is not a two-channel composite, or whose tissue mask fails,
    is reported and skipped rather than taking the whole run down - the same
    behaviour each script implemented separately.
    """
    total = len(paths)
    for index, path in enumerate(paths, 1):
        match = NAME.match(path.name)
        if match is None:
            # Callers derive their list from curated_paths, which only yields
            # matching names - but this is the shared entry point now, and its
            # contract is to report and skip rather than raise mid-batch.
            print(f"[{index:2d}/{total}] SKIP {path.name}: filename does not parse")
            continue
        figure, mouse, region, reporter_raw, slice_id = match.groups()
        stack = tifffile.imread(path)
        if stack.ndim != 3 or stack.shape[0] != 2:
            print(f"[{index:2d}/{total}] SKIP {path.name}: not 2-channel")
            continue
        virus, cd31 = stack[0].astype(np.float32), stack[1].astype(np.float32)
        try:
            tissue = tissue_mask(virus, cd31)
        except ValueError as error:
            print(f"[{index:2d}/{total}] SKIP {path.name}: {error}")
            continue
        yield Section(path=path, index=index, total=total, figure=figure,
                      mouse=mouse, region=region,
                      reporter=short_reporter(reporter_raw),
                      slice_id=slice_id or "",
                      virus=virus, cd31=cd31, tissue=tissue)


def virus_cut(parenchyma_values, k=VIRUS_K):
    """Virus-positive threshold, calibrated on this image's own parenchyma.

    Per-image by design: the dataset README notes brightness was adjusted per
    image, so a fixed absolute cut would confound acquisition settings with
    biology. Returns (cut, background) - the background median is reported
    alongside the cut so a section's calibration is visible in the CSV.

    Takes the already-extracted 1-D sample, not the image and a mask: a caller
    that needs the parenchyma values for anything else (`analyse` needs their
    mean) would otherwise fancy-index a multi-megapixel array three times over.
    """
    background = float(np.median(parenchyma_values))
    mad = float(1.4826 * np.median(np.abs(parenchyma_values - background)))
    return background + k * mad, background


def main():
    pilot = "--full" not in sys.argv
    slices = 3 if "--slices3" in sys.argv else 1
    if pilot:
        paths = section_paths(slices_per_region=slices)
        print(f"PILOT: {len(paths)} images (2 mice per reporter, {slices} slice(s) "
              f"per region). --slices3 for three, --full for everything.\n")
    else:
        paths = section_paths(full=True)
        print(f"{len(paths)} curated two-channel images (Fig 1 and Fig 2)\n")
    print("calibrating one dataset-wide reference_lambda:")
    reference = calibrate_reference(paths, n_sections=min(6, len(paths)))
    print()

    rows = []
    for index, path in enumerate(paths, 1):
        figure, mouse, region, reporter, slice_id = NAME.match(path.name).groups()
        try:
            result = analyse(path, reference)
        except Exception as error:                       # noqa: BLE001
            print(f"[{index:2d}/{len(paths)}] SKIP {path.name}: {error}")
            continue
        row = {"figure": figure, "mouse": mouse, "region": region,
               "region_name": REGION_NAME[region], "reporter": reporter,
               "slice": slice_id or "", **result}
        rows.append(row)
        print(f"[{index:2d}/{len(paths)}] {mouse:6s} {region:2s} "
              f"enrich {result['enrichment']:5.2f}  cover {result['coverage']:5.3f}  "
              f"offtgt {result['off_target']:5.3f}  vessel_af {result['vessel_area_fraction']:.4f}")

    if not rows:
        sys.exit("no sections scored")
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    csv_path = write_csv(rows, out / (f"spinal_cord_specificity_pilot{slices}.csv"
                                      if pilot else "spinal_cord_specificity.csv"))
    print(f"\nwrote {csv_path}")

    print("\n=== by region (mean +/- sd across sections) ===")
    print(f"{'region':14s} {'n':>3s} {'enrichment':>18s} {'coverage':>16s} {'off_target':>16s}")
    for code in ("C", "T", "L", "TL"):
        group = [r for r in rows if r["region"] == code]
        if not group:
            continue
        def stat(key):
            values = np.array([r[key] for r in group])
            return f"{values.mean():.3f} +/- {values.std(ddof=1) if len(values) > 1 else 0:.3f}"
        print(f"{REGION_NAME[code]:14s} {len(group):3d} {stat('enrichment'):>18s} "
              f"{stat('coverage'):>16s} {stat('off_target'):>16s}")

    print("\n=== by mouse and region (enrichment) ===")
    mice = sorted({r["mouse"] for r in rows})
    print(f"{'mouse':8s} " + "".join(f"{REGION_NAME[c]:>16s}" for c in ("C", "T", "L")))
    for mouse in mice:
        cells = []
        for code in ("C", "T", "L"):
            group = [r["enrichment"] for r in rows
                     if r["mouse"] == mouse and r["region"] == code]
            cells.append(f"{np.mean(group):.2f} (n={len(group)})" if group else "-")
        print(f"{mouse:8s} " + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
