"""Interactive tuning viewer for the vessel segmentation pipeline.

Every parameter in this project is tuned by editing a constant, running a batch
over tens of images and reading a CSV, so the visual consequence of a setting is
never on screen next to the setting. This module puts it there.

Two halves. Below, pure functions over arrays - maskable, cacheable, testable
without Qt. Above, napari layers and a magicgui panel over the top. `napari` and
`magicgui` are imported inside the functions that need them, so importing this
module costs nothing and the pure half tests without a display server.

A dataset is a PRESET - a dict of numbers plus two enum choices - never a code
path. See docs/design/2026-08-14-vessel-tuning-gui.md.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from vessel_utils.threshold import segment
from vessel_utils.vesselness import jerman_vesselness

__all__ = ["normalise", "reference_lambda", "stages", "MASK_METHODS", "tissue_mask",
           "find_images", "channel_count", "read_spacing", "read_channels", "readout",
           "PRESETS", "load", "response", "selftest",
           "LAYERS", "build_viewer", "main"]


def normalise(channel, roi):
    """Put a channel on a common intensity scale before filtering.

    Necessary, not cosmetic. Without it the segmented vessel area fraction tracks
    staining contrast rather than vascular density: on the spinal-cord sections it
    spanned 57x across ten sections, collapsing to 1.5x once normalised. Fixing
    the threshold dataset-wide cannot substitute - a common criterion applied to
    inputs on different scales is still a different criterion.

    The median goes to 0 and the 99th percentile to 1, both measured inside `roi`
    only. Clipped below, not above: vessels are the bright tail and must keep
    their contrast.

    Same rule as `scripts/analyse_spinal_cord.normalise_for_segmentation`,
    reimplemented rather than imported because a package must not import from
    `scripts/`. If one changes, change both.
    """
    channel = np.asarray(channel, dtype=np.float32)
    low, high = np.percentile(channel[np.asarray(roi, dtype=bool)], [50, 99])
    return np.clip((channel - low) / max(high - low, 1e-6), 0, None).astype(np.float32)


from vessel_utils.vesselness import max_eigenvalue


def reference_lambda(images, spacing, sigmas, masks=None, percentile=99.9):
    """One tau reference for a whole dataset, not one per image.

    This is the point of `jerman_vesselness(reference_lambda=...)`. Computed per
    image, the response is regularised against that image's own maximum
    eigenvalue, so a fixed threshold means a different thing in every image - and
    the comparison a viewer exists to support is precisely across images. A
    brighter section would get a systematically different vessel criterion from a
    dimmer one, and any difference under test would be partly an artefact of the
    calibration.

    The median across the sample, not the mean, and a high quantile within each
    image rather than its maximum: the maximum is an extreme-value statistic set
    by one bright structure, and it varied fourfold between sections of the same
    spinal cord.

    Args:
        images: a representative sample. Sample across whatever the dataset varies
            in - region, animal, staining run - so the reference is not set by one
            anatomy.
        masks: optional per-image ROI. Background has no structure and only
            dilutes the quantile.
    """
    if not len(images):
        raise ValueError("no images to calibrate from")
    masks = masks if masks is not None else [None] * len(images)
    values = []
    for image, mask in zip(images, masks):
        roi = np.ones(np.shape(image), dtype=bool) if mask is None else \
            np.asarray(mask, dtype=bool)
        values.append(max_eigenvalue(normalise(image, roi), list(sigmas), spacing,
                                     percentile=percentile, mask=roi))
    return float(np.median(values))


def _min_size_pixels(min_vessel_um2, spacing):
    """Physical area to a pixel count. One voxel minimum, never zero."""
    return max(1, int(round(min_vessel_um2 / float(np.prod(spacing)))))


def _response_of(channel, tissue, spacing, sigmas, reference):
    """The bounded Jerman response for one channel - the seconds-scale step.

    Kept separate from thresholding so a caller (the viewer) can compute it once,
    cache it, and re-threshold cheaply. Measured 7.9 s on a 2048^2 section against
    0.3 s to threshold, so bundling the two would make every slider drag pay the
    filter again.
    """
    return jerman_vesselness(normalise(channel, tissue), list(sigmas), spacing,
                             reference_lambda=reference)


def _mask_of(response, tissue, low, high, min_size):
    """Threshold a response into a cleaned vessel mask - the ~100 ms step."""
    return segment(response, low=low, high=high, roi=tissue, min_size=min_size,
                   area_threshold=0, closing_radius=1)


def _top_q(channel, tissue, q, min_size):
    """The top q% of intensity inside tissue - the percentile vessel definition.

    No vesselness and no shape assumption, which is why it is worth showing beside
    the filtered masks: it is the spinal-cord project's primary measure.
    """
    from skimage.morphology import remove_small_objects
    tissue = np.asarray(tissue, dtype=bool)
    cut = np.percentile(np.asarray(channel)[tissue], 100.0 - q)
    return remove_small_objects((np.asarray(channel) >= cut) & tissue, min_size=min_size)


def stages(ref, test, spacing, *, tissue, sigmas, reference, ref_low, ref_high,
           test_low, test_high, min_vessel_um2, virus_k, q,
           ref_response=None, test_response=None):
    """Every intermediate array, so a viewer can show them all as layers.

    `test` may be None for single-channel input, in which case every `test_*` and
    `cut` entry is None. `reference` is the dataset-wide `reference_lambda` and is
    never None here - resolve it before calling.

    `ref_response`/`test_response` let a caller inject an already-computed (and
    cached) vesselness so a threshold change re-thresholds without paying for the
    filter again. None means compute it here, which is the test and batch path.
    When injected, `sigmas` and `reference` no longer affect the result - the
    caller is responsible for having computed the response at those values.
    """
    tissue = np.asarray(tissue, dtype=bool)
    min_size = _min_size_pixels(min_vessel_um2, spacing)

    if ref_response is None:
        ref_response = _response_of(ref, tissue, spacing, sigmas, reference)
    out = {"tissue": tissue, "ref_response": ref_response,
           "ref_vessels": _mask_of(ref_response, tissue, ref_low, ref_high, min_size),
           "ref_top_q": _top_q(ref, tissue, q, min_size),
           "test_response": None, "test_vessels": None,
           "test_positive": None, "cut": None}
    if test is None:
        return out

    if test_response is None:
        test_response = _response_of(test, tissue, spacing, sigmas, reference)
    out["test_response"] = test_response
    out["test_vessels"] = _mask_of(test_response, tissue, test_low, test_high, min_size)

    # Intensity definition of "positive", calibrated on this image's own
    # non-vessel tissue: median + k*MAD. Same rule as
    # scripts/analyse_spinal_cord.virus_cut - per-image by design, because
    # brightness was adjusted per image during acquisition and a fixed absolute
    # cut would confound acquisition settings with biology.
    background_values = np.asarray(test, dtype=np.float32)[tissue & ~out["ref_vessels"]]
    if background_values.size:
        background = float(np.median(background_values))
        mad = float(1.4826 * np.median(np.abs(background_values - background)))
        out["cut"] = background + virus_k * mad
        out["test_positive"] = (np.asarray(test) > out["cut"]) & tissue
    else:
        out["test_positive"] = np.zeros_like(tissue)
    out["test_intensity"] = np.asarray(test, dtype=np.float32)
    return out


MASK_METHODS = ("none", "otsu", "grabcut", "brain")


def tissue_mask(channels, method):
    """Tissue silhouette, by one of four existing maskers.

    Runs on the channel sum: tissue is whatever is bright in *either* channel, so
    a region dim in one still masks.

    none     everything. For an image that is already cropped to tissue.
    otsu     `threshold.otsu_threshold`. Fast and dtype-agnostic, so it is the
             default for an unknown dataset - GrabCut at 7-15 s per image would
             stall every step through a file list with no warm cache.
    grabcut  the vendored entropy-guided GrabCut. Slow but hugs the true edge,
             keeps torn fragments, and rejects background haze via a local-entropy
             seed. This is what produced the shipped spinal-cord segmentations.
    brain    the archive's Triangle threshold. 8-BIT ONLY - cv2.threshold does not
             accept uint16 or float.
    """
    if method not in MASK_METHODS:
        raise ValueError(f"unknown tissue-mask method {method!r}; "
                         f"expected one of {MASK_METHODS}")
    total = np.zeros(np.shape(channels[0]), dtype=np.float32)
    for channel in channels:
        total += np.asarray(channel, dtype=np.float32)

    if method == "none":
        return np.ones(total.shape, dtype=bool)

    if method == "otsu":
        from vessel_utils.threshold import otsu_threshold
        return total > otsu_threshold(total)

    if method == "grabcut":
        # Vendored from UCI-XuLab-RegTools; do not edit it here.
        from vessel_utils._vendor import EntropyGrabCutConfig, compute_entropy_grabcut
        mask = compute_entropy_grabcut(np.ascontiguousarray(total), polarity="bright",
                                       config=EntropyGrabCutConfig()).mask
        if not mask.any():
            raise ValueError("empty tissue mask: GrabCut found no foreground "
                             "(near-blank image)")
        return np.asarray(mask, dtype=bool)

    # brain
    if any(np.asarray(c).dtype != np.uint8 for c in channels):
        raise ValueError("the 'brain' mask needs 8-bit input: it calls cv2 "
                         "THRESH_TRIANGLE, which rejects uint16 and float. "
                         "Convert a copy first, or use 'otsu'.")
    from velazquez_rivera_2025.vessels import get_brain_mask   # frozen; read-only
    return np.asarray(get_brain_mask(total.astype(np.uint8)), dtype=bool)


IMAGE_SUFFIXES = (".tif", ".tiff")


def find_images(directory):
    """Every TIFF in a directory, sorted. Case-insensitive on the suffix."""
    directory = Path(directory)
    paths = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no TIFF images in {directory} "
                         f"(looked for {', '.join(IMAGE_SUFFIXES)})")
    return paths


def channel_count(path):
    """Channels, from the header, without decoding the image.

    Every non-spatial axis multiplies in, rather than keying on the axis being
    labelled "C": tifffile's own heuristic for an untagged leading dimension
    varies by version (observed "Q" for two planes, "S"/RGB for three, on
    tifffile 2026.7.31), so a literal `axes.startswith("C")` silently reads 1
    channel for exactly the multi-channel files this guard exists to catch.
    """
    import tifffile
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        count = 1
        for axis, size in zip(series.axes, series.shape):
            if axis not in ("Y", "X"):
                count *= int(size)
        return count


def read_spacing(path, ndim=2):
    """Physical pixel size in micrometres from the TIFF tags, or None.

    Returned as a per-axis tuple, never a scalar, so the same call works for a
    volume. None means the file does not say - and the caller must surface that
    rather than assume 1.0: with sigmas in micrometres, a wrong spacing silently
    searches the wrong vessel calibres.
    """
    import tifffile
    # TIFF ResolutionUnit: 1 = none, 2 = inch, 3 = centimetre. Value is the length
    # of one unit in micrometres, so um/px = unit_um / (pixels-per-unit).
    to_um = {1: None, 2: 25400.0, 3: 10000.0}
    with tifffile.TiffFile(path) as handle:
        page = handle.pages[0]
        tags = page.tags
        if "XResolution" not in tags or "YResolution" not in tags:
            return None
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
        scale = to_um.get(int(unit))
        if scale is None:
            return None
        out = []
        for name in ("YResolution", "XResolution"):
            numerator, denominator = tags[name].value
            if not numerator:
                return None
            out.append(scale * denominator / numerator)
    return tuple(out[-ndim:]) if ndim <= 2 else (out[0],) * (ndim - 2) + tuple(out)


def read_channels(path, roles):
    """The reference and test channels named by `roles`, as float32.

    `roles` is `(reference_index, test_index)`; `test_index` may be None for
    single-channel work.

    Raises rather than re-indexing when the file cannot supply a role. This is
    deliberate and load-bearing: `scripts/analyse_spinal_cord.curated_paths`
    excludes files with more than two channels outright rather than taking the
    first two, because "the extra-channel files have not been curated yet, so
    which two are green and CD31 is not established, and guessing would silently
    compare the wrong pair". Falling back to [0]/[1] here would produce a
    confident, wrong Dice instead of a visible error.
    """
    import tifffile
    count = channel_count(path)
    wanted = [i for i in roles if i is not None]
    # count > 2 outright, matching curated_paths: a file with an extra,
    # uncurated channel is refused even when the requested indices are in
    # range, because which two of three-plus are reference/test is not
    # guessable from the header.
    if count > 2 or any(i >= count for i in wanted):
        plural = "channel" if count == 1 else "channels"
        raise ValueError(
            f"{Path(path).name} has {count} {plural}, so role indices {tuple(wanted)} "
            f"cannot be read. Channel layout is not guessable - pick indices this "
            f"file actually has, or exclude it.")

    stack = tifffile.imread(path)
    def plane(index):
        return np.asarray(stack if count == 1 else stack[index], dtype=np.float32)
    return plane(roles[0]), (None if roles[1] is None else plane(roles[1]))


TIE_TOLERANCE = 1.5     # achieved/nominal above this means the cutoff had ties


def _ratio(values_in, values_out):
    """Mean inside over mean outside, or None if either side is empty."""
    if not values_in.size or not values_out.size:
        return None
    outside = float(values_out.mean())
    return None if outside == 0 else float(values_in.mean() / outside)


def readout(st, spacing, *, q, plausible=(0.0, 1.0), include_cl_dice=False):
    """The numbers that go beside the picture.

    Argument order for the asymmetric metrics is fixed: `precision(test, ref)`
    asks how much of the TEST mask sits on a reference vessel, `recall(test, ref)`
    how much of the reference the test covers. Swapping them silently reports the
    other question.

    `cl_dice` is optional because it skeletonises, which is the one metric here
    that is not free at slider rates.
    """
    from vessel_utils import metrics

    tissue, ref_vessels = st["tissue"], st["ref_vessels"]
    out = {key: None for key in
           ("test_af", "dice", "jaccard", "precision", "recall", "cl_dice",
            "enrichment", "coverage", "off_target", "enrichment_q")}
    out["ref_af"] = metrics.area_fraction(ref_vessels, tissue)
    out["cut"] = st["cut"]
    warnings = []

    low, high = plausible
    if not low <= out["ref_af"] <= high:
        warnings.append(
            f"ref_af {out['ref_af']:.4f} is outside the plausible band "
            f"({low}-{high}) - check the response is graded, not saturated")

    test_vessels = st["test_vessels"]
    if test_vessels is not None:
        out["test_af"] = metrics.area_fraction(test_vessels, tissue)
        out["dice"] = metrics.dice(test_vessels, ref_vessels)
        out["jaccard"] = metrics.jaccard(test_vessels, ref_vessels)
        out["precision"] = metrics.precision(test_vessels, ref_vessels)
        out["recall"] = metrics.recall(test_vessels, ref_vessels)
        if include_cl_dice:
            out["cl_dice"] = metrics.cl_dice(test_vessels, ref_vessels)

    return _add_intensity_measures(out, st, q, warnings)


def _add_intensity_measures(out, st, q, warnings):
    """Enrichment, coverage, off-target, and the percentile enrichment."""
    tissue, ref_vessels = st["tissue"], st["ref_vessels"]
    test_positive = st["test_positive"]
    if test_positive is None:
        out["warnings"] = warnings
        return out

    test = st.get("test_intensity")
    if test is not None:
        parenchyma = tissue & ~ref_vessels
        out["enrichment"] = _ratio(test[ref_vessels], test[parenchyma])
        # Nominal q against what was actually selected. A plateau of repeated
        # values at the cutoff selects far more than q%, which would otherwise
        # read as a real number.
        top_q, selected = st["ref_top_q"], float(st["ref_top_q"].sum())
        achieved = selected / max(float(tissue.sum()), 1.0)
        if achieved <= TIE_TOLERANCE * (q / 100.0):
            out["enrichment_q"] = _ratio(test[top_q], test[tissue & ~top_q])
        else:
            warnings.append(
                f"top-{q:g}% selected {achieved * 100:.1f}% of tissue - ties at the "
                f"cutoff, so enrichment_q is not reported")

    vessel_area = float(ref_vessels.sum())
    positive_area = float(test_positive.sum())
    if vessel_area:
        out["coverage"] = float((test_positive & ref_vessels).sum() / vessel_area)
    if positive_area:
        out["off_target"] = float((test_positive & ~ref_vessels).sum() / positive_area)
    out["warnings"] = warnings
    return out


SIGMAS = (1.5, 3.0, 6.0, 12.0)      # um, capillary through venule radius

# A preset is numbers plus two enum choices, never a code path. `spacing=None`
# means read the TIFF tags; `reference=None` means calibrate over a sample of the
# loaded directory once, then leave it editable.
PRESETS = {
    "generic": dict(
        spacing=None, roles=(1, 0), mask="otsu", reference=None,
        ref_thr=(0.03, 0.09), test_thr=(0.03, 0.09), min_vessel_um2=6.0,
        virus_k=3.0, q=10.0, plausible=(0.0, 1.0)),
    # The operating point behind the shipped contour JPGs and handoff mask TIFs.
    "spinal-cord shipped": dict(
        spacing=0.650193, roles=(1, 0), mask="grabcut", reference=2.0,
        ref_thr=(0.03, 0.09), test_thr=(0.04, 0.12), min_vessel_um2=6.0,
        virus_k=3.0, q=10.0, plausible=(0.01, 0.10)),
    # Superseded, and kept so its ~43%-of-tissue mask can be looked at: its
    # calibrated reference comes out near 0.5, which saturates the Jerman response
    # so the "vessel" mask is mostly grey matter.
    "spinal-cord superseded": dict(
        spacing=0.650193, roles=(1, 0), mask="grabcut", reference=None,
        ref_thr=(0.02, 0.15), test_thr=(0.02, 0.15), min_vessel_um2=6.0,
        virus_k=3.0, q=10.0, plausible=(0.01, 0.10)),
    # Reproduces NO published figure - the brain-slice figures came from the
    # archive pipeline, which this viewer deliberately does not run.
    "brain-slice starting point": dict(
        spacing=1.0, roles=(1, 0), mask="brain", reference=None,
        ref_thr=(0.03, 0.09), test_thr=(0.03, 0.09), min_vessel_um2=6.0,
        virus_k=3.0, q=10.0, plausible=(0.0, 1.0)),
}


@lru_cache(maxsize=8)
def load(path, mask_method, roles):
    """Channels and tissue mask for one file. Cached: masking is the slow step.

    All arguments must be hashable - `roles` is a tuple, not a list.
    """
    ref, test = read_channels(path, roles)
    channels = (ref,) if test is None else (ref, test)
    return channels, tissue_mask(channels, mask_method)


@lru_cache(maxsize=8)
def response(path, role_index, sigmas, spacing, reference, mask_method, roles):
    """Vesselness for one channel of one file, bounded in [0, 1].

    This is the seconds-scale step, and the only reason the threshold sliders can
    be live: everything downstream of it re-runs in about 100 ms. maxsize=8 holds
    two channels across four files.
    """
    channels, tissue = load(path, mask_method, roles)
    # Same computation _response_of does, keyed by (path, role) for the cache.
    return _response_of(channels[role_index], tissue, spacing, sigmas, reference)


def selftest():
    """Build every layer for a phantom and check the readout against metrics.

    Runs headless, so it is the check to reach for when the viewer misbehaves and
    the question is whether the pipeline or the Qt wiring is at fault.
    """
    from vessel_utils import metrics
    from vessel_utils.synth import phantom

    volume, truth, _ = phantom(shape=(16, 128, 128), spacing=(3.0, 0.75, 0.75), seed=3)
    index = int(np.argmax(truth.reshape(truth.shape[0], -1).sum(axis=1)))
    image, spacing = volume[index].astype(np.float32), (0.75, 0.75)
    tissue = tissue_mask([image], "none")
    reference = reference_lambda([image], spacing, SIGMAS[:3], masks=[tissue])
    assert reference > 0, "calibration produced a non-positive reference"

    st = stages(image, image, spacing, tissue=tissue, sigmas=SIGMAS[:3],
                reference=reference, ref_low=0.03, ref_high=0.09, test_low=0.03,
                test_high=0.09, min_vessel_um2=6.0, virus_k=3.0, q=10.0)
    for key in ("ref_response", "test_response", "ref_vessels", "test_vessels",
                "test_positive", "ref_top_q", "tissue"):
        assert st[key] is not None, f"{key} missing"
        assert st[key].shape == image.shape, f"{key} has the wrong shape"
    assert 0.0 <= st["ref_response"].min() <= st["ref_response"].max() <= 1.0

    got = readout(st, spacing, q=10.0, plausible=(0.0, 1.0), include_cl_dice=True)
    assert got["dice"] == metrics.dice(st["test_vessels"], st["ref_vessels"])
    assert got["dice"] > 0.99, "identical channels must agree"
    print(f"selftest OK - reference {reference:.3f}, "
          f"ref_af {got['ref_af']:.4f}, dice {got['dice']:.4f}")


# ---------------------------------------------------------------------------
# The interactive half. napari and magicgui are imported inside the functions
# below, never at module top level, so `import vessel_utils.gui` stays free of
# Qt and the pure half above tests without a display server.
# ---------------------------------------------------------------------------

LAYERS = [
    # (stage key, layer name, kind, visible, colormap)
    ("ref_image",     "reference", "image",  True,  "magenta"),
    ("test_image",    "test",      "image",  True,  "green"),
    ("ref_response",  "reference vesselness", "image", False, "turbo"),
    ("test_response", "test vesselness",      "image", False, "turbo"),
    ("tissue",        "tissue",    "labels", False, None),
    ("ref_vessels",   "reference vessels", "labels", True,  None),
    ("test_vessels",  "test vessels",      "labels", True,  None),
    ("test_positive", "test+ (intensity)", "labels", False, None),
    ("ref_top_q",     "reference top-q%",  "labels", False, None),
]


def _spacing_tuple(value, ndim=2):
    return (float(value),) * ndim if np.isscalar(value) else tuple(value)


def run_all(viewer, state, panel, preset):
    """Batch every file to a CSV. Placeholder until Task 8 wires the real one."""
    # ponytail: Task 8 replaces this stub with the actual batch-to-CSV run.
    print("batch not yet wired")


def build_viewer(directory, preset_name="generic", show=True):
    """Open the viewer on a directory of images. Returns the napari Viewer.

    `show=False` builds the viewer without showing the window - the only way to
    construct it headlessly, since Qt's `offscreen` platform cannot create an
    OpenGL context on Windows and vispy/napari require one. The real launch
    (`main`) uses the default `show=True`.
    """
    import napari
    from magicgui import magicgui

    paths = find_images(directory)
    preset = dict(PRESETS[preset_name])
    state = {"paths": paths, "preset": preset_name, "reference": preset["reference"],
             "spacing": None, "stages": None}
    viewer = napari.Viewer(title=f"bvp-tune - {Path(directory).name}", show=show)

    def labelled(path):
        """Combobox label, flagging files the current roles cannot be read from."""
        try:
            count = channel_count(path)
        except Exception:                                   # noqa: BLE001
            return f"{path.name}  [unreadable]"
        wanted = [i for i in preset["roles"] if i is not None]
        bad = " [channels: %d - roles %s unavailable]" % (count, tuple(wanted)) \
            if (count > 2 or any(i >= count for i in wanted)) else ""
        return f"{path.name}{bad}"

    def resolve_spacing(path):
        if preset["spacing"] is not None:
            return _spacing_tuple(preset["spacing"])
        from_tags = read_spacing(path)
        if from_tags is None:
            state["spacing_unknown"] = True
            return (1.0, 1.0)
        state["spacing_unknown"] = False
        return from_tags

    def calibrate(spacing, sigmas, mask_method, roles, sample=6):
        """One reference_lambda for the whole directory, never per image."""
        step = max(1, len(paths) // sample)
        images, masks = [], []
        for path in paths[::step][:sample]:
            try:
                channels, tissue = load(path, mask_method, roles)
            except Exception as error:                      # noqa: BLE001
                print(f"  calibration skipped {path.name}: {error}")
                continue
            images.append(channels[0])
            masks.append(tissue)
        if not images:
            raise ValueError("no image in the calibration sample had usable tissue")
        return reference_lambda(images, spacing, sigmas, masks=masks)

    @magicgui(
        auto_call=True,
        call_button="Recompute (spacing / sigmas / reference)",
        file={"choices": paths, "label": "file"},
        mask={"choices": MASK_METHODS, "label": "tissue mask"},
        sigmas={"label": "sigmas (um)"},
        reference={"label": "reference_lambda", "step": 0.05},
        ref_low={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.005},
        ref_high={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.005},
        test_low={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.005},
        test_high={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "step": 0.005},
        virus_k={"widget_type": "FloatSlider", "min": 0.0, "max": 10.0, "step": 0.1},
        q={"widget_type": "FloatSlider", "min": 0.5, "max": 50.0, "step": 0.5},
    )
    def panel(file: Path = paths[0],
              mask: str = preset["mask"],
              sigmas: str = ", ".join(str(s) for s in SIGMAS),
              reference: float = float(preset["reference"] or 0.0),
              ref_low: float = preset["ref_thr"][0],
              ref_high: float = preset["ref_thr"][1],
              test_low: float = preset["test_thr"][0],
              test_high: float = preset["test_thr"][1],
              min_vessel_um2: float = preset["min_vessel_um2"],
              virus_k: float = preset["virus_k"],
              q: float = preset["q"],
              cl_dice: bool = False):
        refresh(locals())

    def refresh(values):
        roles = preset["roles"]
        spacing = resolve_spacing(values["file"])
        sigmas = tuple(float(s) for s in values["sigmas"].replace(",", " ").split())
        reference = values["reference"] or None
        try:
            if reference is None:
                reference = calibrate(spacing, sigmas, values["mask"], roles)
                panel.reference.value = reference
            channels, tissue = load(values["file"], values["mask"], roles)
            # Compute the vesselness through the cache, so a threshold-only change
            # is a cache hit (instant) and stages re-thresholds in ~0.3s rather
            # than recomputing the ~8s filter. sigmas/reference changes miss the
            # cache by construction, which is why they sit behind Recompute.
            ref_response = response(values["file"], 0, sigmas, spacing, reference,
                                    values["mask"], roles)
            test_response = (response(values["file"], 1, sigmas, spacing, reference,
                                      values["mask"], roles)
                             if len(channels) > 1 else None)
            st = stages(channels[0], channels[1] if len(channels) > 1 else None,
                        spacing, tissue=tissue, sigmas=sigmas, reference=reference,
                        ref_low=values["ref_low"], ref_high=values["ref_high"],
                        test_low=values["test_low"], test_high=values["test_high"],
                        min_vessel_um2=values["min_vessel_um2"],
                        virus_k=values["virus_k"], q=values["q"],
                        ref_response=ref_response, test_response=test_response)
        except Exception as error:                          # noqa: BLE001
            # Leave the previous layers up. Selecting a bad file must not blank
            # the display - the batch scripts print SKIP and carry on, same idea.
            report.value = f"{Path(values['file']).name}\n\nFAILED: {error}"
            return

        st = dict(st, ref_image=channels[0],
                  test_image=channels[1] if len(channels) > 1 else None)
        draw(st, spacing)
        numbers = readout(st, spacing, q=values["q"], plausible=preset["plausible"],
                          include_cl_dice=values["cl_dice"])
        report.value = format_report(values["file"], spacing, reference, numbers)
        state["stages"] = st

    def draw(st, spacing):
        for key, name, kind, visible, colormap in LAYERS:
            data = st.get(key)
            if data is None:
                if name in viewer.layers:
                    viewer.layers.remove(name)
                continue
            if name in viewer.layers:
                viewer.layers[name].data = data
                if "response" in key:
                    # Re-pin on every update: the response window must stay 0-1
                    # so Jerman saturation stays visible, never rescaled to data.
                    viewer.layers[name].contrast_limits = (0.0, 1.0)
                continue
            if kind == "image":
                # Response clims are PINNED to 0-1, never auto-scaled: Jerman
                # saturates above tau*reference_lambda/2, and auto-scaling would
                # hide exactly that.
                extra = {"contrast_limits": (0.0, 1.0)} if "response" in key else {}
                viewer.add_image(data, name=name, colormap=colormap,
                                 blending="additive", scale=spacing, **extra)
            else:
                viewer.add_labels(np.asarray(data, dtype=np.uint8), name=name,
                                  opacity=0.4, scale=spacing)
            viewer.layers[name].visible = visible

    def format_report(path, spacing, reference, numbers):
        lines = [str(Path(path).name),
                 f"spacing {spacing[0]:.4g} x {spacing[1]:.4g} um/px"
                 + ("  (NOT in the file - assumed)"
                    if state.get("spacing_unknown") else ""),
                 f"reference_lambda {reference:.4g}", ""]
        for key in ("ref_af", "test_af", "dice", "jaccard", "precision", "recall",
                    "cl_dice", "enrichment", "coverage", "off_target",
                    "enrichment_q", "cut"):
            value = numbers.get(key)
            lines.append(f"{key:14s} " + ("-" if value is None else f"{value:.4f}"))
        for warning in numbers["warnings"]:
            lines.append(f"\n! {warning}")
        return "\n".join(lines)

    from magicgui.widgets import Label, PushButton, TextEdit
    report = TextEdit(value="", label="")
    report.read_only = True
    batch = PushButton(text=f"Run all {len(paths)} -> CSV")
    batch.changed.connect(lambda: run_all(viewer, state, panel, preset))

    viewer.window.add_dock_widget(
        Label(value=f"<b>preset:</b> {preset_name}"), area="right", name="preset")
    viewer.window.add_dock_widget(panel, area="right", name="parameters")
    viewer.window.add_dock_widget(report, area="right", name="metrics")
    viewer.window.add_dock_widget(batch, area="right", name="batch")
    panel.file.choices = paths
    try:
        # Best-effort: reach through magicgui to Qt to hang a per-file tooltip on
        # the combobox. The load-bearing channel guard is read_channels raising;
        # this only annotates. Private path, so tolerate its absence.
        panel.file._widget._qwidget.setToolTip("\n".join(labelled(p) for p in paths))
    except Exception:                                       # noqa: BLE001
        pass
    refresh({name: panel[name].value for name in panel.__signature__.parameters})
    return viewer


def main(argv=None):
    """`bvp-tune [directory] [--preset NAME] [--selftest]`."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="bvp-tune", description="Interactive vessel-segmentation tuning viewer")
    parser.add_argument("directory", nargs="?", help="directory of TIFF images")
    parser.add_argument("--preset", default="generic", choices=sorted(PRESETS))
    parser.add_argument("--selftest", action="store_true",
                        help="run the headless pipeline check and exit")
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if not args.directory:
        parser.error("a directory is required (or use --selftest)")

    try:
        import napari
    except ImportError:
        print('napari is not installed. Run:  pip install -e ".[gui]"')
        return 1
    try:
        build_viewer(args.directory, args.preset)
    except ValueError as error:
        print(f"error: {error}")
        return 1
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
