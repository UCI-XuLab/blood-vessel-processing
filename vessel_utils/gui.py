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

from pathlib import Path

import numpy as np

from vessel_utils.threshold import segment
from vessel_utils.vesselness import jerman_vesselness

__all__ = ["normalise", "reference_lambda", "stages", "MASK_METHODS", "tissue_mask",
           "find_images", "channel_count", "read_spacing", "read_channels", "readout",
           "main"]


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
