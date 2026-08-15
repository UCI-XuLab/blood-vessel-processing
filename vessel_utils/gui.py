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

import numpy as np

from vessel_utils.threshold import segment
from vessel_utils.vesselness import jerman_vesselness

__all__ = ["normalise", "reference_lambda", "stages", "MASK_METHODS", "tissue_mask", "main"]


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
