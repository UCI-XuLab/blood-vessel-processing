"""Tests for the active pipeline.

Unlike the archive's suite, there is no historical implementation to match here.
These check the properties the module claims: that scales are physical, that the
response is bounded and calibre-uniform, that a fixed reference removes the
per-image dependence, and that the metrics behave the way the docstrings say.
"""

import numpy as np
import pytest

from vessel_utils import metrics, sweep, threshold, vesselness


# --------------------------------------------------------------------------
# synthetic vessels
# --------------------------------------------------------------------------

def tube_2d(shape=(96, 96), radius=3.0, centre=None, amplitude=1.0):
    """A horizontal cylinder of given radius, bright on a dark background."""
    centre = centre if centre is not None else shape[0] / 2
    rows = np.arange(shape[0])[:, None]
    distance = np.abs(rows - centre)
    return np.broadcast_to((distance <= radius).astype(np.float32) * amplitude,
                           shape).copy()


def tube_3d(shape=(32, 48, 48), radius=3.0, amplitude=1.0):
    """A cylinder running along the last axis."""
    z = np.arange(shape[0])[:, None, None] - shape[0] / 2
    y = np.arange(shape[1])[None, :, None] - shape[1] / 2
    distance = np.sqrt(z ** 2 + y ** 2)
    return np.broadcast_to((distance <= radius).astype(np.float32) * amplitude,
                           shape).copy()


# --------------------------------------------------------------------------
# hessian and spacing
# --------------------------------------------------------------------------

def test_eigenvalues_sorted_by_magnitude():
    eigenvalues = vesselness.hessian_eigenvalues(tube_2d(), sigma=3.0)
    magnitudes = np.abs(eigenvalues)
    assert (magnitudes[..., 0] <= magnitudes[..., 1] + 1e-6).all()


def test_eigenvalues_work_in_3d():
    eigenvalues = vesselness.hessian_eigenvalues(tube_3d(), sigma=3.0)
    assert eigenvalues.shape == (32, 48, 48, 3)
    magnitudes = np.abs(eigenvalues)
    assert (magnitudes[..., 0] <= magnitudes[..., 1] + 1e-5).all()
    assert (magnitudes[..., 1] <= magnitudes[..., 2] + 1e-5).all()


def test_spacing_makes_sigma_physical():
    """A sigma in microns must select the same structure whatever the voxel size.

    Two images of the same tube, one sampled twice as finely, filtered at the same
    physical sigma, should put their peak response at the same physical place.
    """
    fine = tube_2d(shape=(96, 96), radius=6.0)
    coarse = fine[::2]                       # half the rows: 2x coarser in y

    def physical_width(response, row_spacing):
        """Extent of the responding band along the sampled axis, in physical units."""
        column = response[:, response.shape[1] // 2]
        return float((column > 0.5).sum()) * row_spacing

    width_fine = physical_width(
        vesselness.jerman_vesselness(fine, [6.0], spacing=(1.0, 1.0)), 1.0)
    width_aware = physical_width(
        vesselness.jerman_vesselness(coarse, [6.0], spacing=(2.0, 1.0)), 2.0)
    width_naive = physical_width(
        vesselness.jerman_vesselness(coarse, [6.0], spacing=None), 2.0)

    assert width_fine > 0
    # Spacing-aware: the same physical sigma describes the same physical extent.
    assert width_aware == pytest.approx(width_fine, rel=0.25), (width_fine, width_aware)
    # Ignoring spacing searches the wrong physical scale on anisotropic data.
    assert abs(width_naive - width_fine) > abs(width_aware - width_fine)


def test_rejects_bad_spacing_and_sigma():
    image = tube_2d()
    with pytest.raises(ValueError):
        vesselness.hessian_eigenvalues(image, sigma=0.0)
    with pytest.raises(ValueError):
        vesselness.hessian_eigenvalues(image, sigma=2.0, spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        vesselness.hessian_eigenvalues(image, sigma=2.0, spacing=(0.0, 1.0))
    with pytest.raises(ValueError):
        vesselness.jerman_vesselness(image, [2.0], tau=0.0)
    with pytest.raises(ValueError):
        vesselness.jerman_vesselness(image, [])


# --------------------------------------------------------------------------
# the properties that motivated the switch away from the archive's filter
# --------------------------------------------------------------------------

def test_response_is_bounded_in_unit_interval():
    """Unlike the archive's 0-255 rescale, the response has an absolute scale."""
    for image in (tube_2d(radius=4.0), tube_3d(radius=4.0)):
        response = vesselness.jerman_vesselness(image, [2.0, 4.0, 8.0])
        assert response.min() >= 0.0
        assert response.max() <= 1.0 + 1e-6


def test_response_survives_intensity_scaling_with_a_fixed_reference():
    """The headline consistency property.

    The archive's rescale makes a fixed threshold mean something different in
    every image. With `reference_lambda` pinned, brightening the image must not
    change which voxels clear a fixed cut.
    """
    dim = tube_2d(radius=4.0, amplitude=1.0)
    bright = tube_2d(radius=4.0, amplitude=10.0)
    sigmas = [2.0, 4.0]

    reference = vesselness.max_eigenvalue(dim, sigmas)
    a = vesselness.jerman_vesselness(dim, sigmas, reference_lambda=reference)
    b = vesselness.jerman_vesselness(bright, sigmas, reference_lambda=reference)

    # Same structure, 10x brighter: the mask at a fixed threshold must not move.
    assert np.array_equal(a > 0.5, b > 0.5)


def test_per_image_reference_is_the_thing_that_drifts():
    """Without a fixed reference the response is image-dependent, as documented."""
    dim = tube_2d(radius=4.0, amplitude=1.0)
    spiked = dim.copy()
    spiked[:6, :6] = 50.0                     # a bright artefact

    sigmas = [2.0, 4.0]
    a = vesselness.jerman_vesselness(dim, sigmas)
    b = vesselness.jerman_vesselness(spiked, sigmas)
    assert not np.array_equal(a > 0.5, b > 0.5)

    reference = vesselness.max_eigenvalue(dim, sigmas)
    a_fixed = vesselness.jerman_vesselness(dim, sigmas, reference_lambda=reference)
    b_fixed = vesselness.jerman_vesselness(spiked, sigmas, reference_lambda=reference)
    # Pinning the reference confines the artefact's influence to the artefact.
    differing = np.argwhere((a_fixed > 0.5) != (b_fixed > 0.5))
    if differing.size:
        assert differing[:, 0].max() < 20, "artefact affected distant pixels"


def test_response_is_uniform_across_vessel_calibre():
    """Jerman's selling point over Frangi: similar response for thick and thin."""
    sigmas = [1.0, 2.0, 4.0, 8.0]
    peaks = [vesselness.jerman_vesselness(tube_2d(radius=r), sigmas).max()
             for r in (2.0, 4.0, 8.0)]
    assert min(peaks) > 0.9, f"response varies with calibre: {peaks}"


def test_bright_and_dark_polarity_are_opposites():
    image = tube_2d(radius=4.0)
    bright = vesselness.jerman_vesselness(image, [2.0, 4.0], bright_objects=True)
    dark = vesselness.jerman_vesselness(image, [2.0, 4.0], bright_objects=False)
    assert bright.max() > 0.5
    assert dark[image > 0].max() < bright[image > 0].max()


def test_normalise_reproduces_reference_behaviour():
    """The reference implementation divides by the max and zeroes below 1e-2."""
    image = tube_2d(radius=4.0)
    raw = vesselness.jerman_vesselness(image, [2.0, 4.0], normalise=False)
    normalised = vesselness.jerman_vesselness(image, [2.0, 4.0], normalise=True)
    assert normalised.max() == pytest.approx(1.0, abs=1e-6)
    assert not ((normalised > 0) & (normalised < 1e-2)).any()
    assert raw.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------
# thresholding
# --------------------------------------------------------------------------

def test_hysteresis_recovers_a_faint_bridge_a_single_cut_would_drop():
    response = np.zeros((20, 40), dtype=float)
    response[10, :15] = 0.9          # strong segment
    response[10, 15:25] = 0.4        # faint but connected bridge
    response[10, 25:] = 0.9          # strong segment
    response[2, 5] = 0.4             # isolated faint speck

    single = response > 0.6
    hysteresis = threshold.hysteresis_threshold(response, low=0.3, high=0.6)

    assert not single[10, 20], "single cut should drop the bridge"
    assert hysteresis[10, 20], "hysteresis should keep the connected bridge"
    assert not hysteresis[2, 5], "hysteresis must not keep the isolated speck"


def test_hysteresis_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        threshold.hysteresis_threshold(np.zeros((4, 4)), low=0.9, high=0.1)


def test_clean_mask_works_in_2d_and_3d():
    mask2 = np.zeros((40, 40), bool)
    mask2[10:30, 10:12] = True
    mask2[0, 0] = True                                  # speck
    cleaned2 = threshold.clean_mask(mask2, min_size=5, area_threshold=0,
                                    closing_radius=1)
    assert not cleaned2[0, 0]
    assert cleaned2[20, 10]

    mask3 = np.zeros((20, 20, 20), bool)
    mask3[5:15, 8:12, 8:12] = True
    mask3[0, 0, 0] = True
    cleaned3 = threshold.clean_mask(mask3, min_size=5, area_threshold=0,
                                    closing_radius=1)
    assert not cleaned3[0, 0, 0]
    assert cleaned3[10, 10, 10]


def test_otsu_needs_a_non_constant_response():
    with pytest.raises(ValueError):
        threshold.otsu_threshold(np.ones((8, 8)))


def test_segment_applies_roi_before_cleanup():
    response = np.zeros((30, 30))
    response[10:20, 10:20] = 1.0
    roi = np.zeros((30, 30), bool)
    roi[:15] = True
    mask = threshold.segment(response, 0.4, 0.6, roi=roi, min_size=1,
                             area_threshold=0, closing_radius=0)
    assert mask[12, 12]
    assert not mask[18, 12]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def random_masks(seed=0, fraction=0.05, shape=(128, 128)):
    rng = np.random.default_rng(seed)
    a = rng.random(shape) < fraction
    b = a.copy()
    flip = rng.random(shape) < 0.01
    b[flip] = ~b[flip]
    return a, b


def test_dice_and_jaccard_are_symmetric_precision_recall_are_not():
    a, b = random_masks()
    assert metrics.dice(a, b) == pytest.approx(metrics.dice(b, a))
    assert metrics.jaccard(a, b) == pytest.approx(metrics.jaccard(b, a))
    assert metrics.precision(a, b) != pytest.approx(metrics.recall(a, b))
    assert metrics.precision(a, b) == pytest.approx(metrics.recall(b, a))


def test_jaccard_is_dtype_independent():
    """The archive's `iou` silently halves on float masks; this one must not."""
    a, b = random_masks()
    boolean = metrics.jaccard(a, b)
    numeric = metrics.jaccard(a.astype(float), b.astype(float))
    assert boolean == pytest.approx(numeric)

    from velazquez_rivera_2025.metrics import iou as archive_iou
    assert archive_iou(a, b) == pytest.approx(boolean, abs=1e-6)      # agrees on bool
    assert archive_iou(a.astype(float), b.astype(float)) != pytest.approx(boolean)


def test_identical_and_disjoint_masks_score_as_expected():
    a, _ = random_masks()
    assert metrics.dice(a, a) == pytest.approx(1.0, abs=1e-6)
    assert metrics.jaccard(a, a) == pytest.approx(1.0, abs=1e-6)
    assert metrics.dice(a, ~a) < 1e-6


def test_cl_dice_is_symmetric_and_notices_a_break():
    line = np.zeros((40, 40), bool)
    line[20, 5:35] = True
    broken = line.copy()
    broken[20, 20] = False           # one voxel severs the vessel

    assert metrics.cl_dice(line, broken) == pytest.approx(metrics.cl_dice(broken, line))
    # Voxel overlap barely registers a single missing pixel; topology does.
    assert metrics.dice(line, broken) > 0.98
    assert metrics.cl_dice(line, broken) < metrics.dice(line, broken)


def test_cl_dice_handles_empty_masks():
    empty = np.zeros((16, 16), bool)
    filled = np.zeros((16, 16), bool)
    filled[8, 2:14] = True
    assert metrics.cl_dice(empty, filled) == 0.0
    assert metrics.cl_dice(empty, empty) == 0.0


def test_area_fraction_respects_the_roi():
    mask = np.zeros((20, 20), bool)
    mask[:10, :10] = True
    roi = np.zeros((20, 20), bool)
    roi[:10] = True
    assert metrics.area_fraction(mask) == pytest.approx(0.25)
    assert metrics.area_fraction(mask, roi) == pytest.approx(0.5)
    assert np.isnan(metrics.area_fraction(mask, np.zeros((20, 20), bool)))


def test_agreement_reports_both_area_fractions():
    a, b = random_masks()
    report = metrics.agreement(a, b)
    assert {"dice", "jaccard", "precision_a_in_b", "recall_b_in_a", "cl_dice",
            "area_fraction_a", "area_fraction_b"} <= set(report)
    assert report["voxels_a"] == int(a.sum())


def test_agreement_by_calibre_separates_thick_from_thin():
    """Thin vessels dropped by the other channel must show up as low recall."""
    reference = np.zeros((60, 80), bool)
    reference[10:21, 5:75] = True                 # thick vessel, radius ~6
    reference[45, 5:75] = True                    # capillary, radius ~1

    other = reference.copy()
    other[45, :] = False                          # the other channel missed the capillary

    rows = metrics.agreement_by_calibre(reference, other, edges=[0.0, 3.0, 20.0])
    thin, thick = rows[0], rows[1]
    # Every voxel of a vessel inherits that vessel's radius, so the bins separate
    # cleanly even though the thick vessel spans many distances internally.
    assert thin["voxels"] > 0 and thick["voxels"] > 0
    assert thin["recall"] < 0.01
    assert thick["recall"] > 0.99


def test_agreement_by_calibre_validates_edges():
    mask = np.ones((8, 8), bool)
    with pytest.raises(ValueError):
        metrics.agreement_by_calibre(mask, mask, edges=[1.0])
    with pytest.raises(ValueError):
        metrics.agreement_by_calibre(mask, mask, edges=[2.0, 1.0])


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.dice(np.ones((4, 4), bool), np.ones((5, 5), bool))


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def two_channel_responses():
    rng = np.random.default_rng(3)
    truth = np.zeros((64, 64), dtype=float)
    for row in range(8, 60, 12):
        truth[row:row + 2, 4:60] = 1.0
    a = np.clip(truth * 0.8 + rng.random(truth.shape) * 0.10, 0, 1)
    b = np.clip(truth * 0.7 + rng.random(truth.shape) * 0.10, 0, 1)
    return a, b


def test_threshold_sweep_returns_one_row_per_threshold():
    a, b = two_channel_responses()
    thresholds = np.linspace(0.2, 0.6, 5)
    rows = sweep.threshold_sweep(a, b, thresholds, min_size=1, area_threshold=0,
                                 closing_radius=0)
    assert len(rows) == len(thresholds)
    assert rows[0]["threshold_low"] == pytest.approx(rows[0]["threshold_high"] * 0.5)
    assert all("dice" in row for row in rows)


def test_threshold_sweep_reports_progress_and_validates_inputs():
    a, b = two_channel_responses()
    seen = []
    sweep.threshold_sweep(a, b, [0.3, 0.5], min_size=1, area_threshold=0,
                          closing_radius=0, include_topology=False,
                          progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]

    with pytest.raises(ValueError):
        sweep.threshold_sweep(a, b[:10], [0.3])
    with pytest.raises(ValueError):
        sweep.threshold_sweep(a, b, [0.3], ratio=0.0)


def test_stability_finds_a_plateau():
    rows = [{"threshold_high": t, "dice": d} for t, d in
            [(0.1, 0.20), (0.2, 0.80), (0.3, 0.81), (0.4, 0.82), (0.5, 0.80),
             (0.6, 0.30)]]
    result = sweep.stability(rows, "dice", tolerance=0.05)
    assert result["threshold_low"] == 0.2
    assert result["threshold_high"] == 0.5
    assert result["coverage"] == pytest.approx(4 / 6)


def test_stability_reports_a_narrow_plateau_when_there_is_none():
    rows = [{"threshold_high": t, "dice": d} for t, d in
            [(0.1, 0.1), (0.2, 0.4), (0.3, 0.7), (0.4, 0.95)]]
    result = sweep.stability(rows, "dice", tolerance=0.05)
    assert result["coverage"] < 0.6
    with pytest.raises(ValueError):
        sweep.stability([], "dice")


def test_write_csv_round_trips(tmp_path):
    import csv as csv_module
    a, b = two_channel_responses()
    rows = sweep.threshold_sweep(a, b, [0.3, 0.5], min_size=1, area_threshold=0,
                                 closing_radius=0, include_topology=False)
    path = sweep.write_csv(rows, tmp_path / "sweep.csv")
    with open(path, encoding="utf-8") as handle:
        loaded = list(csv_module.DictReader(handle))
    assert len(loaded) == 2
    assert float(loaded[0]["threshold_high"]) == pytest.approx(0.3)


# --------------------------------------------------------------------------
# package surface
# --------------------------------------------------------------------------

def test_every_exported_name_resolves():
    import vessel_utils
    for name in vessel_utils.__all__:
        assert callable(getattr(vessel_utils, name)), name


def test_metrics_do_not_require_the_vesselness_dependencies():
    """Lazy re-export: reaching for a metric must not import the filter stack."""
    import subprocess
    import sys
    code = (
        "import sys, vessel_utils;"
        "vessel_utils.dice;"
        "assert 'vessel_utils.vesselness' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(
        __import__("pathlib").Path(__file__).resolve().parent.parent))


def test_max_eigenvalue_percentile_is_more_stable_than_the_max():
    """The max is set by one bright structure; a high quantile is not.

    This is why calibration across a dataset should not use the default: a
    single unusually bright object moves the reference, and the reference sets
    the vessel criterion for every image calibrated against it.
    """
    # The image must be large enough that a small speck sits outside the 99.9th
    # percentile: at 96x96 a 3x3 speck is 9 of 9216 pixels and lands exactly in
    # that tail, so the quantile would move too and the test would prove nothing.
    base = tube_2d(shape=(320, 320), radius=4.0, amplitude=1.0)
    with_outlier = base.copy()
    with_outlier[:3, :3] = 60.0                    # 9 of 102400 pixels

    sigmas = [2.0, 4.0]
    max_clean = vesselness.max_eigenvalue(base, sigmas)
    max_spiked = vesselness.max_eigenvalue(with_outlier, sigmas)
    pct_clean = vesselness.max_eigenvalue(base, sigmas, percentile=99.9)
    pct_spiked = vesselness.max_eigenvalue(with_outlier, sigmas, percentile=99.9)

    assert max_spiked > max_clean * 2, "the max should be dragged by the speck"
    assert pct_spiked == pytest.approx(pct_clean, rel=0.25), \
        "a high quantile should barely move"


def test_max_eigenvalue_respects_a_mask_and_validates_percentile():
    image = tube_2d(radius=4.0)
    roi = np.zeros(image.shape, bool)
    roi[image.shape[0] // 2 - 10: image.shape[0] // 2 + 10] = True
    masked = vesselness.max_eigenvalue(image, [2.0], percentile=99.0, mask=roi)
    assert masked > 0
    with pytest.raises(ValueError, match="percentile"):
        vesselness.max_eigenvalue(image, [2.0], percentile=0.0)


def structured_2d(shape=(256, 256), seed=0):
    rng = np.random.default_rng(seed)
    image = rng.random(shape).astype(np.float32) * 50 + 100
    for offset in range(20, shape[0] - 10, 40):
        image[offset:offset + 5, :] += 600
    return image


def test_2d_response_is_near_binary():
    """Jerman in 2D sets lambda_3 := lambda_2, so saturation is unconditional.

    Documented because it changes how the module must be used: in 2D the
    operating point is reference_lambda, not the threshold.
    """
    response = vesselness.jerman_vesselness(structured_2d(), [2.0, 4.0],
                                            reference_lambda=2.0)
    between = np.mean((response > 0.01) & (response < 0.99))
    assert between < 0.05, f"{between:.3f} of voxels lie strictly between 0 and 1"


def test_2d_threshold_barely_moves_the_mask_but_reference_does():
    image = structured_2d()
    response = vesselness.jerman_vesselness(image, [2.0, 4.0], reference_lambda=4.0)
    loose = float(np.mean(response > 0.10))
    tight = float(np.mean(response > 0.90))
    assert abs(loose - tight) < 0.10, "the 2D threshold should be nearly inert"

    # The reference, by contrast, is what actually sets the mask. It needs a wide
    # range to show it: the response only starts responding once tau*ref/2 rises
    # into the bulk of the lambda_2 distribution, so small references all saturate
    # alike. That is itself worth knowing when choosing one.
    small = np.mean(vesselness.jerman_vesselness(image, [2.0, 4.0],
                                                 reference_lambda=2.0) > 0.5)
    large = np.mean(vesselness.jerman_vesselness(image, [2.0, 4.0],
                                                 reference_lambda=40.0) > 0.5)
    assert small - large > 0.15, "reference_lambda must move the 2D mask"


def test_2d_mask_matches_the_cheap_eigenvalue_shortcut():
    """The shortcut a sweep relies on: mask == max_sigma lambda_2 >= tau*ref/2."""
    image = structured_2d()
    sigmas, tau, reference = [2.0, 4.0], 0.75, 4.0
    lambda_2 = np.max([-vesselness.hessian_eigenvalues(image, s)[..., 1]
                       for s in sigmas], axis=0)
    predicted = lambda_2 >= tau * reference / 2
    actual = vesselness.jerman_vesselness(image, sigmas, tau=tau,
                                          reference_lambda=reference) >= 0.99
    assert np.mean(predicted == actual) > 0.97
