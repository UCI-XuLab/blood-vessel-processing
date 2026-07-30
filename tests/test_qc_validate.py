"""Tests for acquisition intake and for validation without dense annotation.

The stereological estimator is checked the only way that means anything: against
a phantom whose true mask is known, by feeding it the *correct* labels a perfect
reviewer would give and requiring the estimate to match the true precision and
recall it could not otherwise see.
"""

import numpy as np
import pytest

from vessel_utils import benchmark, qc, synth, validate


def phantom_pair(seed=0, shape=(32, 64, 64), spacing=(3.0, 0.75, 0.75), **acquisition):
    return synth.phantom(shape=shape, spacing=spacing, seed=seed,
                         acquisition_kwargs=acquisition or None)


# --------------------------------------------------------------------------
# intake
# --------------------------------------------------------------------------

def test_inspect_reports_shape_noise_and_contrast():
    volume, mask, _ = phantom_pair()
    info = qc.inspect_volume(volume, spacing=(3.0, 0.75, 0.75))
    assert info["shape"] == (32, 64, 64)
    assert info["spacing_um"] == (3.0, 0.75, 0.75)
    assert info["noise_sigma"] > 0
    assert info["contrast_to_noise"] > 1.0
    assert 0.0 <= info["saturated_fraction"] <= 1.0


def test_inspect_detects_saturation():
    saturated = np.full((8, 32, 32), 65535, dtype=np.uint16)
    info = qc.inspect_volume(saturated)
    assert info["saturated_fraction"] > 0.99


def test_inspect_rejects_an_all_nan_volume():
    with pytest.raises(ValueError, match="no finite values"):
        qc.inspect_volume(np.full((4, 8, 8), np.nan))


def test_attenuation_estimate_recovers_the_simulated_decay():
    """The estimator must return the decay constant that was put in."""
    mask = np.ones((60, 32, 32), bool)
    true_decay = 0.004
    volume = synth.simulate_acquisition(
        mask, (3.0, 1.0, 1.0), attenuation=true_decay, psf_sigma=(0, 0, 0),
        read_noise=0.0, gain=0.0)
    estimate = qc.estimate_attenuation(volume, spacing=(3.0, 1.0, 1.0),
                                       mask=np.ones_like(mask))
    assert estimate["decay_per_um"] == pytest.approx(true_decay, rel=0.15)
    assert estimate["half_depth_um"] == pytest.approx(np.log(2) / true_decay, rel=0.2)


def test_stripe_index_rises_with_stripe_strength():
    clean = np.random.default_rng(0).random((8, 64, 64)).astype(np.float32) * 50 + 100
    striped = clean * (1 - 0.4 * (np.random.default_rng(1).random(64) < 0.3))[None, :, None]
    assert (qc.stripe_severity(striped)["stripe_index"]
            > qc.stripe_severity(clean)["stripe_index"] * 2)


def test_resolvability_verdicts_match_the_sampling():
    fine = qc.resolvability((1.0, 0.5, 0.5), feature_um=4.0)
    assert fine["resolved"] and fine["worst_samples"] == pytest.approx(4.0)

    marginal = qc.resolvability((2.0, 0.5, 0.5), feature_um=4.0)
    assert marginal["nyquist_only"] and not marginal["resolved"]

    coarse = qc.resolvability((6.0, 0.5, 0.5), feature_um=4.0)
    assert not coarse["resolved"] and "UNRESOLVED" in coarse["verdict"]


def test_suggested_sigmas_never_go_below_the_sampling():
    sigmas = qc.suggest_sigmas((3.0, 0.75, 0.75))
    assert min(sigmas) >= 3.0, "searching finer than the voxel size finds noise"
    assert sigmas == sorted(sigmas)


def test_compare_channels_flags_mismatched_exposure():
    volume, mask, _ = phantom_pair(seed=0)
    dim, _, _ = phantom_pair(seed=0, signal=180.0, read_noise=30.0)
    comparison = qc.compare_channels(volume, dim, spacing=(3.0, 0.75, 0.75))
    assert not comparison["comparable"]
    assert any("contrast-to-noise" in w for w in comparison["warnings"])


def test_compare_channels_accepts_matched_channels():
    a, _, _ = phantom_pair(seed=0)
    b, _, _ = phantom_pair(seed=1)
    comparison = qc.compare_channels(a, b, spacing=(3.0, 0.75, 0.75))
    assert comparison["comparable"], comparison["warnings"]


def test_intake_report_warns_when_spacing_is_missing():
    volume, _, _ = phantom_pair()
    report = qc.intake_report(volume)
    assert any("spacing" in w for w in report["warnings"])
    assert "resolvability" not in report


def test_intake_report_renders_and_flags_unresolvable_sampling():
    volume, _, _ = phantom_pair()
    report = qc.intake_report(volume, spacing=(20.0, 20.0, 20.0))
    assert not report["resolvability"]["resolved"]
    assert "UNRESOLVED" in report["warnings"][0]
    text = qc.format_report(report)
    assert "ACQUISITION INTAKE REPORT" in text
    assert "WARNINGS" in text


# --------------------------------------------------------------------------
# stereological validation
# --------------------------------------------------------------------------

def test_stratified_sample_splits_between_strata():
    _, mask, _ = phantom_pair()
    sample = validate.stratified_sample(mask, n_points=200, seed=0)
    assert len(sample["positive_points"]) == 100
    assert len(sample["negative_points"]) == 100
    assert sample["n_positive_voxels"] == int(mask.sum())
    # Every drawn point must actually lie in its stratum.
    assert all(mask[tuple(p)] for p in sample["positive_points"])
    assert not any(mask[tuple(p)] for p in sample["negative_points"])


def test_stratified_sample_refuses_degenerate_masks():
    empty = np.zeros((8, 16, 16), bool)
    with pytest.raises(ValueError, match="empty"):
        validate.stratified_sample(empty)
    with pytest.raises(ValueError, match="covers everything"):
        validate.stratified_sample(~empty)


def test_estimator_recovers_known_precision_and_recall():
    """Feed it the labels a perfect reviewer would give; it must match the truth."""
    _, truth, _ = phantom_pair(seed=3)
    # A deliberately imperfect prediction: dilate, then drop a slab.
    import scipy.ndimage as ndi
    predicted = ndi.binary_dilation(truth, iterations=1)
    predicted[:6] = False

    actual = benchmark.score_segmentation(predicted, truth)

    sample = validate.stratified_sample(predicted, n_points=4000, seed=0)
    positive_labels = [truth[tuple(p)] for p in sample["positive_points"]]
    negative_labels = [truth[tuple(p)] for p in sample["negative_points"]]
    estimate = validate.estimate_accuracy(sample, positive_labels, negative_labels,
                                          n_bootstrap=500)

    assert estimate["precision"] == pytest.approx(actual["precision"], abs=0.05)
    assert estimate["recall"] == pytest.approx(actual["recall"], abs=0.10)
    assert estimate["precision_ci"][0] <= actual["precision"] <= estimate["precision_ci"][1]


def test_estimator_intervals_widen_with_fewer_points():
    _, truth, _ = phantom_pair(seed=3)
    import scipy.ndimage as ndi
    predicted = ndi.binary_dilation(truth, iterations=1)

    def width(n_points):
        sample = validate.stratified_sample(predicted, n_points=n_points, seed=1)
        pos = [truth[tuple(p)] for p in sample["positive_points"]]
        neg = [truth[tuple(p)] for p in sample["negative_points"]]
        result = validate.estimate_accuracy(sample, pos, neg, n_bootstrap=400)
        return result["precision_ci"][1] - result["precision_ci"][0]

    assert width(60) > width(1200) * 1.5


def test_estimator_checks_label_counts():
    _, mask, _ = phantom_pair()
    sample = validate.stratified_sample(mask, n_points=100, seed=0)
    with pytest.raises(ValueError, match="expected 50 labels"):
        validate.estimate_accuracy(sample, [1] * 10, [0] * 50)


def test_extract_crops_centres_are_correct_even_at_the_edge():
    volume = np.arange(8 * 16 * 16).reshape(8, 16, 16).astype(float)
    points = np.array([[4, 8, 8], [0, 0, 0]])
    crops, centres = validate.extract_crops(volume, points, size=(5, 7, 7))
    for crop, centre, point in zip(crops, centres, points):
        assert crop[centre] == volume[tuple(point)]


# --------------------------------------------------------------------------
# depth invariance
# --------------------------------------------------------------------------

def uniform_density_masks(shape=(48, 64, 64), seed=0):
    """A mask whose vessel density and calibre do not change with depth.

    A tree phantom is the wrong control here: its trunk is shallow and its
    capillaries are deep, so calibre varies with depth and *any* perturbation
    produces depth-dependent agreement. That is the metric working, not failing —
    but it means a genuine flatness test needs a depth-invariant mask.
    """
    rng = np.random.default_rng(seed)
    pattern = np.zeros(shape[1:], bool)
    for offset in range(6, shape[1] - 4, 12):
        pattern[offset:offset + 3, :] = True
    for offset in range(9, shape[2] - 4, 15):
        pattern[:, offset:offset + 3] = True
    truth = np.broadcast_to(pattern, shape).copy()

    other = truth.copy()
    flip = rng.random(shape) < 0.02          # constant error rate at every depth
    other[flip] = ~other[flip]
    return truth, other


def test_agreement_by_depth_is_flat_for_depth_independent_masks():
    truth, other = uniform_density_masks()
    rows = validate.agreement_by_depth(truth, other, n_bins=6)
    result = validate.depth_invariance(rows, tolerance=0.1)
    assert result["flat"], result["verdict"]
    assert result["spread"] < 0.1


def test_depth_drift_is_detected():
    """A channel that degrades with depth must be flagged, not averaged away."""
    _, truth, _ = phantom_pair(seed=4, shape=(48, 64, 64))
    degraded = truth.copy()
    depth = truth.shape[0]
    rng = np.random.default_rng(0)
    for z in range(depth):
        # Progressively drop vessel voxels with depth, as attenuation would.
        drop = rng.random(truth.shape[1:]) < (z / depth) * 0.9
        degraded[z][drop] = False

    rows = validate.agreement_by_depth(truth, degraded, n_bins=6)
    result = validate.depth_invariance(rows, tolerance=0.1)
    assert not result["flat"]
    assert result["total_change"] < 0
    assert "drifts" in result["verdict"]


def test_depth_invariance_needs_usable_bins():
    with pytest.raises(ValueError):
        validate.depth_invariance([])
    with pytest.raises(ValueError, match="at least two"):
        validate.depth_invariance([{"centre": 1.0, "dice": float("nan")},
                                   {"centre": 2.0, "dice": float("nan")}])


def test_agreement_by_depth_validates_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        validate.agreement_by_depth(np.ones((4, 8, 8), bool), np.ones((4, 8, 9), bool))
