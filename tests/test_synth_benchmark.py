"""Tests for the phantom generator and the benchmark harness.

The phantom is the only place in this project with a known-correct answer, so its
own correctness carries weight: if the generator is wrong, every accuracy number
derived from it is wrong in the same direction and nothing downstream will notice.
These check the geometry and the physics separately from the segmentation.
"""

import numpy as np
import pytest

from vessel_utils import benchmark, synth


# --------------------------------------------------------------------------
# tree geometry
# --------------------------------------------------------------------------

def test_tree_branches_rather_than_producing_one_trunk():
    """A trunk sized independently of the box swallows it; the default must not."""
    segments = synth.vascular_tree((144.0, 72.0, 72.0), seed=0)
    assert len(segments) > 20, f"expected a developed tree, got {len(segments)}"


def test_default_root_radius_scales_with_the_box():
    small = synth.default_root_radius((72.0, 36.0, 36.0))
    large = synth.default_root_radius((288.0, 144.0, 144.0))
    assert large == pytest.approx(small * 4, rel=1e-6)


def test_radii_obey_murray_at_bifurcations():
    """r_parent^3 should equal the sum of the daughters' cubes."""
    segments = synth.vascular_tree((144.0, 72.0, 72.0), seed=1, asymmetry=0.0)
    by_start = {}
    for segment in segments:
        by_start.setdefault(segment.start, []).append(segment)

    checked = 0
    for parent in segments:
        children = by_start.get(parent.end, [])
        if len(children) != 2:
            continue
        total = sum(c.radius_start ** 3 for c in children)
        assert total == pytest.approx(parent.radius_start ** 3, rel=1e-6)
        checked += 1
    assert checked > 5, "expected several bifurcations to verify"


def test_radii_span_arteriole_to_capillary():
    segments = synth.vascular_tree((192.0, 96.0, 96.0), min_radius=1.5, seed=2)
    radii = np.array([s.radius_start for s in segments])
    assert radii.min() < 2.5, "tree must reach capillary calibre"
    assert radii.max() > 5.0, "tree must contain larger vessels too"
    assert (radii >= 1.5 * 0.79).all(), "nothing below the stopping radius"


def test_tree_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="too small for a tree"):
        synth.vascular_tree((10.0, 10.0, 10.0), min_radius=5.0)
    with pytest.raises(ValueError, match="asymmetry"):
        synth.vascular_tree((144.0, 72.0, 72.0), asymmetry=0.7)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_rendered_radius_matches_the_requested_radius():
    """A single straight capsule should be as thick as it says it is."""
    spacing = (0.5, 0.5, 0.5)
    radius = 4.0
    segment = synth.Segment(start=(16.0, 16.0, 2.0), end=(16.0, 16.0, 30.0),
                            radius_start=radius, radius_end=radius)
    mask = synth.render_tree([segment], (64, 64, 64), spacing)

    cross_section = mask[:, :, 32]
    measured_area = cross_section.sum() * spacing[0] * spacing[1]
    assert measured_area == pytest.approx(np.pi * radius ** 2, rel=0.1)


def test_rendering_respects_anisotropic_spacing():
    """The same physical vessel occupies fewer voxels along a coarse axis."""
    segment = synth.Segment(start=(30.0, 30.0, 5.0), end=(30.0, 30.0, 55.0),
                            radius_start=6.0, radius_end=6.0)
    mask = synth.render_tree([segment], (20, 60, 60), (3.0, 1.0, 1.0))
    profile = mask[:, :, 30]
    z_extent = profile.any(axis=1).sum() * 3.0
    y_extent = profile.any(axis=0).sum() * 1.0
    assert z_extent == pytest.approx(y_extent, rel=0.25)


def test_tapering_segment_is_thicker_at_the_wide_end():
    segment = synth.Segment(start=(16.0, 16.0, 2.0), end=(16.0, 16.0, 30.0),
                            radius_start=6.0, radius_end=2.0)
    mask = synth.render_tree([segment], (64, 64, 64), (0.5, 0.5, 0.5))
    near = mask[:, :, 8].sum()
    far = mask[:, :, 56].sum()
    assert near > far * 2


def test_segments_leaving_the_box_are_clipped_not_dropped():
    segment = synth.Segment(start=(16.0, 16.0, -20.0), end=(16.0, 16.0, 20.0),
                            radius_start=3.0, radius_end=3.0)
    mask = synth.render_tree([segment], (64, 64, 64), (0.5, 0.5, 0.5))
    assert mask.any()
    assert mask[:, :, 0].any(), "the part inside the box should be rendered"


# --------------------------------------------------------------------------
# acquisition physics
# --------------------------------------------------------------------------

def test_each_degradation_is_independently_disableable():
    """Attribution depends on being able to turn effects off one at a time."""
    mask = np.zeros((16, 32, 32), bool)
    mask[6:10, 12:20, 12:20] = True
    spacing = (3.0, 0.75, 0.75)

    clean = synth.simulate_acquisition(
        mask, spacing, psf_sigma=(0, 0, 0), attenuation=0.0, stripe_strength=0.0,
        read_noise=0.0, gain=0.0)
    assert set(np.unique(clean.round(3))) == {100.0, 800.0}


def test_attenuation_reduces_deep_signal():
    mask = np.ones((40, 16, 16), bool)
    volume = synth.simulate_acquisition(
        mask, (3.0, 1.0, 1.0), attenuation=0.005, psf_sigma=(0, 0, 0),
        read_noise=0.0, gain=0.0)
    assert volume[-1].mean() < volume[0].mean() * 0.6


def test_anisotropic_psf_blurs_more_along_z():
    mask = np.zeros((32, 32, 32), bool)
    mask[16, 16, 16] = True
    volume = synth.simulate_acquisition(
        mask, (1.0, 1.0, 1.0), psf_sigma=(4.0, 1.0, 1.0), attenuation=0.0,
        read_noise=0.0, gain=0.0, background=0.0)
    z_spread = (volume[:, 16, 16] > volume.max() * 0.1).sum()
    x_spread = (volume[16, 16, :] > volume.max() * 0.1).sum()
    assert z_spread > x_spread * 2


def test_lower_gain_produces_more_shot_noise():
    mask = np.zeros((8, 32, 32), bool)
    noisy = synth.simulate_acquisition(mask, (3.0, 1.0, 1.0), gain=0.05,
                                       psf_sigma=(0, 0, 0), read_noise=0.0,
                                       attenuation=0.0)
    clean = synth.simulate_acquisition(mask, (3.0, 1.0, 1.0), gain=50.0,
                                       psf_sigma=(0, 0, 0), read_noise=0.0,
                                       attenuation=0.0)
    assert noisy.std() > clean.std() * 3


def test_phantom_returns_matching_volume_mask_and_segments():
    volume, mask, segments = synth.phantom(shape=(32, 64, 64), seed=0)
    assert volume.shape == mask.shape == (32, 64, 64)
    assert 0.001 < mask.mean() < 0.15, f"implausible vessel fraction {mask.mean()}"
    assert volume[mask].mean() > volume[~mask].mean() * 2
    assert len(segments) > 10


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------

def perfect_segmenter(volume, spacing):
    """Cheats by thresholding the noiseless-ish signal; a sanity upper bound."""
    return volume > (volume.min() + volume.max()) / 2


def test_score_segmentation_rewards_a_perfect_match():
    _volume, mask, _ = synth.phantom(shape=(24, 48, 48), seed=0)
    scores = benchmark.score_segmentation(mask, mask)
    assert scores["dice"] == pytest.approx(1.0, abs=1e-6)
    assert scores["cl_dice"] == pytest.approx(1.0, abs=1e-6)
    assert scores["recall"] == pytest.approx(1.0, abs=1e-6)


def test_score_segmentation_punishes_an_empty_prediction():
    _volume, mask, _ = synth.phantom(shape=(24, 48, 48), seed=0)
    scores = benchmark.score_segmentation(np.zeros_like(mask), mask)
    assert scores["dice"] < 1e-6
    assert scores["cl_dice"] == 0.0


def test_benchmark_runs_several_phantoms_and_summarises():
    rows = benchmark.run_benchmark(perfect_segmenter, n_phantoms=3,
                               shape=(24, 48, 48), seed=0)
    assert len(rows) == 3
    assert {row["seed"] for row in rows} == {0, 1, 2}

    summary = benchmark.summarise(rows)
    assert summary["n_phantoms"] == 3
    assert 0.0 <= summary["dice_mean"] <= 1.0
    assert summary["dice_ci_low"] <= summary["dice_mean"] <= summary["dice_ci_high"]


def test_benchmark_reports_recall_by_calibre():
    rows = benchmark.run_benchmark(perfect_segmenter, n_phantoms=1, shape=(24, 48, 48),
                               spacing=(3.0, 0.75, 0.75), seed=0,
                               calibre_edges=[0.0, 2.0, 4.0, 20.0])
    bins = rows[0]["by_calibre"]
    assert len(bins) == 3
    assert all("recall" in b for b in bins)


def test_sweep_condition_varies_one_acquisition_parameter():
    rows = benchmark.sweep_condition(
        perfect_segmenter, "gain", [0.02, 5.0], n_phantoms=1, shape=(24, 48, 48))
    assert [row["gain"] for row in rows] == [0.02, 5.0]
    # Heavy shot noise must hurt a plain intensity threshold.
    assert rows[0]["dice_mean"] < rows[1]["dice_mean"]


def test_summarise_rejects_empty_input():
    with pytest.raises(ValueError):
        benchmark.summarise([])
