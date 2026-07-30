"""Tests for storage, correction and chunked processing.

The corrections are checked by simulating the artefact on a known-clean volume
and asserting the correction recovers it — the only honest way to test a
correction before real data exists. Each simulator models the artefact the way
the physics produces it: stripes as a multiplicative shadow along the sheet axis,
attenuation as exponential decay with depth.
"""

import numpy as np
import pytest

from vessel_utils import chunked, correct, storage


# --------------------------------------------------------------------------
# simulated acquisitions
# --------------------------------------------------------------------------

def clean_volume(shape=(48, 96, 96), seed=0):
    """A volume with vessel-like tubes on a textured background."""
    rng = np.random.default_rng(seed)
    volume = rng.random(shape).astype(np.float32) * 20 + 100
    for offset in range(12, shape[1] - 8, 24):
        volume[:, offset:offset + 3, :] += 400          # tubes running along x
    for offset in range(16, shape[2] - 8, 32):
        volume[:, :, offset:offset + 2] += 300          # tubes running along y
    return volume


def add_stripes(plane, strength=0.35, axis=1, seed=0):
    """Multiplicative shadows along the illumination axis."""
    rng = np.random.default_rng(seed)
    n = plane.shape[1 - axis]
    shadow = 1.0 - strength * (rng.random(n) < 0.15) * rng.random(n)
    shadow = shadow[:, None] if axis == 1 else shadow[None, :]
    return (plane * shadow).astype(np.float32)


def add_depth_attenuation(volume, decay=0.02, axis=0):
    """Exponential signal loss with depth, as in cleared tissue."""
    depth = volume.shape[axis]
    falloff = np.exp(-decay * np.arange(depth))
    shape = [1] * volume.ndim
    shape[axis] = depth
    return (volume * falloff.reshape(shape)).astype(np.float32), falloff


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def write_plane_series(directory, volume, prefix="plane", pad=4):
    import tifffile
    directory.mkdir(parents=True, exist_ok=True)
    for index, plane in enumerate(volume):
        tifffile.imwrite(directory / f"{prefix}_Z{index:0{pad}d}_C0.tif", plane)
    return str(directory / "*_C0.tif")


def test_plane_series_round_trips_with_spacing(tmp_path):
    volume = clean_volume((20, 64, 64))
    pattern = write_plane_series(tmp_path / "raw", volume)

    store = storage.plane_series_to_zarr(
        pattern, tmp_path / "vol.zarr", spacing=(3.0, 0.75, 0.75),
        chunks=(8, 32, 32))

    array, spacing = storage.open_volume(store)
    assert array.shape == volume.shape
    assert spacing == (3.0, 0.75, 0.75)
    np.testing.assert_allclose(np.asarray(array), volume, rtol=1e-6)


def test_pyramid_levels_have_correct_physical_spacing(tmp_path):
    """Coarse levels must report their own spacing, not the full-resolution one."""
    volume = clean_volume((8, 128, 128))
    pattern = write_plane_series(tmp_path / "raw", volume)
    store = storage.plane_series_to_zarr(
        pattern, tmp_path / "vol.zarr", spacing=(3.0, 0.5, 0.5),
        chunks=(4, 64, 64), n_levels=3)

    for level, expected_xy in [(0, 0.5), (1, 1.0), (2, 2.0)]:
        array, spacing = storage.open_volume(store, level=level)
        assert spacing[0] == 3.0, "z must not be downsampled"
        assert spacing[1] == pytest.approx(expected_xy)
        assert array.shape[1] == 128 // (2 ** level)


def test_unpadded_plane_numbering_is_refused(tmp_path):
    """sorted() puts Z10 before Z9; on 2607 planes that scrambles the volume."""
    volume = clean_volume((12, 32, 32))
    directory = tmp_path / "raw"
    directory.mkdir()
    import tifffile
    for index, plane in enumerate(volume):
        tifffile.imwrite(directory / f"plane_Z{index}_C0.tif", plane)   # no padding

    with pytest.raises(ValueError, match="zero-padded"):
        storage.plane_series_to_zarr(str(directory / "*_C0.tif"),
                                     tmp_path / "vol.zarr", spacing=(3.0, 1.0, 1.0))


def test_inconsistent_plane_shape_is_refused(tmp_path):
    import tifffile
    directory = tmp_path / "raw"
    directory.mkdir()
    tifffile.imwrite(directory / "p_Z0000_C0.tif", np.zeros((32, 32), np.uint16))
    tifffile.imwrite(directory / "p_Z0001_C0.tif", np.zeros((32, 48), np.uint16))
    with pytest.raises(ValueError, match="single consistent volume"):
        storage.plane_series_to_zarr(str(directory / "*_C0.tif"),
                                     tmp_path / "vol.zarr", spacing=(3.0, 1.0, 1.0))


def test_missing_input_and_bad_spacing_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.plane_series_to_zarr(str(tmp_path / "nothing" / "*.tif"),
                                     tmp_path / "v.zarr", spacing=(1, 1, 1))
    pattern = write_plane_series(tmp_path / "raw", clean_volume((4, 16, 16)))
    with pytest.raises(ValueError, match="three positive values"):
        storage.plane_series_to_zarr(pattern, tmp_path / "v.zarr", spacing=(1.0, 0.0, 1.0))


def test_write_volume_preserves_spacing(tmp_path):
    volume = clean_volume((8, 32, 32))
    store = storage.write_volume(volume, tmp_path / "out.zarr", spacing=(3.0, 0.8, 0.8))
    array, spacing = storage.open_volume(store)
    assert spacing == (3.0, 0.8, 0.8)
    np.testing.assert_allclose(np.asarray(array), volume, rtol=1e-6)


def test_pyramid_level_count_scales_with_size():
    assert storage.pyramid_levels((100, 512, 512)) == 1
    assert storage.pyramid_levels((100, 2048, 2048)) == 3
    assert storage.pyramid_levels((2607, 10240, 10240)) >= 5


# --------------------------------------------------------------------------
# destriping
# --------------------------------------------------------------------------

def cross_striped_plane(shape=(256, 256), seed=0):
    """A plane whose structures run *across* the stripe direction.

    Stripes here run along rows, so the vessels are columns. Getting this the
    wrong way round makes the test meaningless: a vessel spanning the full width
    parallel to the stripes is mathematically a stripe, and no filter can tell
    them apart.
    """
    rng = np.random.default_rng(seed)
    plane = rng.random(shape).astype(np.float32) * 20 + 100
    for offset in range(20, shape[1] - 10, 40):
        plane[:, offset:offset + 3] += 400
    return plane


def test_destripe_flattens_stripes_and_keeps_crossing_vessels():
    clean = cross_striped_plane()
    striped = add_stripes(clean, strength=0.4, axis=1)
    fixed = correct.destripe(striped, sigma=6.0, level=4, axis=1)

    def row_modulation(plane):
        # Row means are flat when nothing varies along the stripe direction.
        return float(np.std(plane.mean(axis=1)))

    assert row_modulation(clean) < 2.0, "clean plane must have flat row means"
    assert row_modulation(striped) > 5.0, "stripes must show up in row means"
    assert row_modulation(fixed) < row_modulation(striped) * 0.5

    # The crossing vessels must survive: agreement with the clean plane improves.
    before = np.corrcoef(striped.ravel(), clean.ravel())[0, 1]
    after = np.corrcoef(fixed.ravel(), clean.ravel())[0, 1]
    assert after > before

    # And the vessel columns are still clearly brighter than background.
    column_profile = fixed.mean(axis=0)
    assert column_profile[21] > column_profile[10] * 1.5


def test_destripe_attenuates_vessels_parallel_to_the_stripes():
    """The documented limitation, pinned so nobody is surprised by it.

    A vessel running the full width along the illumination axis is indistinguish-
    able from a shadow, and gets removed with them.
    """
    plane = np.full((256, 256), 100.0, dtype=np.float32)
    plane[120:123, :] += 400.0                      # a vessel parallel to stripes

    fixed = correct.destripe(plane, sigma=6.0, level=4, axis=1)
    contrast_before = plane[121].mean() - plane[10].mean()
    contrast_after = fixed[121].mean() - fixed[10].mean()
    assert contrast_after < contrast_before * 0.5


def test_destripe_clamps_an_over_deep_wavelet_level():
    """pywt only warns when the level exceeds what the plane supports."""
    import warnings
    plane = cross_striped_plane((64, 64))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fixed = correct.destripe(plane, sigma=4.0, level=12, axis=1)
    assert fixed.shape == plane.shape
    assert np.isfinite(fixed).all()


def test_destripe_validates_its_inputs():
    plane = clean_volume((1, 64, 64))[0]
    with pytest.raises(ValueError, match="single planes"):
        correct.destripe(clean_volume((4, 64, 64)), sigma=4.0)
    with pytest.raises(ValueError, match="axis"):
        correct.destripe(plane, sigma=4.0, axis=2)
    with pytest.raises(ValueError, match="sigma"):
        correct.destripe(plane, sigma=0.0)


def test_destripe_preserves_shape_and_dtype():
    plane = clean_volume((1, 96, 80))[0]
    fixed = correct.destripe(plane, sigma=5.0, level=3)
    assert fixed.shape == plane.shape
    assert fixed.dtype == np.float32


# --------------------------------------------------------------------------
# depth attenuation
# --------------------------------------------------------------------------

def test_depth_profile_tracks_the_simulated_falloff():
    clean = clean_volume((40, 64, 64))
    attenuated, falloff = add_depth_attenuation(clean, decay=0.03)

    profile = correct.depth_profile(attenuated, mask=np.ones(clean.shape, bool))
    normalised = profile / profile[0]
    np.testing.assert_allclose(normalised, falloff / falloff[0], rtol=0.05)


def test_correcting_attenuation_flattens_the_profile():
    clean = clean_volume((40, 64, 64))
    attenuated, _ = add_depth_attenuation(clean, decay=0.03)

    mask = np.ones(clean.shape, bool)
    before = correct.depth_profile(attenuated, mask=mask)
    corrected = correct.correct_depth_attenuation(attenuated, mask=mask, sigma=3.0)
    after = correct.depth_profile(corrected, mask=mask)

    # Coefficient of variation along depth should collapse.
    assert np.std(after) / np.mean(after) < np.std(before) / np.mean(before) * 0.2


def test_correction_recovers_relative_depth_signal():
    """The point of the correction: deep tissue must stop looking less vascular."""
    clean = clean_volume((40, 64, 64))
    attenuated, _ = add_depth_attenuation(clean, decay=0.04)
    corrected = correct.correct_depth_attenuation(
        attenuated, mask=np.ones(clean.shape, bool), sigma=3.0)

    shallow_ratio = corrected[:5].mean() / clean[:5].mean()
    deep_ratio = corrected[-5:].mean() / clean[-5:].mean()
    assert deep_ratio / shallow_ratio == pytest.approx(1.0, rel=0.1)


def test_median_profile_resists_vessel_density_changes():
    """A mean profile would track vessel density; the median should not."""
    volume = np.full((20, 64, 64), 100.0, dtype=np.float32)
    volume[10:, :20, :] += 500.0            # a burst of bright vessels, deeper half

    mask = np.ones(volume.shape, bool)
    median = correct.depth_profile(volume, mask=mask, statistic="median")
    mean = correct.depth_profile(volume, mask=mask, statistic="mean")

    assert np.std(median) < 1e-6, "median should be flat: background never changed"
    assert np.std(mean) > 1.0, "mean should move with vessel density"


def test_smooth_profile_interpolates_empty_planes():
    profile = np.array([10.0, np.nan, 8.0, np.nan, np.nan, 5.0])
    smoothed = correct.smooth_profile(profile, sigma=0.0)
    assert np.isfinite(smoothed).all()
    assert smoothed[1] == pytest.approx(9.0)


def test_smooth_profile_rejects_an_unusable_profile():
    with pytest.raises(ValueError, match="no usable values"):
        correct.smooth_profile(np.array([np.nan, 0.0, -1.0]))


def test_profile_measured_coarse_can_correct_fine():
    """The intended workflow: estimate on a pyramid level, apply at full res."""
    clean = clean_volume((40, 64, 64))
    attenuated, _ = add_depth_attenuation(clean, decay=0.03)
    coarse_profile = correct.depth_profile(attenuated[::2],
                                           mask=np.ones((20, 64, 64), bool))
    corrected = correct.correct_depth_attenuation(attenuated, profile=coarse_profile,
                                                  sigma=2.0)
    after = correct.depth_profile(corrected, mask=np.ones(clean.shape, bool))
    assert np.std(after) / np.mean(after) < 0.05


def test_tissue_mask_keeps_dim_deep_tissue():
    """A mask that excluded attenuated tissue would defeat the profile."""
    volume = np.zeros((30, 48, 48), np.float32)
    volume[:, 8:40, 8:40] = 100.0
    volume[20:, 8:40, 8:40] = 30.0             # deep tissue, much dimmer
    mask = correct.tissue_mask(volume, percentile=25)
    assert mask[25, 24, 24], "deep dim tissue must remain inside the mask"
    assert not mask[25, 0, 0]


def test_depth_profile_rejects_mismatched_mask():
    with pytest.raises(ValueError, match="mask shape"):
        correct.depth_profile(np.ones((4, 8, 8)), mask=np.ones((4, 8, 9), bool))


# --------------------------------------------------------------------------
# chunked processing
# --------------------------------------------------------------------------

def test_gaussian_reach_accounts_for_anisotropy():
    reach = chunked.gaussian_reach(6.0, spacing=(3.0, 0.75, 0.75))
    assert reach == (8, 32, 32)


def test_overlap_depth_uses_the_largest_sigma_plus_extra():
    depth = chunked.overlap_depth([1.5, 3.0, 6.0], spacing=(3.0, 1.5, 1.5), extra=2)
    assert depth == (10, 18, 18)
    with pytest.raises(ValueError):
        chunked.overlap_depth([])


def test_halo_too_large_for_chunks_is_refused():
    import dask.array as da
    array = da.zeros((32, 32, 32), chunks=(8, 8, 8))
    with pytest.raises(ValueError, match="too large for chunks"):
        chunked.map_blocks_with_halo(lambda b: b, array, depth=(6, 6, 6))


def test_chunked_filtering_matches_whole_volume_filtering():
    """The seam test: chunked output must equal the unchunked result.

    This is what the halo exists for. Run it whole, run it in small chunks, and
    require agreement — a chunk boundary must not be visible in the output.
    """
    import dask.array as da
    from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue

    volume = clean_volume((32, 96, 96), seed=5)
    spacing = (3.0, 1.0, 1.0)
    sigmas = [2.0, 4.0]
    reference = max_eigenvalue(volume, sigmas, spacing)

    whole = jerman_vesselness(volume, sigmas, spacing, reference_lambda=reference)

    array = da.from_array(volume, chunks=(16, 48, 48))
    lazy = chunked.apply_vesselness(array, sigmas, spacing, reference_lambda=reference)
    tiled = np.asarray(lazy)

    assert tiled.shape == whole.shape
    np.testing.assert_allclose(tiled, whole, atol=1e-4)


def test_chunked_vesselness_requires_a_fixed_reference():
    """Without it each block regularises against its own maximum: real seams."""
    import dask.array as da
    array = da.from_array(clean_volume((24, 64, 64)), chunks=(12, 32, 32))
    with pytest.raises(ValueError, match="reference_lambda is required"):
        chunked.apply_vesselness(array, [2.0, 4.0], (3.0, 1.0, 1.0))


def test_chunked_result_is_lazy_until_requested():
    import dask.array as da
    array = da.from_array(clean_volume((24, 64, 64)), chunks=(12, 32, 32))
    lazy = chunked.apply_vesselness(array, [2.0], (3.0, 1.0, 1.0),
                                    reference_lambda=1.0)
    assert isinstance(lazy, da.Array)
    assert lazy.shape == (24, 64, 64)


# --------------------------------------------------------------------------
# the stages compose
# --------------------------------------------------------------------------

def test_full_stage_zero_to_two_runs_end_to_end(tmp_path):
    """Convert, correct, segment — on a simulated acquisition with both artefacts."""
    import dask.array as da
    from vessel_utils.vesselness import max_eigenvalue

    clean = clean_volume((32, 96, 96), seed=7)
    attenuated, _ = add_depth_attenuation(clean, decay=0.03)
    acquired = np.stack([add_stripes(plane, strength=0.3, axis=1, seed=i)
                         for i, plane in enumerate(attenuated)])

    pattern = write_plane_series(tmp_path / "raw", acquired)
    store = storage.plane_series_to_zarr(pattern, tmp_path / "vol.zarr",
                                         spacing=(3.0, 1.0, 1.0), chunks=(16, 48, 48))
    array, spacing = storage.open_volume(store)

    destriped = np.stack([correct.destripe(plane, sigma=5.0, level=3, axis=1)
                          for plane in np.asarray(array)])
    corrected = correct.correct_depth_attenuation(
        destriped, mask=correct.tissue_mask(destriped), sigma=3.0)

    reference = max_eigenvalue(corrected, [2.0, 4.0], spacing)
    response = chunked.apply_vesselness(
        da.from_array(corrected, chunks=(16, 48, 48)), [2.0, 4.0], spacing,
        reference_lambda=reference)
    result = np.asarray(response)

    assert result.shape == clean.shape
    assert 0.0 <= result.min() and result.max() <= 1.0 + 1e-6
    assert result.max() > 0.5, "the tubes should produce a strong response"
