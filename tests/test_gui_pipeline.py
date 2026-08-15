"""Tests for the tuning viewer's pure half.

The Qt wiring is exercised by a human on every launch and is not tested here.
What is tested is the pipeline underneath it and the guards around loading,
because those are what could produce a confident wrong number rather than a
visible error.

Everything here runs from a clone: `vessel_utils.synth.phantom` supplies images
with exact ground truth, so no lab share is needed. Same reason as
tests/test_script_helpers.py.
"""

from pathlib import Path

import numpy as np
import pytest

from vessel_utils import gui
from vessel_utils.synth import phantom


SPACING_2D = (0.75, 0.75)


@pytest.fixture(scope="module")
def slice_2d():
    """A 2D slice of a phantom, with its ground-truth mask and a tissue ROI.

    Module-scoped: the phantom is the slowest thing in this file and it is
    deterministic, so building it once is safe.
    """
    volume, truth, _ = phantom(shape=(24, 160, 160), spacing=(3.0, 0.75, 0.75), seed=7)
    index = int(np.argmax(truth.reshape(truth.shape[0], -1).sum(axis=1)))
    image = volume[index].astype(np.float32)
    return image, truth[index], np.ones_like(image, dtype=bool)


def test_normalise_maps_roi_median_to_zero(slice_2d):
    image, _, roi = slice_2d
    out = gui.normalise(image, roi)
    assert out.dtype == np.float32
    assert out.min() == 0.0                       # clipped at the median
    assert np.median(out) == pytest.approx(0.0, abs=1e-6)


def test_stages_finds_the_phantom_vessels(slice_2d):
    image, truth, roi = slice_2d
    from vessel_utils import metrics

    st = gui.stages(image, None, SPACING_2D, tissue=roi, sigmas=(1.5, 3.0, 6.0),
                    reference=2.0, ref_low=0.03, ref_high=0.09,
                    test_low=0.03, test_high=0.09, min_vessel_um2=6.0,
                    virus_k=3.0, q=10.0)

    assert st["ref_response"].shape == image.shape
    assert st["ref_response"].min() >= 0.0 and st["ref_response"].max() <= 1.0
    assert st["ref_vessels"].dtype == bool
    assert st["test_vessels"] is None              # single-channel input
    # A correctly wired pipeline finds the phantom. A miswired one still runs.
    assert metrics.dice(st["ref_vessels"], truth) > 0.3


def test_stages_masks_everything_to_the_roi(slice_2d):
    image, _, _ = slice_2d
    roi = np.zeros(image.shape, dtype=bool)
    roi[40:120, 40:120] = True

    st = gui.stages(image, image, SPACING_2D, tissue=roi, sigmas=(1.5, 3.0),
                    reference=2.0, ref_low=0.03, ref_high=0.09,
                    test_low=0.03, test_high=0.09, min_vessel_um2=6.0,
                    virus_k=3.0, q=10.0)

    for key in ("ref_vessels", "test_vessels", "test_positive", "ref_top_q"):
        assert not st[key][~roi].any(), f"{key} leaked outside the ROI"


def test_min_vessel_is_physical_not_pixels(slice_2d):
    """The same um^2 must mean fewer pixels at coarser spacing."""
    image, _, roi = slice_2d
    common = dict(tissue=roi, sigmas=(1.5, 3.0), reference=2.0, ref_low=0.03,
                  ref_high=0.09, test_low=0.03, test_high=0.09,
                  min_vessel_um2=50.0, virus_k=3.0, q=10.0)
    fine = gui.stages(image, None, (0.25, 0.25), **common)["ref_vessels"]
    coarse = gui.stages(image, None, (2.0, 2.0), **common)["ref_vessels"]
    # Not an equality claim - just that spacing reaches the size filter at all.
    assert fine.sum() != coarse.sum()


# --------------------------------------------------------------------------
# reference_lambda
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def three_slices():
    """Three 2D slices from different phantoms - a miniature 'dataset'."""
    out = []
    for seed in (1, 2, 3):
        volume, truth, _ = phantom(shape=(12, 96, 96), spacing=(3.0, 0.75, 0.75),
                                   seed=seed)
        index = int(np.argmax(truth.reshape(truth.shape[0], -1).sum(axis=1)))
        out.append(volume[index].astype(np.float32))
    return out


def test_reference_lambda_is_the_median_of_its_sample(three_slices):
    from vessel_utils.vesselness import max_eigenvalue
    sigmas, spacing = (1.5, 3.0), (0.75, 0.75)
    masks = [np.ones(i.shape, dtype=bool) for i in three_slices]

    each = [max_eigenvalue(gui.normalise(i, m), list(sigmas), spacing,
                           percentile=99.9, mask=m)
            for i, m in zip(three_slices, masks)]
    got = gui.reference_lambda(three_slices, spacing, sigmas, masks=masks)

    assert got == pytest.approx(float(np.median(each)))


def test_reference_lambda_ignores_order(three_slices):
    sigmas, spacing = (1.5, 3.0), (0.75, 0.75)
    forward = gui.reference_lambda(three_slices, spacing, sigmas)
    backward = gui.reference_lambda(list(reversed(three_slices)), spacing, sigmas)
    assert forward == pytest.approx(backward)


def test_reference_lambda_survives_a_duplicated_image(three_slices):
    """The property that makes it a dataset constant, not a sample artefact.

    A median over an odd sample and the same sample plus one duplicate of its
    middle element is unchanged. If this breaks, the statistic is tracking which
    files happened to be sampled.
    """
    sigmas, spacing = (1.5, 3.0), (0.75, 0.75)
    base = gui.reference_lambda(three_slices, spacing, sigmas)
    each = sorted(gui.reference_lambda([i], spacing, sigmas) for i in three_slices)
    middle = next(i for i in three_slices
                  if gui.reference_lambda([i], spacing, sigmas) == pytest.approx(each[1]))
    assert gui.reference_lambda(three_slices + [middle], spacing, sigmas) \
        == pytest.approx(base)


def test_reference_lambda_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="no images"):
        gui.reference_lambda([], (0.75, 0.75), (1.5, 3.0))


# --------------------------------------------------------------------------
# tissue_mask dispatch
# --------------------------------------------------------------------------

def test_mask_methods_are_the_documented_four():
    assert gui.MASK_METHODS == ("none", "otsu", "grabcut", "brain")


def test_mask_none_is_everything(slice_2d):
    image, _, _ = slice_2d
    mask = gui.tissue_mask([image], "none")
    assert mask.dtype == bool
    assert mask.all()


def test_mask_otsu_separates_signal_from_background():
    """A bright square on a dark field: Otsu must find the square, not the field."""
    image = np.full((64, 64), 10.0, dtype=np.float32)
    image[16:48, 16:48] = 200.0
    mask = gui.tissue_mask([image], "otsu")
    assert mask[16:48, 16:48].all()
    assert not mask[:8, :8].any()


def test_mask_otsu_works_on_uint16():
    """dtype-agnostic is the whole reason otsu is the generic default."""
    image = np.full((64, 64), 10, dtype=np.uint16)
    image[16:48, 16:48] = 40000
    assert gui.tissue_mask([image], "otsu")[16:48, 16:48].all()


def test_mask_sums_the_channels():
    """Tissue is whatever is bright in EITHER channel."""
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.zeros((64, 64), dtype=np.float32)
    a[8:24, 8:24] = 500.0
    b[40:56, 40:56] = 500.0
    mask = gui.tissue_mask([a, b], "otsu")
    assert mask[8:24, 8:24].all() and mask[40:56, 40:56].all()


def test_mask_rejects_an_unknown_method(slice_2d):
    image, _, _ = slice_2d
    with pytest.raises(ValueError, match="unknown tissue-mask method"):
        gui.tissue_mask([image], "magic")


def test_mask_brain_rejects_non_8bit():
    """get_brain_mask calls cv2 THRESH_TRIANGLE, which is 8-bit only.

    Refusing beats a cv2 error from four frames down, and beats silently
    converting - a converted copy is the caller's decision, not ours.
    """
    with pytest.raises(ValueError, match="8-bit"):
        gui.tissue_mask([np.zeros((32, 32), dtype=np.uint16)], "brain")


# --------------------------------------------------------------------------
# discovery, spacing, and the channel-count guard
# --------------------------------------------------------------------------

import tifffile


def _write(path, array, resolution=None):
    kwargs = {"resolution": resolution, "resolutionunit": "CENTIMETER"} \
        if resolution else {}
    tifffile.imwrite(path, array, **kwargs)
    return path


def test_find_images_is_sorted_and_case_insensitive(tmp_path):
    for name in ("b.tif", "a.TIF", "c.tiff", "notes.txt"):
        if name.endswith(".txt"):
            (tmp_path / name).write_text("x")
        else:
            _write(tmp_path / name, np.zeros((8, 8), dtype=np.uint16))
    assert [p.name for p in gui.find_images(tmp_path)] == ["a.TIF", "b.tif", "c.tiff"]


def test_find_images_reports_an_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no TIFF"):
        gui.find_images(tmp_path)


def test_channel_count_reads_the_header(tmp_path):
    one = _write(tmp_path / "one.tif", np.zeros((16, 16), dtype=np.uint16))
    two = _write(tmp_path / "two.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    three = _write(tmp_path / "three.tif", np.zeros((3, 16, 16), dtype=np.uint16))
    assert gui.channel_count(one) == 1
    assert gui.channel_count(two) == 2
    assert gui.channel_count(three) == 3


def test_read_channels_returns_the_requested_roles(tmp_path):
    stack = np.zeros((2, 16, 16), dtype=np.uint16)
    stack[0] = 11
    stack[1] = 22
    path = _write(tmp_path / "pair.tif", stack)
    ref, test = gui.read_channels(path, (1, 0))
    assert ref[0, 0] == 22 and test[0, 0] == 11


def test_read_channels_allows_single_channel_mode(tmp_path):
    path = _write(tmp_path / "solo.tif", np.full((16, 16), 5, dtype=np.uint16))
    ref, test = gui.read_channels(path, (0, None))
    assert ref[0, 0] == 5 and test is None


def test_read_channels_refuses_a_role_the_file_cannot_supply(tmp_path):
    """THE guard. Never re-index, never fall back to [0]/[1]."""
    path = _write(tmp_path / "solo.tif", np.zeros((16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="has 1 channel"):
        gui.read_channels(path, (1, 0))


def test_read_channels_refuses_an_uncurated_extra_channel_file(tmp_path):
    """A 3-channel file with roles (1, 0) is exactly the silent-wrong-pair case."""
    path = _write(tmp_path / "trio.tif", np.zeros((3, 16, 16), dtype=np.uint16))
    with pytest.raises(ValueError, match="3 channels"):
        gui.read_channels(path, (1, 0))


def test_read_spacing_from_tags(tmp_path):
    # 10000 px/cm == 1 um/px
    path = _write(tmp_path / "cal.tif", np.zeros((16, 16), dtype=np.uint16),
                  resolution=(10000, 10000))
    assert gui.read_spacing(path) == pytest.approx((1.0, 1.0))


def test_read_spacing_returns_none_when_absent(tmp_path):
    path = _write(tmp_path / "raw.tif", np.zeros((16, 16), dtype=np.uint16))
    assert gui.read_spacing(path) is None


# --------------------------------------------------------------------------
# readout
# --------------------------------------------------------------------------

def _stages_for_readout(slice_2d, **overrides):
    image, _, roi = slice_2d
    kwargs = dict(tissue=roi, sigmas=(1.5, 3.0), reference=2.0, ref_low=0.03,
                  ref_high=0.09, test_low=0.03, test_high=0.09,
                  min_vessel_um2=6.0, virus_k=3.0, q=10.0)
    kwargs.update(overrides)
    return gui.stages(image, image, SPACING_2D, **kwargs), kwargs


def test_readout_matches_direct_metric_calls(slice_2d):
    from vessel_utils import metrics
    st, _ = _stages_for_readout(slice_2d)
    got = gui.readout(st, SPACING_2D, q=10.0)

    assert got["dice"] == pytest.approx(
        metrics.dice(st["test_vessels"], st["ref_vessels"]))
    assert got["precision"] == pytest.approx(
        metrics.precision(st["test_vessels"], st["ref_vessels"]))
    assert got["recall"] == pytest.approx(
        metrics.recall(st["test_vessels"], st["ref_vessels"]))
    assert got["ref_af"] == pytest.approx(
        metrics.area_fraction(st["ref_vessels"], st["tissue"]))


def test_readout_identical_channels_score_one(slice_2d):
    """Same image in both roles with the same thresholds: Dice must be 1."""
    st, _ = _stages_for_readout(slice_2d)
    got = gui.readout(st, SPACING_2D, q=10.0)
    assert got["dice"] == pytest.approx(1.0, abs=1e-6)
    assert got["jaccard"] == pytest.approx(1.0, abs=1e-6)


def test_readout_skips_cl_dice_unless_asked(slice_2d):
    st, _ = _stages_for_readout(slice_2d)
    assert gui.readout(st, SPACING_2D, q=10.0)["cl_dice"] is None
    assert gui.readout(st, SPACING_2D, q=10.0, include_cl_dice=True)["cl_dice"] \
        == pytest.approx(1.0, abs=1e-6)


def test_readout_flags_an_implausible_area_fraction(slice_2d):
    st, _ = _stages_for_readout(slice_2d)
    warnings = gui.readout(st, SPACING_2D, q=10.0, plausible=(0.0, 1e-9))["warnings"]
    assert any("ref_af" in w and "plausible" in w for w in warnings)
    assert not gui.readout(st, SPACING_2D, q=10.0,
                           plausible=(0.0, 1.0))["warnings"]


def test_readout_single_channel_leaves_pairwise_entries_none(slice_2d):
    image, _, roi = slice_2d
    st = gui.stages(image, None, SPACING_2D, tissue=roi, sigmas=(1.5, 3.0),
                    reference=2.0, ref_low=0.03, ref_high=0.09, test_low=0.03,
                    test_high=0.09, min_vessel_um2=6.0, virus_k=3.0, q=10.0)
    got = gui.readout(st, SPACING_2D, q=10.0)
    assert got["ref_af"] is not None
    for key in ("dice", "precision", "recall", "enrichment", "coverage"):
        assert got[key] is None


def test_readout_guards_a_tied_percentile_selection(slice_2d):
    """A plateau of repeated values can select far more than q% of the tissue.

    That silently collapses several q onto one selection and fakes stability
    across q, so it must read as absent rather than as a number.
    """
    st, _ = _stages_for_readout(slice_2d)
    st = dict(st)
    st["ref_top_q"] = st["tissue"].copy()          # 100% selected, nominal q=10
    assert gui.readout(st, SPACING_2D, q=10.0)["enrichment_q"] is None


# --------------------------------------------------------------------------
# presets and caching
# --------------------------------------------------------------------------

def test_presets_carry_the_documented_values():
    shipped = gui.PRESETS["spinal-cord shipped"]
    assert shipped["spacing"] == 0.650193
    assert shipped["roles"] == (1, 0)
    assert shipped["mask"] == "grabcut"
    assert shipped["reference"] == 2.0
    assert shipped["ref_thr"] == (0.03, 0.09)
    assert shipped["test_thr"] == (0.04, 0.12)
    assert shipped["plausible"] == (0.01, 0.10)

    superseded = gui.PRESETS["spinal-cord superseded"]
    assert superseded["reference"] is None            # calibrated, not fixed
    assert superseded["ref_thr"] == (0.02, 0.15)
    assert superseded["test_thr"] == (0.02, 0.15)

    assert gui.PRESETS["generic"]["mask"] == "otsu"   # fast + dtype-agnostic
    assert gui.PRESETS["generic"]["plausible"] == (0.0, 1.0)


def test_brain_slice_preset_is_labelled_a_starting_point():
    """It reproduces no published figure - those came from the archive pipeline."""
    assert "starting point" in " ".join(gui.PRESETS).lower()


def test_every_preset_names_a_real_mask_method():
    for name, preset in gui.PRESETS.items():
        assert preset["mask"] in gui.MASK_METHODS, name


def test_load_is_cached_by_its_arguments(tmp_path):
    stack = np.zeros((2, 32, 32), dtype=np.uint16)
    stack[:, 8:24, 8:24] = 4000
    path = _write(tmp_path / "pair.tif", stack)

    gui.load.cache_clear()
    first = gui.load(path, "otsu", (1, 0))
    assert gui.load.cache_info().misses == 1
    second = gui.load(path, "otsu", (1, 0))
    assert gui.load.cache_info().hits == 1
    assert first[1] is second[1]                       # same object, not a copy

    gui.load(path, "none", (1, 0))                     # different method -> miss
    assert gui.load.cache_info().misses == 2


def test_response_is_cached_and_reference_invalidates_it(tmp_path):
    stack = np.zeros((2, 48, 48), dtype=np.uint16)
    stack[:, 20:28, 8:40] = 5000
    path = _write(tmp_path / "bar.tif", stack)

    gui.response.cache_clear()
    common = (path, 0, (1.5, 3.0), (0.75, 0.75))
    gui.response(*common, 2.0, "none", (1, 0))
    gui.response(*common, 2.0, "none", (1, 0))
    assert gui.response.cache_info().hits == 1
    gui.response(*common, 5.0, "none", (1, 0))         # new reference -> miss
    assert gui.response.cache_info().misses == 2


def test_response_rejects_unhashable_arguments(tmp_path):
    path = _write(tmp_path / "x.tif", np.zeros((2, 16, 16), dtype=np.uint16))
    with pytest.raises(TypeError):
        gui.response(path, 0, [1.5, 3.0], (0.75, 0.75), 2.0, "none", (1, 0))


def test_selftest_passes():
    gui.selftest()


# --------------------------------------------------------------------------
# batch
# --------------------------------------------------------------------------

def _two_channel_dir(tmp_path, n=3):
    paths = []
    for i in range(n):
        stack = np.zeros((2, 48, 48), dtype=np.uint16)
        stack[:, 18 + i:26 + i, 6:42] = 5000
        paths.append(_write(tmp_path / f"s{i}.tif", stack))
    return paths


BATCH_PARAMS = dict(mask="none", sigmas=(1.5, 3.0), reference=2.0, ref_low=0.03,
                    ref_high=0.09, test_low=0.03, test_high=0.09,
                    min_vessel_um2=6.0, virus_k=3.0, q=10.0, cl_dice=False,
                    spacing=(0.75, 0.75))


def test_batch_rows_one_row_per_file(tmp_path):
    paths = _two_channel_dir(tmp_path)
    rows, failures = gui.batch_rows(paths, gui.PRESETS["generic"], BATCH_PARAMS)
    assert len(rows) == 3 and not failures
    assert rows[0]["file"] == "s0.tif"
    for key in ("ref_af", "dice", "precision", "recall"):
        assert key in rows[0]


def test_batch_rows_collects_failures_instead_of_raising(tmp_path):
    paths = _two_channel_dir(tmp_path, n=2)
    broken = _write(tmp_path / "solo.tif", np.zeros((16, 16), dtype=np.uint16))
    rows, failures = gui.batch_rows(paths + [broken], gui.PRESETS["generic"],
                                   BATCH_PARAMS)
    assert len(rows) == 2
    assert len(failures) == 1 and "solo.tif" in failures[0]


def test_batch_csv_name_never_collides(tmp_path):
    existing = {"spinal_cord_specificity.csv", "dice_between_channels_full.csv",
                "dice_between_channels_pilot.csv"}
    for name in existing:
        (tmp_path / name).write_text("x")
    first = gui.batch_csv_path(tmp_path, "spinal-cord shipped")
    first.write_text("x")
    second = gui.batch_csv_path(tmp_path, "spinal-cord shipped")
    assert first.name not in existing and second.name not in existing
    assert first != second                       # never overwrites


# --------------------------------------------------------------------------
# $BVP_DATA override (independent of the viewer)
# --------------------------------------------------------------------------

def test_analyse_spinal_cord_data_honours_bvp_data(tmp_path, monkeypatch):
    """The lab share is not mounted on most machines; the override is the way in."""
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    monkeypatch.setenv("BVP_DATA", str(tmp_path))
    module = importlib.reload(importlib.import_module("analyse_spinal_cord"))
    assert module.DATA == tmp_path


def test_analyse_spinal_cord_data_defaults_to_the_share(monkeypatch):
    """Unset means unchanged behaviour - every existing script must be unaffected."""
    import importlib
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    monkeypatch.delenv("BVP_DATA", raising=False)
    module = importlib.reload(importlib.import_module("analyse_spinal_cord"))
    assert str(module.DATA) == r"Z:\Lab\Eric V\BEC Spinal Cords\composites_EV"
