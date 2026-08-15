"""Tests for the tuning viewer's pure half.

The Qt wiring is exercised by a human on every launch and is not tested here.
What is tested is the pipeline underneath it and the guards around loading,
because those are what could produce a confident wrong number rather than a
visible error.

Everything here runs from a clone: `vessel_utils.synth.phantom` supplies images
with exact ground truth, so no lab share is needed. Same reason as
tests/test_script_helpers.py.
"""

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
