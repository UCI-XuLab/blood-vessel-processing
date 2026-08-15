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
