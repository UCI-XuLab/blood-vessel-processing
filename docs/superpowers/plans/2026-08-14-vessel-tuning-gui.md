# Vessel Tuning Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A napari viewer that runs the `vessel_utils` segmentation pipeline live on any 2D vessel image, shows every intermediate as a toggleable layer, and lets the parameters be tuned with the result on screen.

**Architecture:** One new module, `vessel_utils/gui.py`, built in two halves. The lower half is pure functions over numpy arrays — tissue masking, vesselness, thresholding, metrics — testable with no Qt and no lab data, because `vessel_utils.synth.phantom` supplies images with exact ground truth. The upper half wires those to napari layers and a magicgui panel, with two `functools.lru_cache`s between them so cheap knobs update on drag while expensive ones sit behind a Recompute button. A dataset is a preset: a dict of numbers plus two enum choices. There are no adapter classes.

**Tech Stack:** Python 3.9+, napari 0.8 + PyQt6 (via `napari[all]`), magicgui, numpy, scikit-image, tifffile, opencv-python. All already present except the napari stack.

**Spec:** [docs/design/2026-08-14-vessel-tuning-gui.md](../../design/2026-08-14-vessel-tuning-gui.md) — read it alongside this plan; the plan argues from it.

## Global Constraints

- **Never modify `velazquez_rivera_2025/`.** It is frozen and `tests/test_archive_frozen.py` fails if any file in it changes. Importing from it is fine.
- **Never add an import to `vessel_utils/__init__.py`.** It must import nothing; `tests/test_vessel_utils.py::test_package_init_stays_empty` pins this. Editing its *docstring* is fine.
- **`vessel_utils/` must never import from `scripts/`.** A package importing an ad-hoc script directory is backwards. Where a 2–3 line rule is needed from there, reimplement it and add a comment naming the original so the two stay in step.
- **No new dependencies beyond `napari[all]`.** `zarr`, `dask`, `PyWavelets` and `tqdm` were removed from the runtime deps in #37. Do not reintroduce them. CLAUDE.md: *"Do not re-add a module here speculatively — the last round cost ~2,000 lines and three dependencies carrying nothing."*
- **Do not re-add any deleted `vessel_utils` module** (`storage`, `chunked`, `correct`, `qc`, `validate`, `ensemble`, `sweep`).
- **`napari` and `magicgui` are imported inside the functions that need them**, never at module top level, so `import vessel_utils.gui` stays cheap and the pure half is testable without Qt.
- **`reference_lambda` is per-dataset, never per-image.** Computed once per `(dataset, sigmas)`, then editable. Recomputing per image would make a fixed threshold mean a different thing in every image.
- **Vesselness response layers get `contrast_limits=(0.0, 1.0)`, pinned.** Never auto-scaled. Jerman saturates above `tau * reference_lambda / 2`, and auto-scaling hides that, which is how a saturated operating point survived in this repo.
- **Colour convention: reference = magenta, test = green.** Matches the merge views, notebooks and handoff.
- **`min_vessel` is entered in µm² and converted with `prod(spacing)`.** Never in pixels. `spacing` is a per-axis tuple throughout, never a scalar, so 3D is an addition rather than a rewrite.
- **Output CSV filename must not collide** with `spinal_cord_specificity.csv`, `dice_between_channels_full.csv`, `dice_between_channels_pilot.csv`, or `enrichment_cd31_percentile_*.csv`.
- **Exact preset values, copied verbatim from the spec:**
  - `generic`: spacing from TIFF tags, roles `(1, 0)`, mask `otsu`, reference calibrated, both thresholds `(0.03, 0.09)`, `min_vessel_um2=6.0`, plausible `(0.0, 1.0)`
  - `spinal-cord shipped`: spacing `0.650193`, roles `(1, 0)`, mask `grabcut`, reference `2.0`, ref `(0.03, 0.09)`, test `(0.04, 0.12)`, `min_vessel_um2=6.0`, plausible `(0.01, 0.10)`
  - `spinal-cord superseded`: as shipped, but reference calibrated and both thresholds `(0.02, 0.15)`
  - `brain-slice starting point`: spacing `1.0`, mask `brain`, reference calibrated, both `(0.03, 0.09)`, plausible `(0.0, 1.0)`. **Labelled "starting point" in the UI** — it reproduces no published figure.
- `virus_k` default `3.0`; `q` default `10.0`; `sigmas` default `(1.5, 3.0, 6.0, 12.0)` µm.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `vessel_utils/gui.py` | **Create.** Everything. Sections in order: presets, image discovery and loading, the pure pipeline, the metrics readout, the caches, the napari/magicgui wiring, batch CSV, `main()`. Kept as one file because the spec sizes it at ~330 lines; if it passes ~450 during implementation, split the pure half into `vessel_utils/tuning.py` and leave `gui.py` as the Qt wiring only. |
| `tests/test_gui_pipeline.py` | **Create.** Covers the pure half only, on `synth.phantom` and synthetic TIFFs in `tmp_path`. No Qt, no lab data — following `tests/test_script_helpers.py`. |
| `vessel_utils/__init__.py` | **Modify.** One line in the docstring module list. |
| `CLAUDE.md` | **Modify.** One row in the `vessel_utils` table. |
| `pyproject.toml` | **Modify.** `gui` extra + `[project.scripts]`. |
| `scripts/analyse_spinal_cord.py` | **Modify.** One line, `DATA` reads `$BVP_DATA`. Independent of the GUI. |

---

### Task 1: The pure pipeline core

The single function every later task builds on: arrays in, arrays out, no Qt, no caching, no widgets.

**Files:**
- Create: `vessel_utils/gui.py`
- Create: `tests/test_gui_pipeline.py`

**Interfaces:**
- Consumes: `vessel_utils.vesselness.jerman_vesselness`, `vessel_utils.threshold.segment`, `vessel_utils.synth.phantom`.
- Produces:
  - `normalise(channel, roi) -> np.ndarray[float32]`
  - `stages(ref, test, spacing, *, tissue, sigmas, reference, ref_low, ref_high, test_low, test_high, min_vessel_um2, virus_k, q) -> dict` with keys `ref_response`, `test_response`, `ref_vessels`, `test_vessels`, `test_positive`, `ref_top_q`, `cut`, `tissue`. `test`, and therefore `test_*`, may be `None` for single-channel input.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vessel_utils.gui'`

- [ ] **Step 3: Write the minimal implementation**

Create `vessel_utils/gui.py`:

```python
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

__all__ = ["normalise", "stages", "main"]


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 4 passed. If `test_stages_finds_the_phantom_vessels` fails on the Dice
floor, print `metrics.dice(...)` and check the phantom slice actually contains
vessels before loosening the floor — a floor that passes on an empty mask tests
nothing.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: the pure pipeline core"
```

---

### Task 2: `reference_lambda` — the per-dataset calibration

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Consumes: `vessel_utils.vesselness.max_eigenvalue`, `gui.normalise`.
- Produces: `reference_lambda(images, spacing, sigmas, masks=None, percentile=99.9) -> float`

Why it lives here and not in `vessel_utils.vesselness`: one caller, and
`vessel_utils/__init__.py`'s docstring already prescribes the recipe as a
one-liner. CLAUDE.md forbids speculative additions to that package, and the sin
it names for the seven modules deleted in #37 was that every caller was its own
test. It graduates to `vesselness.py` if a second caller appears.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k reference_lambda -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'reference_lambda'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py`, after `normalise`, and add `"reference_lambda"` to `__all__`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: per-dataset reference_lambda calibration"
```

---

### Task 3: Tissue-mask dispatch

Four options, all reusing an existing implementation. `otsu` is the default for
`generic` because it is the only one that is both fast and dtype-agnostic:
GrabCut is 7–15 s per image, and `brain` needs 8-bit input.

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Consumes: `vessel_utils.threshold.otsu_threshold`, `vessel_utils._vendor.compute_entropy_grabcut`, `velazquez_rivera_2025.vessels.get_brain_mask`.
- Produces: `MASK_METHODS: tuple[str, ...]` = `("none", "otsu", "grabcut", "brain")`, and `tissue_mask(channels, method) -> np.ndarray[bool]`, where `channels` is a sequence of 2D arrays (masking runs on their sum, so a section torn or dim in one channel still masks).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k mask -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'MASK_METHODS'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py` and extend `__all__` with `"MASK_METHODS"`, `"tissue_mask"`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: tissue-mask dispatch over four existing maskers"
```

---

### Task 4: Image discovery, spacing detection, and the channel-count guard

The guard is the point of this task. Globbing a directory reintroduces a hazard
the spinal-cord curation handles today: `curated_paths` excludes files with more
than two channels outright rather than taking the first two, because "which two
are green and CD31 is not established, and guessing would silently compare the
wrong pair". A viewer that quietly fell back to `[0]`/`[1]` would emit a
confident, wrong Dice.

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Consumes: `tifffile`.
- Produces:
  - `find_images(directory) -> list[Path]` — sorted `*.tif`/`*.tiff`, case-insensitive.
  - `channel_count(path) -> int` — from the header, without decoding.
  - `read_spacing(path, ndim=2) -> tuple[float, ...] | None` — µm per pixel from the TIFF tags, `None` when absent.
  - `read_channels(path, roles) -> tuple[np.ndarray, np.ndarray | None]` — raises `ValueError` when `roles` exceeds the file's channel count. `roles` is `(reference_index, test_index)`; `test_index` may be `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k "find_images or channel_count or read_channels or read_spacing" -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'find_images'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py`, extending `__all__` with `"find_images"`,
`"channel_count"`, `"read_spacing"`, `"read_channels"`:

```python
from pathlib import Path

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
    """Channels, from the header, without decoding the image."""
    import tifffile
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        return int(series.shape[0]) if series.axes.startswith("C") else 1


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
    if any(i >= count for i in wanted):
        plural = "channel" if count == 1 else "channels"
        raise ValueError(
            f"{Path(path).name} has {count} {plural}, so role indices {tuple(wanted)} "
            f"cannot be read. Channel layout is not guessable - pick indices this "
            f"file actually has, or exclude it.")

    stack = tifffile.imread(path)
    def plane(index):
        return np.asarray(stack if count == 1 else stack[index], dtype=np.float32)
    return plane(roles[0]), (None if roles[1] is None else plane(roles[1]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: image discovery, spacing tags, channel-role guard"
```

---

### Task 5: The metrics readout

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Consumes: `vessel_utils.metrics`, `gui.stages`.
- Produces: `readout(st, spacing, *, q, plausible=(0.0, 1.0), include_cl_dice=False) -> dict` with float values plus a `warnings` list of strings. Keys: `ref_af`, `test_af`, `dice`, `jaccard`, `precision`, `recall`, `cl_dice`, `enrichment`, `coverage`, `off_target`, `enrichment_q`, `cut`, `warnings`. Pairwise entries are `None` in single-channel mode.

Argument order for the asymmetric metrics is fixed and stated: `precision(test, ref)` asks how much of the *test* mask is on a reference vessel; `recall(test, ref)` asks how much of the reference is covered. Do not swap them.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k readout -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'readout'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py`, extending `__all__` with `"readout"`:

```python
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
```

Then add one line to `stages`, immediately before `return out` in the two-channel
branch, so the readout can compute intensity ratios without re-reading the file:

```python
    out["test_intensity"] = np.asarray(test, dtype=np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 30 passed.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: metrics readout with the plausible-band and tie guards"
```

---

### Task 6: Presets, the caches, and `--selftest`

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Produces:
  - `PRESETS: dict[str, dict]` — exactly the values in Global Constraints.
  - `load(path, mask_method, roles) -> tuple[tuple[np.ndarray, ...], np.ndarray]` — `lru_cache(maxsize=8)`.
  - `response(path, role_index, sigmas, spacing, reference, mask_method, roles) -> np.ndarray` — `lru_cache(maxsize=8)`.
  - `selftest() -> None` — raises `AssertionError` on failure.

`lru_cache` needs hashable arguments, so `sigmas`, `spacing` and `roles` are
tuples at every call site. Passing a list will raise `TypeError: unhashable`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k "preset or cache or selftest" -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'PRESETS'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py`, extending `__all__` with `"PRESETS"`, `"load"`,
`"response"`, `"selftest"`:

```python
from functools import lru_cache

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: presets, response caching, headless selftest"
```

---

### Task 7: napari wiring, the parameter panel, and the entry point

The first task whose deliverable is verified by looking at it. Everything below
it is already tested; this is the part a human exercises on every launch.

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `build_viewer(directory, preset_name) -> napari.Viewer`, `main(argv=None) -> int`.

- [ ] **Step 1: Add the dependency and the entry point**

In `pyproject.toml`, add to `[project.optional-dependencies]` — **not** to `dev`,
so `pip install -e ".[dev]"` keeps the test environment free of Qt and ~45
packages:

```toml
# The interactive tuning viewer. Deliberately not in `dev`: napari pulls PyQt6 and
# about 45 packages, and neither the tests nor the notebooks need any of it.
gui = ["napari[all]"]
```

And a new table at the end of the file:

```toml
[project.scripts]
bvp-tune = "vessel_utils.gui:main"
```

- [ ] **Step 2: Install it and confirm it resolves**

Run: `.venv\Scripts\python.exe -m pip install -e ".[gui]"`
Expected: installs `napari` 0.8.x, `PyQt6`, `magicgui`, `vispy`, `superqt`.
Then confirm the pure half still imports without Qt loaded:

Run: `.venv\Scripts\python.exe -c "import vessel_utils.gui, sys; print('napari' in sys.modules)"`
Expected: `False` — napari must not be imported at module level.

- [ ] **Step 3: Write the viewer**

Add to `vessel_utils/gui.py`:

```python
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


def build_viewer(directory, preset_name="generic"):
    """Open the viewer on a directory of images. Returns the napari Viewer."""
    import napari
    from magicgui import magicgui

    paths = find_images(directory)
    preset = dict(PRESETS[preset_name])
    state = {"paths": paths, "preset": preset_name, "reference": preset["reference"],
             "spacing": None, "stages": None}
    viewer = napari.Viewer(title=f"bvp-tune - {Path(directory).name}")

    def labelled(path):
        """Combobox label, flagging files the current roles cannot be read from."""
        try:
            count = channel_count(path)
        except Exception:                                   # noqa: BLE001
            return f"{path.name}  [unreadable]"
        wanted = [i for i in preset["roles"] if i is not None]
        bad = " [channels: %d - roles %s unavailable]" % (count, tuple(wanted)) \
            if any(i >= count for i in wanted) else ""
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
    panel.file._widget._qwidget.setToolTip("\n".join(labelled(p) for p in paths))
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
```

- [ ] **Step 4: Verify the headless paths**

Run: `.venv\Scripts\python.exe -m vessel_utils.gui --selftest`
Expected: `selftest OK - reference ..., ref_af ..., dice 1.0000`

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 37 passed, still.

- [ ] **Step 5: Launch it and look at it**

Run: `.venv\Scripts\bvp-tune.exe data/composites_EV --preset "spinal-cord shipped"`

Confirm by eye, and fix anything that fails:
1. A window opens with nine layers listed (or seven, for a single-channel file).
2. Reference is magenta, test is green.
3. Toggling each layer's eye icon shows and hides it; pan and zoom work.
4. `reference vesselness` is hidden by default; showing it gives a `turbo` image whose contrast limits read 0.00–1.00 and do **not** move when you change file.
5. Dragging `ref_high` visibly changes `reference vessels` within about a second.
6. Changing `sigmas` does nothing until **Recompute** is pressed.
7. The metrics panel shows numbers, and `ref_af` for this preset lands in 0.01–0.10 with no warning.
8. Switching to `--preset "spinal-cord superseded"` and recalibrating gives an `ref_af` near 0.4 **with** the plausible-band warning shown.

- [ ] **Step 6: Commit**

```bash
git add vessel_utils/gui.py pyproject.toml
git commit -m "Tuning viewer: napari layers, magicgui panel, bvp-tune entry point"
```

---

### Task 8: Run all → CSV

**Files:**
- Modify: `vessel_utils/gui.py`
- Modify: `tests/test_gui_pipeline.py`

**Interfaces:**
- Produces: `batch_rows(paths, preset, params) -> tuple[list[dict], list[str]]` — rows and failure messages; and `run_all(viewer, state, panel, preset)`, the button handler that writes the file.

The filename must not collide with any existing result CSV. It carries the preset
and a counter, and never overwrites.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k batch -v`
Expected: FAIL — `AttributeError: module 'vessel_utils.gui' has no attribute 'batch_rows'`

- [ ] **Step 3: Write the minimal implementation**

Add to `vessel_utils/gui.py`, extending `__all__` with `"batch_rows"`, `"batch_csv_path"`:

```python
def batch_rows(paths, preset, params):
    """Apply the current parameters to every file. Returns (rows, failures).

    Failures are collected, not raised: one unreadable file must not abandon the
    other sixty. Same contract as the batch scripts' SKIP lines.
    """
    roles, rows, failures = preset["roles"], [], []
    for path in paths:
        try:
            channels, tissue = load(path, params["mask"], roles)
            st = stages(channels[0], channels[1] if len(channels) > 1 else None,
                        params["spacing"], tissue=tissue, sigmas=params["sigmas"],
                        reference=params["reference"], ref_low=params["ref_low"],
                        ref_high=params["ref_high"], test_low=params["test_low"],
                        test_high=params["test_high"],
                        min_vessel_um2=params["min_vessel_um2"],
                        virus_k=params["virus_k"], q=params["q"])
            numbers = readout(st, params["spacing"], q=params["q"],
                              plausible=preset["plausible"],
                              include_cl_dice=params["cl_dice"])
        except Exception as error:                          # noqa: BLE001
            failures.append(f"{Path(path).name}: {error}")
            continue
        warnings = numbers.pop("warnings")
        rows.append({"file": Path(path).name, **numbers,
                     "warnings": " | ".join(warnings)})
    return rows, failures


def batch_csv_path(directory, preset_name):
    """A filename that cannot collide with a published result CSV, ever.

    `tuned_` prefix plus the preset plus a counter. The published CSVs are
    spinal_cord_specificity.csv, dice_between_channels_{full,pilot}.csv and
    enrichment_cd31_percentile_*.csv; none starts with `tuned_`. The counter means
    a second run does not overwrite the first, so two operating points can be
    compared afterwards.
    """
    slug = preset_name.replace(" ", "-")
    directory = Path(directory)
    index = 1
    while (directory / f"tuned_{slug}_{index:02d}.csv").exists():
        index += 1
    return directory / f"tuned_{slug}_{index:02d}.csv"


def run_all(viewer, state, panel, preset):
    """The Run-all button: batch the current parameters and write a CSV."""
    params = {name: panel[name].value for name in panel.__signature__.parameters}
    params["sigmas"] = tuple(
        float(s) for s in str(params["sigmas"]).replace(",", " ").split())
    params["spacing"] = state["spacing"] or _spacing_tuple(preset["spacing"] or 1.0)
    params["reference"] = panel.reference.value

    rows, failures = batch_rows(state["paths"], preset, params)
    if not rows:
        print("no file produced a row; nothing written")
        for failure in failures:
            print(f"  SKIP {failure}")
        return
    import csv
    out = batch_csv_path(Path(state["paths"][0]).parent, state["preset"])
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows, {len(failures)} skipped)")
    for failure in failures:
        print(f"  SKIP {failure}")
```

Also record the resolved spacing in `refresh` so the batch reuses it — add after
`spacing = resolve_spacing(values["file"])`:

```python
        state["spacing"] = spacing
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -v`
Expected: 40 passed.

- [ ] **Step 5: Verify the button in the running app**

Run: `.venv\Scripts\bvp-tune.exe data/composites_EV --preset "spinal-cord shipped"`,
press **Run all**, and confirm the console names a written
`data/composites_EV/tuned_spinal-cord-shipped_01.csv`, that its row count plus
skip count equals the file count, and that the uncurated >2-channel files appear
as SKIP lines rather than as rows.

- [ ] **Step 6: Commit**

```bash
git add vessel_utils/gui.py tests/test_gui_pipeline.py
git commit -m "Tuning viewer: batch run to a non-colliding CSV"
```

---

### Task 9: Documentation, and the `$BVP_DATA` fix

Two unrelated deliverables, kept in one task because neither carries its own test
cycle. The `$BVP_DATA` line is independent of the GUI and worth having regardless.

**Files:**
- Modify: `vessel_utils/__init__.py`
- Modify: `CLAUDE.md`
- Modify: `scripts/analyse_spinal_cord.py`
- Modify: `tests/test_gui_pipeline.py`

- [ ] **Step 1: Write the failing test for the data-directory override**

Append to `tests/test_gui_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gui_pipeline.py -k bvp_data -v`
Expected: FAIL on the first test — `DATA` is the hardcoded `Z:` path regardless.

- [ ] **Step 3: Make the three documentation and configuration edits**

In `scripts/analyse_spinal_cord.py`, add `import os` to the standard-library
imports and replace the `DATA` line:

```python
# Overridable because the share is not mounted on most machines, and the seven
# scripts that route through this module are all unusable without it. Unset means
# the lab path, so nothing changes for anyone who has Z:.
DATA = Path(os.environ.get("BVP_DATA", r"Z:\Lab\Eric V\BEC Spinal Cords\composites_EV"))
```

In `vessel_utils/__init__.py`, add one line to the module list **inside the
docstring** — this file must import nothing, and
`test_vessel_utils.py::test_package_init_stays_empty` pins that:

```
    gui          interactive tuning viewer (needs the `[gui]` extra)
```

In `CLAUDE.md`, add a row to the `vessel_utils` table after `benchmark.py`:

```markdown
| [gui.py](vessel_utils/gui.py) | `main`, `build_viewer`, `stages`, `readout`, `reference_lambda`, `PRESETS` — the interactive tuning viewer, run as `bvp-tune <dir>`. The one module here that is an application rather than a pipeline stage: it drives the others and adds no algorithm. Needs the `[gui]` extra (`napari[all]`), imported inside functions so `import vessel_utils.gui` stays cheap. A dataset is a `PRESETS` entry — numbers plus two enum choices — never a code path; `python -m vessel_utils.gui --selftest` checks the pipeline half headlessly. |
```

- [ ] **Step 4: Run the whole suite**

Run: `.venv\Scripts\python.exe -m pytest -v`
Expected: everything passes, including `test_archive_frozen.py`,
`test_notebooks.py`, `test_equivalence.py` and
`test_vessel_utils.py::test_package_init_stays_empty`. If the frozen-archive test
fails, something touched `velazquez_rivera_2025/` — revert that; the `brain` mask
option imports from it and must never modify it.

- [ ] **Step 5: Confirm the existing scripts are unaffected**

Run: `.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'scripts'); import analyse_spinal_cord, dice_between_channels, sweep_spinal_cord; print('imports clean')"`
Expected: `imports clean`.

- [ ] **Step 6: Commit**

```bash
git add vessel_utils/__init__.py CLAUDE.md scripts/analyse_spinal_cord.py tests/test_gui_pipeline.py
git commit -m "Document the tuning viewer; let \$BVP_DATA override the lab share"
```

---

## Self-Review

**Spec coverage.** Every section maps to a task: framework choice → Task 7 (napari
+ magicgui); presets → Task 6; `reference_lambda` per-dataset → Task 2 and
`calibrate` in Task 7; cost cascade and the two caches → Task 6; both virus
definitions always shown → Task 1 (`test_positive`, `ref_top_q`) and Task 7
(layers); nine layers with pinned response clims → Task 7; dock and readout →
Tasks 5 and 7; plausible-band flag → Task 5; percentile tie guard → Task 5; Run
all → CSV with a non-colliding name → Task 8; data location → Task 9; tissue-mask
dropdown over four existing maskers → Task 3; the channel-count guard → Task 4;
error handling (leave layers up, napari missing, empty directory, absent spacing,
batch failures collected) → Tasks 4, 7, 8; testing on `phantom` with no Qt →
Tasks 1–6; `--selftest` → Task 6.

3D-readiness is a constraint rather than a task, and is enforced by construction:
`spacing` is a tuple everywhere, `min_vessel` is µm² converted through
`prod(spacing)`, and no 2D-only call appears — `test_min_vessel_is_physical_not_pixels`
in Task 1 pins the conversion.

**Two spec items deliberately not built**, both listed as out of scope there: the
archive backend, and 3D volume loading (which would also mean restoring
`vessel_utils/storage.py` and re-adding `zarr` and `dask`).

**Known simplification.** The combobox flags unreadable files through a tooltip
rather than per-item styling, because magicgui exposes no per-choice disable. The
guard that matters is in `read_channels`, which raises; the tooltip is a
convenience. If it proves too easy to miss, filter the choices instead of
labelling them.
