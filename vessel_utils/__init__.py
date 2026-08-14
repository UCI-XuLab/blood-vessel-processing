"""Active vessel segmentation and channel-comparison code.

The published pipeline lives in `velazquez_rivera_2025` and is frozen. This
package is where new work goes, and it is free to differ — the point is to do
better, not to reproduce.

    vesselness   Jerman vesselness: physical scales, 2D/3D, bounded response
    threshold    hysteresis thresholding and mask clean-up
    metrics      Dice, Jaccard, precision, recall, clDice, area fractions
    synth        synthetic vasculature with a simulated lightsheet acquisition
    benchmark    scoring a segmenter against phantoms with known ground truth
    _vendor      the entropy-guided GrabCut tissue masker, from UCI-XuLab-RegTools

Typical use for comparing two channels::

    from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue
    from vessel_utils.threshold import segment

    spacing = (0.65, 0.65)               # um per pixel
    sigmas = [1.5, 3.0, 6.0, 12.0]       # bracket the vessel radii, in um

    # Pin the tau reference once so a threshold means the same thing everywhere.
    reference = max_eigenvalue(calibration_image, sigmas, spacing, percentile=99.9)

    va = jerman_vesselness(channel_a, sigmas, spacing, reference_lambda=reference)
    mask_a = segment(va, low=0.03, high=0.09, roi=tissue)

Import the submodule by name — `from vessel_utils import metrics`, or
`from vessel_utils.threshold import segment`. Both forms make the import system
load that one submodule and nothing else, so reaching for a metric does not pull
in anything `vesselness` needs. This module deliberately imports nothing itself,
which is what keeps that true; `test_package_init_stays_empty` pins it.

Note that a bare `import vessel_utils` does **not** make the submodules reachable
as attributes: `vessel_utils.metrics` after only `import vessel_utils` raises
`AttributeError`. An earlier version of this file supported that through a lazy
`__getattr__` over a 45-name table; the table served no caller and was removed.
Name the submodule in the import and both forms above work.

**What is not here, and why.** `storage`, `chunked` and `correct` (chunked
OME-Zarr conversion, halo-sized blockwise filtering, lightsheet destriping and
depth-attenuation correction), plus `qc`, `validate`, `ensemble` and `sweep`,
were written for whole-brain lightsheet volumes and for validation designs that
no analysis in this repository ever ran — every caller was its own test. They
were removed rather than left to rot into documentation of work that did not
happen; `git log` has them if that work restarts. Removing them also dropped
`zarr`, `dask` and `PyWavelets` from the dependency list.
"""
