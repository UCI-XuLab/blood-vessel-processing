"""Active vessel segmentation and channel-comparison code.

The published pipeline lives in `velazquez_rivera_2025` and is frozen. This
package is where new work goes, and it is free to differ — the point is to do
better, not to reproduce.

    storage      chunked OME-Zarr conversion and access for volumes > memory
    correct      lightsheet destriping and depth attenuation correction
    chunked      running filters over large volumes with correctly sized halos
    vesselness   Jerman vesselness: physical scales, 2D/3D, bounded response
    threshold    hysteresis thresholding and mask clean-up
    metrics      Dice, Jaccard, precision, recall, clDice, area fractions
    sweep        threshold sensitivity analysis
    synth        synthetic vasculature with a simulated lightsheet acquisition
    ensemble     combining several segmentations and mapping their disagreement
    benchmark    scoring a segmenter against phantoms with known ground truth
    qc           first-contact inspection of a new acquisition
    validate     accuracy on real data without dense annotation

Typical use for comparing two channels::

    from vessel_utils.vesselness import jerman_vesselness, max_eigenvalue
    from vessel_utils.sweep import threshold_sweep, stability

    spacing = (3.0, 0.75, 0.75)          # z, y, x in um
    sigmas = [1.5, 3.0, 6.0, 12.0]       # bracket the vessel radii, in um

    # Pin the tau reference once so a threshold means the same thing everywhere.
    reference = max_eigenvalue(calibration_volume, sigmas, spacing)

    va = jerman_vesselness(channel_a, sigmas, spacing, reference_lambda=reference)
    vb = jerman_vesselness(channel_b, sigmas, spacing, reference_lambda=reference)

    rows = threshold_sweep(va, vb, np.linspace(0.05, 0.6, 12), roi=brain_mask)
    print(stability(rows, "dice"))

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
"""
