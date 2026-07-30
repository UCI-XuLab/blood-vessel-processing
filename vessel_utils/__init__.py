"""Active vessel segmentation and channel-comparison code.

The published pipeline lives in `velazquez_rivera_2025` and is frozen. This
package is where new work goes, and it is free to differ — the point is to do
better, not to reproduce.

    vesselness   Jerman vesselness: physical scales, 2D/3D, bounded response
    threshold    hysteresis thresholding and mask clean-up
    metrics      Dice, Jaccard, precision, recall, clDice, area fractions
    sweep        threshold sensitivity analysis

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

Names are re-exported lazily, so importing `metrics` does not pull in anything
`vesselness` needs.
"""

import importlib

__all__ = [
    # vesselness
    "hessian_eigenvalues", "jerman_vesselness", "max_eigenvalue",
    # threshold
    "hysteresis_threshold", "otsu_threshold", "clean_mask", "segment",
    # metrics
    "dice", "jaccard", "precision", "recall", "cl_dice", "area_fraction",
    "agreement", "agreement_by_calibre",
    # sweep
    "threshold_sweep", "stability", "write_csv",
]

_ORIGIN = {
    "hessian_eigenvalues": "vesselness", "jerman_vesselness": "vesselness",
    "max_eigenvalue": "vesselness",
    "hysteresis_threshold": "threshold", "otsu_threshold": "threshold",
    "clean_mask": "threshold", "segment": "threshold",
    "dice": "metrics", "jaccard": "metrics", "precision": "metrics",
    "recall": "metrics", "cl_dice": "metrics", "area_fraction": "metrics",
    "agreement": "metrics", "agreement_by_calibre": "metrics",
    "threshold_sweep": "sweep", "stability": "sweep", "write_csv": "sweep",
}


def __getattr__(name):
    if name in _ORIGIN:
        module = importlib.import_module(f"{__name__}.{_ORIGIN[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
