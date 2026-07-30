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

Names are re-exported lazily, so importing `metrics` does not pull in anything
`vesselness` needs.
"""

import importlib

__all__ = [
    # storage
    "plane_series_to_zarr", "open_volume", "write_volume", "pyramid_levels",
    "read_spacing",
    # correct
    "destripe", "tissue_mask", "depth_profile", "correct_depth_attenuation",
    "smooth_profile",
    # chunked
    "gaussian_reach", "overlap_depth", "map_blocks_with_halo", "apply_vesselness",
    # vesselness
    "hessian_eigenvalues", "jerman_vesselness", "max_eigenvalue",
    # threshold
    "hysteresis_threshold", "otsu_threshold", "clean_mask", "segment",
    # metrics
    "dice", "jaccard", "precision", "recall", "cl_dice", "area_fraction",
    "agreement", "agreement_by_calibre",
    # sweep
    "threshold_sweep", "stability", "write_csv",
    # synth
    "Segment", "vascular_tree", "render_tree", "simulate_acquisition", "phantom",
    # ensemble
    "vote_fraction", "consensus", "disagreement_map", "pairwise_agreement",
    "redundancy",
    # benchmark
    "score_segmentation", "run_benchmark", "sweep_condition", "summarise",
    # qc
    "inspect_volume", "estimate_attenuation", "stripe_severity", "resolvability",
    "compare_channels", "suggest_sigmas", "intake_report", "format_report",
    # validate
    "stratified_sample", "extract_crops", "estimate_accuracy",
    "agreement_by_depth", "depth_invariance",
]

_ORIGIN = {
    "plane_series_to_zarr": "storage", "open_volume": "storage",
    "write_volume": "storage", "pyramid_levels": "storage",
    "read_spacing": "storage",
    "destripe": "correct", "tissue_mask": "correct", "depth_profile": "correct",
    "correct_depth_attenuation": "correct", "smooth_profile": "correct",
    "gaussian_reach": "chunked", "overlap_depth": "chunked",
    "map_blocks_with_halo": "chunked", "apply_vesselness": "chunked",
    "hessian_eigenvalues": "vesselness", "jerman_vesselness": "vesselness",
    "max_eigenvalue": "vesselness",
    "hysteresis_threshold": "threshold", "otsu_threshold": "threshold",
    "clean_mask": "threshold", "segment": "threshold",
    "dice": "metrics", "jaccard": "metrics", "precision": "metrics",
    "recall": "metrics", "cl_dice": "metrics", "area_fraction": "metrics",
    "agreement": "metrics", "agreement_by_calibre": "metrics",
    "threshold_sweep": "sweep", "stability": "sweep", "write_csv": "sweep",
    "Segment": "synth", "vascular_tree": "synth", "render_tree": "synth",
    "simulate_acquisition": "synth", "phantom": "synth",
    "vote_fraction": "ensemble", "consensus": "ensemble",
    "disagreement_map": "ensemble", "pairwise_agreement": "ensemble",
    "redundancy": "ensemble",
    "score_segmentation": "benchmark", "run_benchmark": "benchmark",
    "sweep_condition": "benchmark", "summarise": "benchmark",
    "inspect_volume": "qc", "estimate_attenuation": "qc", "stripe_severity": "qc",
    "resolvability": "qc", "compare_channels": "qc", "suggest_sigmas": "qc",
    "intake_report": "qc", "format_report": "qc",
    "stratified_sample": "validate", "extract_crops": "validate",
    "estimate_accuracy": "validate", "agreement_by_depth": "validate",
    "depth_invariance": "validate",
}


_SUBMODULES = {"storage", "correct", "chunked", "vesselness", "threshold",
               "metrics", "sweep", "synth", "ensemble", "benchmark", "qc",
               "validate"}


def __getattr__(name):
    # A submodule always wins over a re-exported function of the same name.
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _ORIGIN:
        module = importlib.import_module(f"{__name__}.{_ORIGIN[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
