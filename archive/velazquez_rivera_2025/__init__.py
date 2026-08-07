"""Shared helpers for the blood vessel processing notebooks.

Modules follow the pipeline stages:

    io        reading TIFF slices and unpacking channels
    enhance   contrast, gamma, histogram equalisation, N4 bias correction
    vessels   Hessian vesselness detection, mask post-processing, brain masking
    metrics   agreement metrics between two binary masks
    viz       inline previews and figure export

Names are re-exported here for convenience, but resolved lazily so that
importing one module does not pull in another's heavy dependency — reaching for
`metrics` should not require `itk` to be installed.
"""

import importlib

__all__ = [
    # io
    "read_tif", "load_channel", "load_channels", "load_3_channels",
    # enhance
    "auto_contrast", "gamma_correction", "histogram_equalization",
    "compute_average_image", "n4_bias_correction",
    # vessels
    "detect_vessels", "process_vessels", "get_brain_mask",
    # metrics
    "dice_coefficient", "iou", "precision", "recall", "rand_index",
    # viz
    "show", "show3", "show_4", "save_figure",
]

_ORIGIN = {
    "read_tif": "io", "load_channel": "io", "load_channels": "io", "load_3_channels": "io",
    "auto_contrast": "enhance", "gamma_correction": "enhance",
    "histogram_equalization": "enhance", "compute_average_image": "enhance",
    "n4_bias_correction": "enhance",
    "detect_vessels": "vessels", "process_vessels": "vessels", "get_brain_mask": "vessels",
    "dice_coefficient": "metrics", "iou": "metrics", "precision": "metrics",
    "recall": "metrics", "rand_index": "metrics",
    "show": "viz", "show3": "viz", "show_4": "viz", "save_figure": "viz",
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
