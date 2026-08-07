"""Vendored third-party modules. See README.md for provenance and re-sync steps.

The entropy-guided GrabCut tissue masker is re-exported here so callers write
``from vessel_utils._vendor import compute_entropy_grabcut`` rather than reaching
into the vendored file layout.
"""

from .grabcut import (
    EntropyGrabCutConfig,
    EntropyGrabCutResult,
    compute_entropy_grabcut,
)

__all__ = [
    "compute_entropy_grabcut",
    "EntropyGrabCutConfig",
    "EntropyGrabCutResult",
]
