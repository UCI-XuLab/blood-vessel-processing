"""Running filters over volumes larger than memory.

Every filter in this package has a neighbourhood: a Gaussian at sigma reaches
about 3*sigma, a closing reaches its footprint radius, a distance transform
reaches arbitrarily far. Apply one chunk at a time and each chunk's border is
computed from truncated data, leaving a visible seam at every chunk boundary —
and on a vessel mask a seam is not cosmetic, it severs vessels and changes the
connectivity that clDice and any skeleton-based measure depend on.

The fix is to give each chunk a halo of neighbouring data, compute, then discard
the halo. `dask.array.map_overlap` does the bookkeeping; what this module adds is
choosing the halo correctly from the filter's actual reach in physical units, so
it is right by construction rather than by being checked. The one guard that does
exist rejects a halo too *large* for the chunks, where each block would be mostly
halo and the work would be dominated by recomputation.
"""

import numpy as np

__all__ = ["gaussian_reach", "overlap_depth", "map_blocks_with_halo",
           "apply_vesselness"]

# Must match the truncation `scipy.ndimage.gaussian_filter` actually uses, which
# defaults to 4.0 — not the 3.0 that is conventionally quoted for a Gaussian's
# "effective" width. Sizing the halo at 3 sigma while the filter reaches 4 leaves
# a real seam at every chunk boundary: small in absolute terms, but concentrated
# exactly on the block edges, which is where a severed vessel changes topology.
GAUSSIAN_TRUNCATE = 4.0


def gaussian_reach(sigma, spacing=None, truncate=GAUSSIAN_TRUNCATE):
    """Voxels a Gaussian of physical width `sigma` reaches along each axis.

    Anisotropic spacing is the whole point: at (3.0, 0.75, 0.75) um a sigma of
    6 um reaches 6 voxels in z but 24 in x. A single scalar halo would be either
    wasteful in z or wrong in xy.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    spacing = np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)
    return tuple(int(np.ceil(truncate * sigma / s)) for s in spacing)


def overlap_depth(sigmas, spacing=None, extra=0, truncate=GAUSSIAN_TRUNCATE):
    """Halo depth per axis for a multiscale filter.

    Sized from the *largest* sigma, since one pass over the chunk computes every
    scale. `extra` adds room for whatever runs after the filter — a binary
    closing of radius r needs r more voxels, a small-object removal needs none.
    """
    sigmas = [float(s) for s in sigmas]
    if not sigmas:
        raise ValueError("at least one sigma is required")
    reach = gaussian_reach(max(sigmas), spacing, truncate)
    return tuple(r + int(extra) for r in reach)


def map_blocks_with_halo(func, array, depth, dtype=None, boundary="nearest",
                         **kwargs):
    """Apply `func` blockwise with a halo, then trim it away.

    Args:
        func: callable taking an ndarray block and returning one of the same shape.
        array: dask array.
        depth: per-axis halo in voxels, from `overlap_depth`.
        boundary: how to extend the volume edge. "nearest" matches the mode the
            filters use internally, so a chunk at the volume boundary sees the
            same data it would if the volume were processed whole.

    Returns:
        A dask array. Nothing is computed until you ask for it.
    """
    import dask.array as da

    if array.ndim != len(depth):
        raise ValueError(f"depth has {len(depth)} entries for a {array.ndim}D array")

    smallest = tuple(min(c) for c in array.chunks)
    if any(d * 2 >= s for d, s in zip(depth, smallest)):
        raise ValueError(
            f"halo {depth} is too large for chunks {smallest}: each block would be "
            f"mostly halo. Rechunk larger, or filter at a coarser pyramid level."
        )

    return da.map_overlap(func, array, depth=depth, boundary=boundary,
                          dtype=dtype or array.dtype, **kwargs)


def apply_vesselness(array, sigmas, spacing, tau=0.75, bright_objects=True,
                     reference_lambda=None, dtype=np.float32, extra_depth=0):
    """Jerman vesselness over a chunked volume, with the halo sized automatically.

    `reference_lambda` matters more here than anywhere else. Left as None, each
    *block* would regularise against its own maximum eigenvalue, so the response
    would mean something different in every chunk and the seams would become
    genuine discontinuities in the output rather than merely edge artefacts.
    Compute it once with `max_eigenvalue` on a representative sub-volume and pass
    it in; this function refuses to run without it.

    Returns:
        A lazy dask array of the vesselness response.
    """
    from .vesselness import jerman_vesselness

    if reference_lambda is None:
        raise ValueError(
            "reference_lambda is required for chunked processing: without it each "
            "block regularises against its own maximum and the response is not "
            "comparable between chunks. Compute one with max_eigenvalue() on a "
            "representative sub-volume."
        )

    depth = overlap_depth(sigmas, spacing, extra=extra_depth)

    def block(data):
        return jerman_vesselness(data, sigmas, spacing=spacing, tau=tau,
                                 bright_objects=bright_objects,
                                 reference_lambda=reference_lambda,
                                 normalise=False, dtype=dtype)

    return map_blocks_with_halo(block, array, depth, dtype=dtype)
