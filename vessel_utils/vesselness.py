"""Hessian vesselness filters.

Two things differ from the published implementation in `velazquez_rivera_2025`:

**Physical scale.** Sigmas are in physical units when `spacing` is supplied, so a
sigma of 5 means 5 um whether the voxels are isotropic or not. The archive runs
on raw arrays with implicit 1x1 spacing, which on anisotropic lightsheet data
(3 um in z, something else in xy) means the filter searches for different vessel
calibres along different axes.

**Response scaling.** `itk.RescaleIntensityImageFilter`, used by the archive,
stretches each image's own min/max across 0-255. A fixed threshold on that output
is a *relative* criterion: it means something different in every image, and in
particular something different in each of the two channels being compared.
Jerman's response is instead bounded in [0, 1] by construction, saturating at 1
for strong vessels, so a fixed threshold keeps its meaning across images.

One residual image dependence remains in Jerman: the `tau` regularisation refers
to the image's own maximum eigenvalue. Pass `reference_lambda` to pin it to a
value computed once over the dataset and remove that dependence too.

Reference: T. Jerman, F. Pernus, B. Likar, Z. Spiclin, "Enhancement of Vascular
Structures in 3D and 2D Angiographic Images", IEEE TMI 35(9):2107-2118, 2016.
Implemented from the authors' reference code at
https://github.com/timjerman/JermanEnhancementFilter
"""

import numpy as np
import scipy.ndimage as ndi

__all__ = ["hessian_eigenvalues", "jerman_vesselness", "max_eigenvalue"]


def _resolve_spacing(spacing, ndim):
    if spacing is None:
        return np.ones(ndim, dtype=float)
    spacing = np.atleast_1d(np.asarray(spacing, dtype=float))
    if spacing.size == 1:
        spacing = np.repeat(spacing, ndim)
    if spacing.size != ndim:
        raise ValueError(f"spacing has {spacing.size} entries for a {ndim}D image")
    if np.any(spacing <= 0):
        raise ValueError(f"spacing must be positive, got {spacing}")
    return spacing


def hessian_eigenvalues(image, sigma, spacing=None, dtype=np.float32):
    """Scale-normalised Hessian eigenvalues, sorted by increasing magnitude.

    Args:
        image: 2D or 3D array.
        sigma: float, smoothing scale in physical units (same units as spacing).
        spacing: per-axis voxel size, in the array's axis order. None means
            isotropic unit spacing.
        dtype: working precision for the eigen decomposition.

    Returns:
        Array of shape image.shape + (ndim,), eigenvalues ordered so that
        |e[..., 0]| <= |e[..., 1]| <= ... — the convention Jerman and Frangi use.

    Memory note: for 3D this materialises a (..., 3, 3) tensor, roughly 9x the
    image in `dtype`. Process large volumes in overlapping chunks; an overlap of
    about 3 * max(sigma) / spacing voxels keeps boundary effects out of the
    interior.
    """
    image = np.asarray(image, dtype=dtype)
    ndim = image.ndim
    if ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got {ndim}D")
    spacing = _resolve_spacing(spacing, ndim)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    sigma_voxels = sigma / spacing
    # gamma=2 scale normalisation, so responses are comparable across sigmas.
    gamma = sigma ** 2

    components = {}
    for i in range(ndim):
        for j in range(i, ndim):
            order = [0] * ndim
            order[i] += 1
            order[j] += 1
            derivative = ndi.gaussian_filter(image, sigma=sigma_voxels, order=order,
                                             mode="nearest")
            # Convert the derivative to physical units before scale normalising.
            components[(i, j)] = derivative * (gamma / (spacing[i] * spacing[j]))

    if ndim == 2:
        # Closed form is much faster than a general solver for 2x2.
        a, b, c = components[(0, 0)], components[(0, 1)], components[(1, 1)]
        centre = 0.5 * (a + c)
        spread = np.sqrt(np.square(0.5 * (a - c)) + np.square(b))
        eigenvalues = np.stack([centre - spread, centre + spread], axis=-1)
    else:
        hessian = np.empty(image.shape + (ndim, ndim), dtype=dtype)
        for (i, j), value in components.items():
            hessian[..., i, j] = value
            hessian[..., j, i] = value
        eigenvalues = np.linalg.eigvalsh(hessian)

    order = np.argsort(np.abs(eigenvalues), axis=-1)
    return np.take_along_axis(eigenvalues, order, axis=-1)


def max_eigenvalue(image, sigmas, spacing=None, bright_objects=True, dtype=np.float32):
    """Largest regularisation eigenvalue over all scales, for `reference_lambda`.

    Run this once over a representative subset of the dataset and pass the result
    to every `jerman_vesselness` call. That makes the tau regularisation a fixed
    criterion instead of a per-image one, which is what lets a single threshold
    mean the same thing in both channels and across slices.
    """
    best = 0.0
    for sigma in sigmas:
        eigenvalues = hessian_eigenvalues(image, sigma, spacing, dtype)
        largest = eigenvalues[..., -1]
        if bright_objects:
            largest = -largest
        finite = largest[np.isfinite(largest)]
        if finite.size:
            best = max(best, float(finite.max()))
    return best


def jerman_vesselness(image, sigmas, spacing=None, tau=0.75, bright_objects=True,
                      reference_lambda=None, normalise=False, dtype=np.float32):
    """Jerman vesselness: a bounded, calibre-uniform tubular response.

    Args:
        image: 2D or 3D array.
        sigmas: scales to search, in physical units if `spacing` is given. Choose
            them to bracket the vessel radii you care about.
        spacing: per-axis voxel size in the array's axis order.
        tau: regularisation strength in (0, 1]. Lower values widen the saturated
            region, giving a more uniform response but less contrast between
            calibres. 0.75 is the reference default.
        bright_objects: True for bright vessels on a dark background, the normal
            fluorescence case. The archive's `detect_vessels` instead passes
            `SetBrightObject(False)` and then inverts its threshold in
            `process_vessels`, reaching the same place by a double negation.
        reference_lambda: fix the tau reference instead of using this image's own
            maximum. See `max_eigenvalue`. None reproduces the reference
            implementation.
        normalise: divide by the response maximum and zero anything below 1e-2,
            as the reference implementation does. Off by default: it reintroduces
            exactly the per-image dependence this module exists to avoid. Turn it
            on only to reproduce reference outputs.

    Returns:
        Float array in [0, 1], same shape as `image`. Saturates at 1 for vessels
        at or above the regularisation scale, so a fixed threshold is meaningful
        across images.
    """
    if not 0 < tau <= 1:
        raise ValueError(f"tau must lie in (0, 1], got {tau}")
    sigmas = [float(s) for s in sigmas]
    if not sigmas:
        raise ValueError("at least one sigma is required")

    image = np.asarray(image, dtype=dtype)
    vesselness = np.zeros(image.shape, dtype=dtype)

    for sigma in sigmas:
        eigenvalues = hessian_eigenvalues(image, sigma, spacing, dtype)
        if image.ndim == 2:
            # In 2D the reference uses the single cross-sectional eigenvalue for
            # both roles, so lambda_rho is regularised from lambda_2 itself.
            lambda_2 = eigenvalues[..., 1]
            if bright_objects:
                lambda_2 = -lambda_2
            lambda_3 = lambda_2
        else:
            lambda_2 = eigenvalues[..., 1]
            lambda_3 = eigenvalues[..., 2]
            if bright_objects:
                lambda_2 = -lambda_2
                lambda_3 = -lambda_3

        if reference_lambda is None:
            finite = lambda_3[np.isfinite(lambda_3)]
            peak = float(finite.max()) if finite.size else 0.0
        else:
            peak = float(reference_lambda)
        cutoff = tau * peak

        lambda_rho = np.where(
            lambda_3 <= 0, 0.0,
            np.where(lambda_3 <= cutoff, cutoff, lambda_3),
        ).astype(dtype)

        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = np.power(lambda_2 + lambda_rho, 3)
            response = (np.square(lambda_2) * (lambda_rho - lambda_2) * 27.0
                        / denominator)

        # Saturate where the vessel is at or above the regularisation scale.
        response = np.where((lambda_2 >= lambda_rho / 2) & (lambda_rho > 0), 1.0, response)
        response = np.where((lambda_2 <= 0) | (lambda_rho <= 0), 0.0, response)
        response[~np.isfinite(response)] = 0.0

        np.maximum(vesselness, response, out=vesselness)

    if normalise:
        peak = float(vesselness.max())
        if peak > 0:
            vesselness /= peak
        vesselness[vesselness < 1e-2] = 0.0

    return vesselness
