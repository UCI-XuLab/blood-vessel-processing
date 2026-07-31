"""Separate grey from white matter in a spinal cord section, automatically.

Why the compartment must come from CD31, not the virus channel
--------------------------------------------------------------
Grey matter is obvious in the virus channel — it is the bright butterfly, because
that is where leak transduction into neurons sits. Segmenting it that way would
be circular: the compartment would be defined by the signal whose
compartment-dependence is under test, and "the virus is brighter in grey matter"
would follow by construction.

What failed in the first attempt, and why
-----------------------------------------
Smoothing the full vessel mask at 150 um and splitting with Otsu produced masks
that did not trace the butterfly: fragmented in cervical, displaced in thoracic,
and a featureless blob in lumbar, all reporting a grey fraction near 50%.

Three causes, each fixed here:

  - Otsu assumes two modes. A map smoothed at 150 um is a broad unimodal
    gradient, so Otsu cuts near the middle of the range whatever the anatomy —
    which is exactly why every section came back near half grey. This module now
    fits a two-component Gaussian mixture and *checks* the components are
    actually separated, refusing to split when they are not.

  - All vessel calibres were pooled. The grey/white contrast is specifically a
    capillary density difference; large penetrating and pial vessels are as
    common in white matter and dilute it. Only the capillary-scale response is
    used now.

  - Bright pial staining at the section rim survived a 40 um erosion and
    dominated the cervical density map, which peaked at the edge rather than in
    the butterfly. The rim is cut harder here.

A method that can fail quietly is worse than none, so `compartment_masks`
returns a confidence report and raises when the split is not supported.

Work in progress: on the pilot sections these masks locate the horns but do not
trace the butterfly cleanly, and the grey fraction comes out anatomically
inverted. Not used for any reported result. Requires scikit-learn (not a package
dependency; install it separately to run this script).
"""

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import (binary_closing, disk, remove_small_holes,
                                remove_small_objects)

__all__ = ["capillary_density", "split_threshold", "compartment_masks"]

CAPILLARY_SIGMAS = [1.5, 3.0]   # um; capillary radii, not penetrating vessels
SMOOTH_UM = 70.0                # >> capillary spacing, << compartment
RIM_UM = 90.0                   # pial staining reaches further than 40 um
MIN_REGION_UM2 = 20000.0
MIN_SEPARATION = 1.0            # component means must differ by this many sd


def capillary_density(cd31_normalised, tissue, um_per_px, sigmas=CAPILLARY_SIGMAS,
                      reference_lambda=2.5, smooth_um=SMOOTH_UM):
    """Local capillary area fraction, the quantity that actually differs.

    Restricted to capillary scales on purpose: grey matter carries two to four
    times the capillary density of white matter, while large vessels are present
    in both and wash the contrast out.
    """
    from vessel_utils.threshold import segment
    from vessel_utils.vesselness import jerman_vesselness

    spacing = (um_per_px, um_per_px)
    response = jerman_vesselness(cd31_normalised, sigmas, spacing,
                                 reference_lambda=reference_lambda)
    capillaries = segment(response, low=0.25, high=0.5, roi=tissue,
                          min_size=max(1, int(round(3.0 / um_per_px ** 2))),
                          area_threshold=0, closing_radius=0)
    density = ndi.gaussian_filter(capillaries.astype(np.float32),
                                  sigma=smooth_um / um_per_px)
    # Renormalise inside tissue: the Gaussian pulls in zeros from outside the
    # section, which would otherwise read as low density all round the rim.
    weight = ndi.gaussian_filter(tissue.astype(np.float32),
                                 sigma=smooth_um / um_per_px)
    return np.where(weight > 1e-3, density / np.maximum(weight, 1e-3), 0.0)


def split_threshold(values, min_separation=MIN_SEPARATION):
    """Two-component Gaussian mixture split, with a check that it is justified.

    Returns (threshold, report). The report carries the component means and the
    separation in pooled standard deviations. A unimodal distribution will still
    yield a threshold from any fitting procedure; the separation is what says
    whether that threshold means anything.
    """
    from sklearn.mixture import GaussianMixture

    sample = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if sample.shape[0] > 200_000:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(sample.shape[0], 200_000, replace=False)]

    model = GaussianMixture(n_components=2, random_state=0).fit(sample)
    means = np.sort(model.means_.ravel())
    order = np.argsort(model.means_.ravel())
    sds = np.sqrt(model.covariances_.ravel()[order])
    weights = model.weights_.ravel()[order]

    pooled = float(np.sqrt((sds[0] ** 2 + sds[1] ** 2) / 2))
    separation = float((means[1] - means[0]) / pooled) if pooled > 0 else 0.0

    # Threshold where the two components' weighted densities cross.
    grid = np.linspace(means[0], means[1], 512)
    def density(index):
        return (weights[index] * np.exp(-0.5 * ((grid - means[index]) / sds[index]) ** 2)
                / sds[index])
    crossing = grid[np.argmin(np.abs(density(0) - density(1)))]

    return float(crossing), {
        "low_mean": float(means[0]), "high_mean": float(means[1]),
        "separation_sd": separation, "weights": weights.tolist(),
        "separated": separation >= min_separation,
    }


def compartment_masks(cd31_normalised, tissue, um_per_px, reference_lambda=2.5,
                      min_separation=MIN_SEPARATION, strict=True):
    """Split tissue into (grey, white, report) by capillary density.

    Raises ValueError when the density distribution does not support a split,
    unless `strict` is False. Silently returning a meaningless half-and-half
    partition is the failure mode this exists to prevent.
    """
    tissue = np.asarray(tissue, dtype=bool)
    # Euclidean rim, matching analyse_spinal_cord.tissue_mask. Iterating a
    # 4-connected structure is an L1 erosion, so a diagonal edge would only be
    # cleared to RIM_UM / sqrt(2) ~ 64 um instead of 90 — under-delivering the
    # rim exactly where pial staining runs along oblique borders.
    core = tissue & (ndi.distance_transform_edt(tissue) > RIM_UM / um_per_px)
    if not core.any():
        raise ValueError("section is smaller than the rim exclusion")

    density = capillary_density(cd31_normalised, tissue, um_per_px,
                                reference_lambda=reference_lambda)
    threshold, report = split_threshold(density[core], min_separation)
    report["threshold"] = threshold

    if strict and not report["separated"]:
        raise ValueError(
            f"capillary density is not bimodal inside this section "
            f"(component separation {report['separation_sd']:.2f} sd, need "
            f"{min_separation}). A threshold here would be arbitrary."
        )

    grey = (density > threshold) & core
    min_px = int(round(MIN_REGION_UM2 / um_per_px ** 2))
    grey = remove_small_objects(grey, min_size=min_px)
    grey = remove_small_holes(grey, area_threshold=min_px)
    grey = binary_closing(grey, footprint=disk(int(round(40.0 / um_per_px)))) & core

    report["grey_fraction"] = float(grey.sum() / core.sum()) if core.any() else 0.0
    return grey, core & ~grey, report
