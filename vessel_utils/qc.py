"""First contact with a new acquisition.

Run this before any analysis, while the microscope is still available. Most of
what it reports cannot be fixed later: if the sampling is too coarse to resolve a
capillary, or one channel is clipping, or attenuation has swallowed the deep
half of the brain, the useful response is to re-image — and that option expires.

It also answers, from the data itself, most of the parameters the rest of the
pipeline needs: what sigma range to search, roughly where a threshold should sit,
and whether the two channels are comparable in the first place.

Nothing here needs the voxel size to run, but supplying it turns several
observations into physical statements — which is the difference between "the PSF
is 4 voxels wide" and "capillaries are unresolvable".
"""

import numpy as np

__all__ = ["inspect_volume", "estimate_attenuation", "stripe_severity",
           "resolvability", "compare_channels", "intake_report", "format_report"]

# A capillary lumen is roughly 2-4 um across in mouse cortex.
CAPILLARY_DIAMETER_UM = 4.0


def inspect_volume(volume, spacing=None, sample=2_000_000, seed=0):
    """Basic condition of one channel: range, saturation, noise, contrast.

    Large volumes are sub-sampled; every statistic here is a distribution
    property, so a few million voxels is ample and reading 500 GB is not.
    """
    volume = np.asarray(volume)
    flat = volume.reshape(-1)
    if flat.size > sample:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.size, size=sample, replace=False)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("volume has no finite values")

    percentiles = np.percentile(flat, [0.1, 1, 25, 50, 75, 99, 99.9])
    info = np.iinfo(volume.dtype) if np.issubdtype(volume.dtype, np.integer) else None
    ceiling = info.max if info is not None else float(flat.max())

    # Background is taken as the lower quartile: vessels are a bright minority,
    # so the bulk of the distribution is parenchyma.
    background = float(percentiles[2])
    # Noise from a robust spread of the darker half, which contains no vessels.
    dark = flat[flat <= percentiles[3]]
    noise = float(1.4826 * np.median(np.abs(dark - np.median(dark)))) if dark.size else float("nan")

    return {
        "shape": tuple(int(s) for s in volume.shape),
        "dtype": str(volume.dtype),
        "spacing_um": tuple(spacing) if spacing is not None else None,
        "voxels": int(np.prod(volume.shape)),
        "gigabytes": float(np.prod(volume.shape) * volume.dtype.itemsize / 1e9),
        "p0_1": float(percentiles[0]), "p1": float(percentiles[1]),
        "median": float(percentiles[3]),
        "p99": float(percentiles[5]), "p99_9": float(percentiles[6]),
        "background": background,
        "noise_sigma": noise,
        "contrast_to_noise": float((percentiles[5] - background) / noise)
        if noise and np.isfinite(noise) and noise > 0 else float("nan"),
        "saturated_fraction": float(np.mean(flat >= ceiling * 0.999)),
        "dynamic_range_used": float((flat.max() - flat.min()) / ceiling)
        if ceiling else float("nan"),
    }


def estimate_attenuation(volume, spacing=None, mask=None, axis=0):
    """Fit exponential signal decay with depth, per micrometre.

    Returned as a decay constant and a half-depth. The half-depth is the number
    to look at: if it is comparable to the brain's thickness, the deep half of
    every measurement is systematically dimmer, and in a two-channel comparison
    that becomes a spurious gradient because wavelengths attenuate differently.
    """
    from .correct import depth_profile, smooth_profile

    volume = np.asarray(volume)
    profile = smooth_profile(depth_profile(volume, mask=mask, axis=axis), sigma=2.0)
    depth_voxels = np.arange(profile.size)
    step = float(spacing[axis]) if spacing is not None else 1.0
    depth_um = depth_voxels * step

    positive = profile > 0
    if positive.sum() < 3:
        raise ValueError("not enough usable planes to fit attenuation")
    slope, intercept = np.polyfit(depth_um[positive], np.log(profile[positive]), 1)
    decay = float(-slope)

    return {
        "decay_per_um": decay,
        "half_depth_um": float(np.log(2) / decay) if decay > 1e-12 else float("inf"),
        "surface_level": float(np.exp(intercept)),
        "retained_at_end": float(profile[-1] / profile[0]) if profile[0] > 0 else float("nan"),
        "profile": profile,
        "units": "physical" if spacing is not None else "voxels",
    }


def stripe_severity(volume, axis=1, sample_planes=16, seed=0):
    """How much of the signal varies along the illumination axis.

    Stripes are constant along the sheet direction, so they show up as variation
    between the *means* of lines running that way. Reported relative to overall
    contrast, so it is comparable between channels.

    A caveat that matters: vasculature genuinely aligned with the sheet inflates
    this. Read it as an upper bound on stripe artefact, and confirm on a region
    you know before destriping aggressively.
    """
    volume = np.asarray(volume)
    if volume.ndim == 2:
        planes = [volume]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(volume.shape[0],
                             size=min(sample_planes, volume.shape[0]), replace=False)
        planes = [volume[i] for i in sorted(indices)]

    ratios = []
    for plane in planes:
        line_means = plane.mean(axis=axis)
        spread = float(np.std(plane))
        if spread > 0:
            ratios.append(float(np.std(line_means) / spread))
    if not ratios:
        return {"stripe_index": float("nan"), "planes_sampled": 0}
    return {
        "stripe_index": float(np.mean(ratios)),
        "stripe_index_max": float(np.max(ratios)),
        "planes_sampled": len(planes),
    }


def resolvability(spacing, feature_um=CAPILLARY_DIAMETER_UM):
    """Whether a feature of the given size is sampled well enough to segment.

    Nyquist puts the bare minimum at two samples across a feature, but a filter
    that must estimate a second derivative needs more than the bare minimum;
    below about three samples a vessel is a blur with no measurable calibre.

    This is the check most worth running before the microscope is packed up.
    """
    spacing = np.asarray(spacing, dtype=float)
    samples = feature_um / spacing
    worst_axis = int(np.argmin(samples))
    return {
        "feature_um": feature_um,
        "samples_per_feature": tuple(float(s) for s in samples),
        "worst_axis": worst_axis,
        "worst_samples": float(samples[worst_axis]),
        "resolved": bool(samples[worst_axis] >= 3.0),
        "nyquist_only": bool(2.0 <= samples[worst_axis] < 3.0),
        "verdict": (
            "resolved" if samples[worst_axis] >= 3.0 else
            "marginal - at Nyquist, calibre will not be measurable"
            if samples[worst_axis] >= 2.0 else
            "UNRESOLVED - features this size cannot be recovered by any filter"
        ),
    }


def compare_channels(channel_a, channel_b, spacing=None, names=("a", "b")):
    """Whether two channels are on comparable footing before any segmentation.

    A comparison assumes the channels differ in biology, not in exposure. Where
    they differ in dynamic range, noise, or attenuation, that difference will
    reappear later as an apparent difference in vessel detection — and the
    asymmetric metrics, precision and recall, are where it will land.
    """
    info_a = inspect_volume(channel_a, spacing)
    info_b = inspect_volume(channel_b, spacing)
    attenuation_a = estimate_attenuation(channel_a, spacing)
    attenuation_b = estimate_attenuation(channel_b, spacing)

    warnings = []
    cnr_a, cnr_b = info_a["contrast_to_noise"], info_b["contrast_to_noise"]
    if np.isfinite(cnr_a) and np.isfinite(cnr_b) and min(cnr_a, cnr_b) > 0:
        if max(cnr_a, cnr_b) / min(cnr_a, cnr_b) > 2.0:
            warnings.append(
                f"contrast-to-noise differs {max(cnr_a, cnr_b) / min(cnr_a, cnr_b):.1f}x "
                f"between channels; the weaker one will lose capillaries first and "
                f"that shows up as an asymmetry in precision vs recall"
            )
    for info, name in ((info_a, names[0]), (info_b, names[1])):
        if info["saturated_fraction"] > 1e-4:
            warnings.append(
                f"channel {name} has {info['saturated_fraction']:.2%} saturated voxels; "
                f"clipped vessel cores flatten the Hessian and suppress vesselness"
            )
    ratio = (attenuation_a["decay_per_um"] + 1e-12) / (attenuation_b["decay_per_um"] + 1e-12)
    if ratio > 1.5 or ratio < 1 / 1.5:
        warnings.append(
            f"attenuation differs {max(ratio, 1 / ratio):.1f}x between channels; "
            f"correct each with its own depth profile, never a shared one"
        )

    return {
        names[0]: info_a, names[1]: info_b,
        f"attenuation_{names[0]}": attenuation_a,
        f"attenuation_{names[1]}": attenuation_b,
        "warnings": warnings,
        "comparable": not warnings,
    }


def suggest_sigmas(spacing, min_radius_um=1.0, max_radius_um=12.0, n=4):
    """A sigma range bracketing the vessel radii, logarithmically spaced.

    Clamped below at the coarsest voxel: searching for a scale finer than the
    sampling finds noise, not vessels.
    """
    floor = float(np.max(spacing)) if spacing is not None else 1.0
    low = max(min_radius_um, floor)
    high = max(max_radius_um, low * 2)
    return [float(s) for s in np.geomspace(low, high, n)]


def intake_report(channel_a, channel_b=None, spacing=None, names=("ch0", "ch1"),
                  feature_um=CAPILLARY_DIAMETER_UM):
    """Everything worth knowing about a new acquisition, in one dict."""
    report = {"channels": {}, "warnings": []}

    report["channels"][names[0]] = inspect_volume(channel_a, spacing)
    report["channels"][names[0]]["attenuation"] = estimate_attenuation(channel_a, spacing)
    report["channels"][names[0]]["stripes"] = stripe_severity(channel_a)

    if channel_b is not None:
        report["channels"][names[1]] = inspect_volume(channel_b, spacing)
        report["channels"][names[1]]["attenuation"] = estimate_attenuation(channel_b, spacing)
        report["channels"][names[1]]["stripes"] = stripe_severity(channel_b)
        comparison = compare_channels(channel_a, channel_b, spacing, names)
        report["comparable"] = comparison["comparable"]
        report["warnings"].extend(comparison["warnings"])

    if spacing is not None:
        report["resolvability"] = resolvability(spacing, feature_um)
        report["suggested_sigmas_um"] = suggest_sigmas(spacing)
        if not report["resolvability"]["resolved"]:
            report["warnings"].insert(0, report["resolvability"]["verdict"])
    else:
        report["warnings"].append(
            "no voxel spacing supplied; resolvability and physical sigmas cannot "
            "be assessed. This is the single most useful number to obtain."
        )
    return report


def format_report(report):
    """Render an intake report as text."""
    lines = ["=" * 68, "ACQUISITION INTAKE REPORT", "=" * 68]

    for name, info in report["channels"].items():
        lines.append(f"\n{name}")
        lines.append(f"  shape {info['shape']}  {info['dtype']}  {info['gigabytes']:.1f} GB")
        if info["spacing_um"]:
            lines.append(f"  voxel size {info['spacing_um']} um")
        lines.append(f"  background {info['background']:.0f}   p99 {info['p99']:.0f}   "
                     f"noise sigma {info['noise_sigma']:.1f}")
        lines.append(f"  contrast-to-noise {info['contrast_to_noise']:.1f}   "
                     f"saturated {info['saturated_fraction']:.3%}")
        attenuation = info["attenuation"]
        lines.append(f"  attenuation half-depth {attenuation['half_depth_um']:.0f} "
                     f"{'um' if attenuation['units'] == 'physical' else 'planes'}   "
                     f"signal retained at deepest plane {attenuation['retained_at_end']:.2f}")
        lines.append(f"  stripe index {info['stripes']['stripe_index']:.3f} "
                     f"(upper bound; aligned vessels inflate it)")

    if "resolvability" in report:
        r = report["resolvability"]
        lines.append(f"\nresolvability of a {r['feature_um']:.0f} um feature")
        lines.append(f"  samples per feature {tuple(round(s, 2) for s in r['samples_per_feature'])}")
        lines.append(f"  verdict: {r['verdict']}")
    if "suggested_sigmas_um" in report:
        lines.append(f"\nsuggested sigmas (um): "
                     f"{[round(s, 2) for s in report['suggested_sigmas_um']]}")

    lines.append("\n" + "-" * 68)
    if report["warnings"]:
        lines.append("WARNINGS")
        for warning in report["warnings"]:
            lines.append(f"  - {warning}")
    else:
        lines.append("no warnings")
    lines.append("=" * 68)
    return "\n".join(lines)
