"""Lightsheet-specific corrections, applied before any segmentation.

Two artefacts in this modality corrupt a channel comparison rather than merely
degrading it, because both act differently on the two channels being compared.

**Stripes.** Absorbers in the illumination path cast shadows along the sheet
axis, producing long straight bands. A Hessian vesselness filter is built to
respond to exactly that geometry, so stripes are not background noise here — they
are false vessels. Their pattern depends on what the light passed through on the
way in, which differs per illumination side and per wavelength.

**Depth attenuation.** Signal falls off with depth into cleared tissue, and 488,
561 and 640 nm fall off at different rates. Left uncorrected, the deeper part of
the brain looks systematically less vascular in one channel than the other, and
that gradient enters the agreement metric as if it were biology. If a single
correction is worth running, it is this one.

Both are estimated on a coarse pyramid level and applied at full resolution: the
stripe pattern and the depth profile are smooth, so nothing is gained by
estimating them from capillary-scale data, and it is far cheaper.
"""

import numpy as np
import scipy.ndimage as ndi

__all__ = ["destripe", "tissue_mask", "depth_profile", "correct_depth_attenuation",
           "smooth_profile"]


# --------------------------------------------------------------------------
# stripes
# --------------------------------------------------------------------------

def destripe(image, sigma, level=6, wavelet="db5", axis=0):
    """Remove stripe artefacts with the wavelet-FFT method.

    Decompose into wavelet bands, damp the frequency components that run along
    the stripe direction in each band's detail coefficients, then reconstruct.
    Because stripes are narrow in frequency along one axis and broad along the
    other, this removes them while leaving structures that merely happen to be
    elongated — real vessels included — largely intact.

    Args:
        image: 2D array, a single plane.
        sigma: width of the frequency-domain notch. Larger removes more stripe
            energy and more real signal with it. Start around 3-8 and check
            against a region you know is vessel.
        level: wavelet decomposition depth. Deeper reaches broader stripes.
        wavelet: PyWavelets name. db5 is the usual choice for this method.
        axis: the axis stripes run *along*. For a sheet illuminating along x with
            planes stored (y, x), stripes run along x, so axis=1.

    Returns:
        Destriped plane, float32.

    **Known limitation.** This cannot distinguish a stripe from a vessel that runs
    the full length of the image parallel to the illumination axis — both are
    constant along the same direction, so such a vessel is attenuated along with
    the artefact. Finite-length and obliquely oriented vessels survive. If the
    vasculature has a strong preferred orientation aligned with the sheet, check
    a known region before trusting the output.

    Reference: Munch, Trtik, Marone, Stampanoni, "Stripe and ring artifact
    removal with combined wavelet-Fourier filtering", Optics Express 17(10), 2009.
    """
    import pywt

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"destripe operates on single planes, got {image.ndim}D")
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    # Work in an orientation where stripes run along rows, then rotate back.
    working = image if axis == 0 else image.T

    # Requesting more levels than the plane supports makes every coefficient a
    # boundary artefact; pywt only warns, so clamp instead.
    usable = pywt.dwt_max_level(min(working.shape), pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(int(level), usable))

    coefficients = pywt.wavedec2(working, wavelet, level=level, mode="symmetric")
    filtered = [coefficients[0]]
    for horizontal, vertical, diagonal in coefficients[1:]:
        # Stripes along rows concentrate in the *vertical* detail band.
        filtered.append((horizontal, _damp_stripe_band(vertical, sigma), diagonal))

    restored = pywt.waverec2(filtered, wavelet, mode="symmetric")
    restored = restored[:working.shape[0], :working.shape[1]]
    return np.asarray(restored if axis == 0 else restored.T, dtype=np.float32)


def _damp_stripe_band(band, sigma):
    """Suppress the low-frequency ridge that stripes occupy in a detail band."""
    spectrum = np.fft.fftshift(np.fft.fft(band, axis=0), axes=0)
    rows = spectrum.shape[0]
    coordinate = np.arange(rows) - rows // 2
    damping = 1.0 - np.exp(-(coordinate ** 2) / (2.0 * sigma ** 2))
    spectrum *= damping[:, None]
    return np.real(np.fft.ifft(np.fft.ifftshift(spectrum, axes=0), axis=0))


# --------------------------------------------------------------------------
# depth attenuation
# --------------------------------------------------------------------------

def tissue_mask(volume, percentile=25, min_fraction=0.01):
    """Coarse mask separating tissue from surrounding medium.

    Deliberately crude — it exists to decide which voxels contribute to the depth
    profile, not to delineate anatomy. Estimate it on a coarse pyramid level.

    A threshold at a low percentile of the *non-empty* voxels keeps dim deep
    tissue inside the mask, which matters: masking tightly around bright tissue
    would exclude exactly the attenuated regions the profile needs to measure,
    and the correction would then flatten a curve fitted only to its own bright end.
    """
    volume = np.asarray(volume)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("volume has no finite values")
    threshold = np.percentile(finite, percentile)
    mask = volume > threshold
    if mask.mean() < min_fraction:
        # Threshold landed above almost everything; fall back to Otsu-like split.
        threshold = float(finite.mean())
        mask = volume > threshold
    return mask


def depth_profile(volume, mask=None, axis=0, statistic="median"):
    """Per-plane signal level along the depth axis, measured inside tissue.

    The median is the default rather than the mean because vessels are a bright
    minority: a mean tracks vessel density as much as illumination, so correcting
    by it would partly divide out the signal being measured.

    Returns:
        1D array, one value per plane along `axis`. Planes with no masked voxels
        give nan, which `smooth_profile` interpolates across.
    """
    volume = np.asarray(volume)
    if mask is None:
        mask = tissue_mask(volume)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != volume.shape:
        raise ValueError(f"mask shape {mask.shape} != volume shape {volume.shape}")

    reducer = {"median": np.median, "mean": np.mean}.get(statistic)
    if reducer is None:
        raise ValueError(f"statistic must be 'median' or 'mean', got {statistic!r}")

    volume = np.moveaxis(volume, axis, 0)
    mask = np.moveaxis(mask, axis, 0)

    profile = np.full(volume.shape[0], np.nan, dtype=np.float64)
    for index in range(volume.shape[0]):
        values = volume[index][mask[index]]
        if values.size:
            profile[index] = reducer(values)
    return profile


def smooth_profile(profile, sigma=10.0):
    """Interpolate gaps and smooth, so the correction cannot chase noise.

    A profile estimated plane by plane carries its own sampling noise. Dividing by
    it unsmoothed would stamp that noise into every plane as a global gain change,
    which is worse than the attenuation being corrected.
    """
    profile = np.asarray(profile, dtype=np.float64)
    valid = np.isfinite(profile) & (profile > 0)
    if not valid.any():
        raise ValueError("profile has no usable values")

    indices = np.arange(profile.size)
    filled = np.interp(indices, indices[valid], profile[valid])
    if sigma > 0:
        filled = ndi.gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    return filled


def correct_depth_attenuation(volume, profile=None, mask=None, axis=0,
                              sigma=10.0, reference=None, dtype=np.float32):
    """Divide out the depth-dependent signal decay.

    Args:
        volume: the data to correct.
        profile: precomputed depth profile. None estimates one from `volume`.
            Pass an explicitly precomputed profile when correcting full-resolution
            data using a profile measured on a coarse level, which is the intended
            use.
        mask: tissue mask for profile estimation, if `profile` is not given.
        axis: depth axis.
        sigma: smoothing applied to the profile before division.
        reference: level to normalise to. None uses the profile's maximum, so the
            brightest depth is left unchanged and deeper planes are scaled up.

    Returns:
        Corrected volume.

    Correct each channel with *its own* profile. Sharing one profile across
    channels reintroduces precisely the asymmetry this is meant to remove, since
    the wavelengths attenuate at different rates.
    """
    volume = np.asarray(volume)
    if profile is None:
        profile = depth_profile(volume, mask=mask, axis=axis)
    profile = smooth_profile(profile, sigma=sigma)

    depth = volume.shape[axis]
    if profile.size != depth:
        # Profile measured on a coarser level: resample onto this axis.
        profile = np.interp(np.linspace(0, 1, depth),
                            np.linspace(0, 1, profile.size), profile)

    reference = float(np.max(profile)) if reference is None else float(reference)
    if reference <= 0:
        raise ValueError("reference level must be positive")

    gain = reference / np.clip(profile, 1e-6, None)
    shape = [1] * volume.ndim
    shape[axis] = depth
    return (volume * gain.reshape(shape)).astype(dtype, copy=False)
