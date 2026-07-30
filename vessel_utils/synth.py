"""Synthetic vasculature with a simulated lightsheet acquisition.

This exists because there is no hand-annotated ground truth. A phantom has a
known answer by construction, so it gives two things nothing else here can:

  1. **Real accuracy numbers.** Dice, clDice and recall-by-calibre against a
     mask that is correct by definition, rather than against the other channel.
  2. **Attribution.** Each degradation is applied separately, so you can ask
     which one actually costs you recall — and find that it is the z sampling,
     or the noise floor, rather than the filter.

The obvious caveat: a phantom measures performance on the phantom. It is
evidence about the pipeline, not proof about the brain, and it is only as good as
the degradation model. Treat a number from here as an upper bound, and keep the
generator's assumptions in view — they are all parameters, deliberately.

The tree follows Murray's law at bifurcations, r_parent^3 = r_1^3 + r_2^3, which
is what actual vasculature approximately obeys. That matters more than it might
seem: it sets the radius distribution, and the radius distribution is exactly
what determines whether a pipeline's failures land on capillaries.
"""

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

__all__ = ["Segment", "vascular_tree", "render_tree", "simulate_acquisition",
           "default_root_radius",
           "phantom"]


@dataclass(frozen=True)
class Segment:
    """A tapering vessel segment between two points, in physical units."""
    start: tuple
    end: tuple
    radius_start: float
    radius_end: float


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _perpendicular(direction, rng):
    """A random unit vector perpendicular to `direction`."""
    for _ in range(10):
        candidate = rng.normal(size=3)
        projected = candidate - np.dot(candidate, direction) * direction
        norm = np.linalg.norm(projected)
        if norm > 1e-6:
            return projected / norm
    # Degenerate only if direction is not finite; fall back to any axis.
    return np.array([1.0, 0.0, 0.0])


def default_root_radius(extent, length_factor=6.0, murray_exponent=3.0,
                        n_levels=7):
    """Trunk radius that lets a tree of `n_levels` span the box.

    Segment length is `length_factor * radius` and radius shrinks by
    2**(-1/murray_exponent) per level, so the reach along the growth axis is a
    geometric series. Sizing the trunk from the box rather than fixing it means a
    phantom is never one segment that swallows the whole volume — which is what
    happens with a trunk chosen independently of the extent.
    """
    ratio = 0.5 ** (1.0 / murray_exponent)
    reach = length_factor * (1.0 - ratio ** n_levels) / (1.0 - ratio)
    return float(max(extent)) / reach


def vascular_tree(extent, root_radius=None, min_radius=1.5, n_levels=7,
                  murray_exponent=3.0, length_factor=6.0, branch_angle=0.6,
                  asymmetry=0.15, tortuosity=0.15, seed=0):
    """Grow a bifurcating vessel tree inside a physical bounding box.

    Args:
        extent: (z, y, x) size of the region in micrometres.
        root_radius: radius of the trunk, micrometres. None derives one that lets
            the tree span the box — see `default_root_radius`.
        min_radius: stop branching below this radius. Set it to your expected
            capillary radius; segments near it are what the pipeline will
            struggle with, and what the benchmark should therefore contain.
        n_levels: maximum bifurcation depth.
        murray_exponent: 3.0 gives Murray's law. Lower values make daughters
            larger relative to the parent, i.e. a less rapidly narrowing tree.
        length_factor: segment length as a multiple of its radius.
        branch_angle: bifurcation half-angle in radians.
        asymmetry: fractional difference in the two daughters' flow split. 0
            gives symmetric bifurcations, which look unnaturally regular.
        tortuosity: how much each segment wanders from straight.

    Returns:
        List of `Segment`, coordinates in micrometres relative to the box origin.
    """
    extent = np.asarray(extent, dtype=float)
    if root_radius is None:
        root_radius = default_root_radius(extent, length_factor, murray_exponent,
                                          n_levels)
    if root_radius <= min_radius:
        raise ValueError(
            f"root_radius {root_radius:.2f} must exceed min_radius {min_radius}; "
            f"the box ({extent}) is too small for a tree of {n_levels} levels"
        )
    if not 0 <= asymmetry < 0.5:
        raise ValueError(f"asymmetry must lie in [0, 0.5), got {asymmetry}")

    rng = np.random.default_rng(seed)
    segments = []

    # Enter through the middle of the face perpendicular to the longest axis, so
    # the tree has the most room to develop before it leaves the box.
    growth_axis = int(np.argmax(extent))
    start = extent / 2.0
    start[growth_axis] = 0.0
    direction = np.zeros(3)
    direction[growth_axis] = 1.0

    # A segment may leave the box and be clipped by the renderer — real vessels
    # cross the edge of a field of view. Only stop growing once the segment's
    # *start* is outside, otherwise one long trunk terminates every branch.
    margin = extent * 0.25

    def grow(point, direction, radius, level):
        if level >= n_levels or radius < min_radius:
            return
        if np.any(point < -margin) or np.any(point > extent + margin):
            return
        length = length_factor * radius
        wander = _perpendicular(direction, rng) * tortuosity * length
        end = point + direction * length + wander

        split = 0.5 + rng.uniform(-asymmetry, asymmetry)
        child_a = radius * split ** (1.0 / murray_exponent)
        child_b = radius * (1.0 - split) ** (1.0 / murray_exponent)

        # Uniform radius along the segment, stepping down only at the
        # bifurcation. Tapering the parent toward max(child_a, child_b) would
        # match the larger daughter and leave an exact discontinuity at the
        # smaller one, biasing the rendered calibre near every asymmetric
        # bifurcation toward the larger sibling — which matters here because
        # recall-by-calibre is the measurement this phantom exists to support.
        segments.append(Segment(tuple(point), tuple(end), radius, radius))

        forward = _unit(end - point)
        axis = _perpendicular(forward, rng)
        for child_radius, sign in ((child_a, 1.0), (child_b, -1.0)):
            angle = branch_angle * (1.0 + rng.uniform(-0.3, 0.3)) * sign
            new_direction = _unit(forward * np.cos(angle) + axis * np.sin(angle))
            grow(end, new_direction, child_radius, level + 1)

    grow(start, direction, root_radius, 0)
    if not segments:
        raise ValueError("no segments generated; check extent and radii")
    return segments


def render_tree(segments, shape, spacing):
    """Rasterise segments into a boolean mask.

    Each segment is a tapering capsule: a voxel belongs to it when its distance
    to the segment's axis is under the radius interpolated along that axis. Only
    the segment's bounding box is evaluated, so cost scales with vessel volume
    rather than with the volume of the box.
    """
    shape = tuple(int(s) for s in shape)
    spacing = np.asarray(spacing, dtype=float)
    if len(shape) != 3 or spacing.size != 3:
        raise ValueError("render_tree works in 3D")

    mask = np.zeros(shape, dtype=bool)
    grids = [np.arange(n) * s for n, s in zip(shape, spacing)]

    for segment in segments:
        start = np.asarray(segment.start, dtype=float)
        end = np.asarray(segment.end, dtype=float)
        radius = max(segment.radius_start, segment.radius_end)

        low = np.minimum(start, end) - radius
        high = np.maximum(start, end) + radius
        lo_idx = np.maximum(np.floor(low / spacing).astype(int), 0)
        hi_idx = np.minimum(np.ceil(high / spacing).astype(int) + 1, shape)
        if np.any(lo_idx >= hi_idx):
            continue

        local = np.meshgrid(*[g[a:b] for g, a, b in zip(grids, lo_idx, hi_idx)],
                            indexing="ij")
        points = np.stack(local, axis=-1)

        axis = end - start
        length_squared = float(np.dot(axis, axis))
        if length_squared <= 0:
            continue
        offset = points - start
        t = np.clip((offset @ axis) / length_squared, 0.0, 1.0)
        closest = start + t[..., None] * axis
        distance = np.linalg.norm(points - closest, axis=-1)
        local_radius = segment.radius_start + t * (segment.radius_end
                                                   - segment.radius_start)

        window = tuple(slice(a, b) for a, b in zip(lo_idx, hi_idx))
        mask[window] |= distance <= local_radius

    return mask


def simulate_acquisition(mask, spacing, signal=800.0, background=100.0,
                         psf_sigma=(2.0, 0.6, 0.6), attenuation=0.0015,
                         stripe_strength=0.0, stripe_density=0.12,
                         read_noise=8.0, gain=1.0, seed=0, dtype=np.float32):
    """Degrade a ground-truth mask into something resembling a real acquisition.

    Each effect is independent and off by default where it can be, so a benchmark
    can attribute a loss in recall to one cause rather than to "the simulation".

    Args:
        mask: boolean ground truth.
        spacing: (z, y, x) voxel size in micrometres.
        signal, background: intensity of vessel and parenchyma before noise.
        psf_sigma: point spread function width in micrometres, per axis. Lightsheet
            PSFs are markedly worse along the detection axis, which is why the
            default is anisotropic; this is usually what limits capillary
            separation, not the segmentation filter.
        attenuation: exponential decay per micrometre of depth. 0 disables it.
        stripe_strength: peak fractional shadowing. 0 disables stripes.
        read_noise: Gaussian read noise standard deviation.
        gain: photons per intensity unit for the Poisson shot noise. Lower means
            noisier; this is the knob that decides whether capillaries are
            detectable at all.

    Returns:
        Simulated volume.
    """
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)

    volume = np.where(mask, signal, background).astype(np.float64)

    psf_voxels = np.asarray(psf_sigma, dtype=float) / spacing
    if np.any(psf_voxels > 0):
        volume = ndi.gaussian_filter(volume, sigma=psf_voxels, mode="nearest")

    if attenuation:
        depth_um = np.arange(mask.shape[0]) * spacing[0]
        volume *= np.exp(-attenuation * depth_um)[:, None, None]

    if stripe_strength:
        # Shadows cast along the illumination axis: constant down the last axis.
        shadow = np.ones(mask.shape[1])
        struck = rng.random(mask.shape[1]) < stripe_density
        shadow[struck] -= stripe_strength * rng.random(struck.sum())
        volume *= shadow[None, :, None]

    if gain > 0:
        volume = rng.poisson(np.clip(volume * gain, 0, None)) / gain
    if read_noise:
        volume = volume + rng.normal(0.0, read_noise, size=volume.shape)

    return np.clip(volume, 0, None).astype(dtype)


def phantom(shape=(64, 128, 128), spacing=(3.0, 0.75, 0.75), seed=0,
            tree_kwargs=None, acquisition_kwargs=None):
    """A simulated acquisition and its exact ground-truth mask.

    Returns:
        (volume, mask, segments) — the segments are returned so a benchmark can
        report performance against the true radius of each vessel rather than
        against a radius estimated from the mask.
    """
    shape = tuple(int(s) for s in shape)
    spacing = tuple(float(s) for s in spacing)
    extent = [n * s for n, s in zip(shape, spacing)]

    segments = vascular_tree(extent, seed=seed, **(tree_kwargs or {}))
    mask = render_tree(segments, shape, spacing)
    volume = simulate_acquisition(mask, spacing, seed=seed,
                                  **(acquisition_kwargs or {}))
    return volume, mask, segments
