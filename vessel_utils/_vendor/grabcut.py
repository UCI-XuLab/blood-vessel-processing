# Vendored from UCI-XuLab-RegTools @ 9f2e5fb
# (regtools/tissue_masking/core/grabcut.py). Do not edit here except the
# masking_thresholds import below; re-sync by re-copying. See _vendor/README.md.
"""Training-free entropy-guided GrabCut tissue masking.

The most accurate masker in this project: **0.9817 mean Dice / 0.9373 worst over 33
ground-truth cases across 11 intensity sources** - three Nissl builds, serial two-photon,
light-sheet fluorescence, three MRI contrasts, MERFISH, CCF average template and Waxholm
T2*. No trained model, no weights, deterministic (OpenCV's RNG is reset to a fixed seed
before every GrabCut call).

Per section: percentile-normalize to a tissue score, compute a local rank-entropy image,
derive three GrabCut seeds from them, refine with ``cv2.grabCut``, then clean specks and
fill holes.

**When to use this rather than the cheap threshold.** ``masking.compute_tissue_mask`` uses
the same seed (percentile-normalize + Multi-Otsu-4 + background guard) without the GrabCut
refinement, and that seed is what fixed the failures this project was chasing. GrabCut buys
the last percent or two of boundary accuracy for roughly 1.7 s per megapixel, of which
77-88% is ``cv2.grabCut`` itself. Take it where a mask is a deliverable; skip it where the
consumer reduces the mask to a scalar (the 2D registration ``foreground_init`` uses only a
foreground area and centroid, so the refinement changes nothing it can see) or where a GUI
must stay responsive - ``analysis_scale=0.5`` gives ~3x for a mask agreement of 0.993 if
both matter.

Scale stability is the property that distinguishes it: across an 8x resolution range on a
B0039 STPT section its foreground moved 0.1890 -> 0.1867, where a Triangle threshold on the
same images went 0.1858 -> 0.0003.

Vendoring
---------
This module is self-contained except for ONE companion file:
``regtools/utils/masking_thresholds.py`` (``BACKGROUND_PERCENTILE`` and
``_percentile_rank``, imported below). The two ship together on purpose - the shared
background constant is what keeps this masker and the 2D-registration foreground guard
from drifting - so to vendor the method, copy BOTH files. External runtime dependencies
are only numpy, OpenCV (cv2), scipy, and scikit-image. The default ``EntropyGrabCutConfig``
IS the benchmark configuration (percentile 1-99.5 normalize + Multi-Otsu-4 seed); construct
it with no arguments to reproduce the published atlas-segmentation Dice.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes
from skimage.filters import threshold_li, threshold_multiotsu, threshold_otsu, threshold_triangle
from skimage.filters.rank import entropy
from skimage.measure import label as cc_label
from skimage.morphology import disk

# The rank the definite-background seed claims outright, and the floor the seed guard tests
# against. The two rules must not be able to drift apart, because the entire point of the
# guard is that they contradict each other.
#
# Taken from masking_thresholds rather than declared here, so this number and the one the 2D
# registration foreground guards on are the same BY CONSTRUCTION. That module's docstring
# states the two cannot drift; a second literal in this file is precisely what would have
# made that statement false while looking correct.
from .masking_thresholds import BACKGROUND_PERCENTILE as _BACKGROUND_PERCENTILE
from .masking_thresholds import _percentile_rank


@dataclass(frozen=True)
class EntropyGrabCutConfig:
    low_percentile: float = 1.0
    high_percentile: float = 99.5
    entropy_radius: int = 5
    grabcut_iterations: int = 5
    rng_seed: int = 20260723
    min_speck_pixels: int = 64
    min_speck_fraction: float = 1e-4
    min_relative_component_fraction: float = 0.01
    fill_holes: bool = True
    analysis_scale: float = 1.0
    entropy_tiles: int | None = None
    seed_threshold: str = "multiotsu4"
    gate_classes: int = 3
    seed_background_guard: bool = True
    # Stop once the mask settles instead of always running `grabcut_iterations`. GrabCut is
    # 77-88% of runtime and scales linearly in iterations, so this is the only speed lever
    # that does not cost accuracy. Measured over all 21 corpus cases: it converges on 21/21
    # at a mean of 3.0 iterations for 1.23x, and preserves atlas accuracy exactly - mean Dice
    # 0.9855 against the five-iteration baseline's 0.9855.
    #
    # A fixed count is faster and worse. Three iterations gives 1.44x but drops atlas mean to
    # 0.9838 and worst-case to 0.9633 (from 0.9685), because one case - CCF average template
    # coronal 1100 - genuinely has not settled by then. The adaptive rule detects that and
    # keeps going, which a constant cannot.
    grabcut_convergence_fraction: float | None = 0.001
    grabcut_max_iterations: int = 20


@dataclass
class EntropyGrabCutCoreResult:
    raw_mask: np.ndarray
    seed_threshold: float
    fallback: str
    timings_s: dict[str, float]
    grabcut_iterations_run: int = 0
    converged: bool = False
    seed_classes_used: int | None = None
    # Percentile rank of the seed threshold within the score. The definite-*background*
    # seed claims everything at or below the 20th percentile, so a seed below that rank is
    # proposing as probable foreground the very pixels the background rule rejects. That
    # contradiction is the signature of a seed that has fallen into a background spike.
    seed_percentile: float | None = None
    gate_classes_used: int | None = None
    # Whether Li won the max() that forms the definite-foreground gate. Measured across the
    # corpus: Li never wins - Multi-Otsu took all 168 configurations - so the gate's
    # inertness is not explained by Li dominating it. The upper Multi-Otsu threshold is
    # simply stable across class counts, unlike the lower one the seed uses.
    gate_li_dominated: bool | None = None
    # Class counts the background guard rejected before settling on the one used. Empty on
    # every case in the benchmark corpus; a non-empty value means this image would have
    # produced a runaway mask at the configured count.
    seed_guard_rejected: tuple[int, ...] = ()


@dataclass
class EntropyGrabCutResult:
    mask: np.ndarray
    raw_mask: np.ndarray
    seed_threshold: float
    fallback: str
    timings_s: dict[str, float]
    grabcut_iterations_run: int = 0
    converged: bool = False
    seed_classes_used: int | None = None
    seed_percentile: float | None = None
    gate_classes_used: int | None = None
    gate_li_dominated: bool | None = None
    # Class counts the background guard rejected before settling on the one used. Empty on
    # every case in the benchmark corpus; a non-empty value means this image would have
    # produced a runaway mask at the configured count.
    seed_guard_rejected: tuple[int, ...] = ()


@dataclass
class EntropyGrabCutRequest:
    label: str
    image: np.ndarray
    polarity: str = "bright"


def resolve_worker_budget(
    request_count: int,
    *,
    max_workers: int | None = None,
    opencv_threads_per_worker: int | None = None,
    cpu_count: int | None = None,
) -> tuple[int, int]:
    """Choose a bounded process/OpenCV thread budget for a batch."""
    cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    if request_count <= 0:
        return 0, 0
    workers = min(
        request_count,
        max(1, int(max_workers)) if max_workers is not None else min(4, cpus),
    )
    threads = (
        max(1, cpus // workers)
        if opencv_threads_per_worker is None
        else max(1, int(opencv_threads_per_worker))
    )
    if workers * threads > cpus:
        raise ValueError(
            f"worker budget would oversubscribe {cpus} logical CPUs: "
            f"{workers} workers x {threads} OpenCV threads"
        )
    return workers, threads


def percentile_normalize(array: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    """Map finite percentile limits to [0, 1], following the notebook exactly."""
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0, 1).astype(
        np.float32
    )


def _validate_raw_image(image: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "image must be a numeric 2D grayscale array or have at least 3 RGB channels"
        ) from exc
    if values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0:
        return values
    if (
        values.ndim == 3
        and values.shape[0] > 0
        and values.shape[1] > 0
        and values.shape[2] >= 3
    ):
        return values[..., :3]
    raise ValueError(
        "image must be a nonempty 2D grayscale array or have at least 3 RGB channels"
    )


def _validate_polarity(polarity: str) -> None:
    if polarity not in {"bright", "dark"}:
        raise ValueError(f"Unknown polarity: {polarity!r}")


def border_band(shape: tuple[int, int], fraction: float = 0.02) -> np.ndarray:
    """Return the notebook's outer-image background band."""
    width = max(1, int(np.ceil(min(shape) * fraction)))
    band = np.zeros(shape, dtype=bool)
    band[:width] = True
    band[-width:] = True
    band[:, :width] = True
    band[:, -width:] = True
    return band


def tissue_score(
    image: np.ndarray, polarity: str, low: float = 1.0, high: float = 99.5
) -> np.ndarray:
    """Normalize raw grayscale/RGB data so larger values indicate likely tissue."""
    _validate_polarity(polarity)
    image = _validate_raw_image(image)
    if image.ndim == 3:
        channels = np.stack(
            [percentile_normalize(image[..., channel], low, high) for channel in range(3)],
            axis=-1,
        )
        if polarity == "bright":
            return np.max(channels, axis=-1).astype(np.float32)
        band = border_band(channels.shape[:2])
        background = np.median(channels[band], axis=0)
        epsilon = 1.0 / 255.0
        optical_distance = np.linalg.norm(
            np.maximum(np.log((background + epsilon) / (channels + epsilon)), 0.0),
            axis=-1,
        )
        return percentile_normalize(optical_distance, low, high)

    normalized = percentile_normalize(image, low, high)
    if polarity == "bright":
        return normalized
    return (1.0 - normalized).astype(np.float32)


def clean_mask(mask: np.ndarray, config: EntropyGrabCutConfig | None = None) -> np.ndarray:
    """Remove small 8-connected components, then fill retained-mask holes."""
    config = config or EntropyGrabCutConfig()
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    labels = cc_label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    largest = int(counts[1:].max()) if counts.size > 1 else 0
    if largest == 0:
        return np.zeros_like(mask)
    min_size = max(
        config.min_speck_pixels,
        int(np.ceil(mask.size * config.min_speck_fraction)),
        int(np.ceil(largest * config.min_relative_component_fraction)),
    )
    keep = np.flatnonzero(counts >= min_size)
    keep = keep[keep != 0]
    retained = np.isin(labels, keep)
    if config.fill_holes:
        retained = binary_fill_holes(retained)
    return retained.astype(bool, copy=False)


def _validate_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("score must be a nonempty 2D array")
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)


def _resolve_entropy_tiles(tiles: int | None, height: int, radius: int) -> int:
    """Pick a row-tile count, bounded so every tile is wider than its own halo."""
    if tiles is not None and tiles < 1:
        raise ValueError("tiles must be at least 1")
    # A tile must be tall enough that its halo does not dominate it, or tiling costs
    # more in duplicated halo work than it saves.
    ceiling = max(1, height // (4 * radius + 1))
    if tiles is None:
        # Auto: follow the OpenCV thread budget. In a batch worker that budget is
        # already set by _initialize_worker, so tiling cannot oversubscribe the CPUs
        # that resolve_worker_budget handed out.
        tiles = max(1, cv2.getNumThreads())
    return max(1, min(tiles, ceiling))


def local_entropy_image(
    score: np.ndarray, radius: int = 5, *, tiles: int | None = 1
) -> np.ndarray:
    """Compute the notebook's rank entropy image from a normalized tissue score.

    Rank entropy is a local operator: a pixel's value depends only on the disk of
    ``radius`` around it. Splitting the image into row bands and giving each band a
    halo of exactly ``radius`` therefore reproduces the whole-image result bitwise,
    which is what makes ``tiles`` a pure speed knob rather than an approximation.
    ``tiles=None`` follows the current OpenCV thread budget; ``tiles=1`` disables it.
    """
    if radius < 1:
        raise ValueError("radius must be at least 1")
    score_u8 = np.round(np.clip(score, 0, 1) * 255).astype(np.uint8)
    footprint = disk(radius)
    height = score_u8.shape[0]
    resolved = _resolve_entropy_tiles(tiles, height, radius)
    if resolved <= 1:
        return entropy(score_u8, footprint).astype(np.float32)

    bounds = [
        (index * height // resolved, (index + 1) * height // resolved)
        for index in range(resolved)
    ]
    output = np.empty(score_u8.shape, dtype=np.float32)

    def fill(band: tuple[int, int]) -> None:
        start, stop = band
        padded_start = max(0, start - radius)
        padded_stop = min(height, stop + radius)
        tile = entropy(score_u8[padded_start:padded_stop], footprint)
        offset = start - padded_start
        output[start:stop] = tile[offset: offset + (stop - start)]

    with ThreadPoolExecutor(max_workers=resolved) as executor:
        for _ in executor.map(fill, bounds):
            pass
    return output


def _parse_multiotsu_classes(kind: str) -> int:
    """Parse ``"multiotsu<N>"`` into its class count.

    The class count is the one parameter of this method that rests on the benchmark
    corpus rather than on theory, so it has to be sweepable to be defensible. Naming it
    in the seed string rather than adding a second field keeps one source of truth: the
    seed kind and its class count cannot disagree.
    """
    if not kind.startswith("multiotsu"):
        raise ValueError(f"Unknown seed_threshold: {kind!r}")
    suffix = kind[len("multiotsu"):]
    if not suffix.isdigit():
        raise ValueError(f"Unknown seed_threshold: {kind!r}")
    classes = int(suffix)
    if classes < 2:
        raise ValueError(f"seed_threshold needs at least 2 classes: {kind!r}")
    return classes


def _gate_high_threshold(score: np.ndarray, classes: int) -> tuple[float, int]:
    """Upper Multi-Otsu threshold for the definite-foreground gate, with a class stepdown.

    The seed has stepped its class count down since it moved to Multi-Otsu; this gate never
    did. Without the stepdown a near-binary image raises here, and because the handler
    treats that as a seeding failure the whole method degrades to a bare threshold with
    GrabCut never running - a far worse outcome than thresholding with one class fewer.

    Sweeping this count across the benchmark corpus moved nothing (21/21 cases indifferent
    at every count from 2 to 5), so this is an exception-safety fix and not a retune: on
    any image where the original call succeeds, the result is unchanged.
    """
    for candidate in range(classes, 1, -1):
        try:
            return float(threshold_multiotsu(score, classes=candidate)[-1]), candidate
        except ValueError as exc:
            if "different values" not in str(exc):
                raise
            continue
    return float(threshold_otsu(score)), 0


def _seed_threshold(score: np.ndarray, kind: str) -> tuple[float, int | None]:
    """Threshold that marks probable foreground before GrabCut refines it.

    The seed does not need to be accurate - GrabCut can pull a generous seed back but can
    never recover tissue the seed never proposed - so it needs to be permissive and, above
    all, stable. Triangle is neither reliably: it locates a geometric elbow that ceases to
    exist once the histogram resolves a hard background spike, which is why its behaviour
    flips with downsample factor and histogram bin count. Multi-Otsu optimises class
    variance over the whole histogram and has no degenerate case here.

    Four classes is empirical, and the neighbours fail on both sides: three classes
    under-segments Ctl1 DAPI (0.76 -> 0.26 foreground) and five over-segments the
    low-tissue B0039 sections into a runaway mask, besides costing 3 s per call.

    Returns ``(threshold, classes_used)``. ``classes_used`` is ``None`` for the Triangle
    seed (class stepdown does not apply to it), the number of classes that Multi-Otsu
    actually thresholded with (counting down from the requested count to 2), or ``0`` if
    Multi-Otsu could not threshold at any class count and the caller fell all the way back
    to plain Otsu. A degraded class count materially changes the seed (see above), so this
    must be visible to callers rather than silently absorbed.
    """
    if kind == "triangle":
        return float(threshold_triangle(score)), None
    requested = _parse_multiotsu_classes(kind)
    # Multi-Otsu needs at least as many discretized levels as classes. A near-binary
    # image has fewer, so step the class count down rather than let the seed raise -
    # a seed that raises leaves the fallback with no threshold and yields an empty mask.
    # Only that specific too-few-levels failure is degraded past - any other ValueError
    # (a real bug, not a low-level-count image) must still surface to the caller.
    for classes in range(requested, 1, -1):
        try:
            return float(threshold_multiotsu(score, classes=classes)[0]), classes
        except ValueError as exc:
            if "different values" not in str(exc):
                raise
            continue
    return float(threshold_otsu(score)), 0


def _grabcut_labels_and_features(
    score: np.ndarray,
    entropy_image: np.ndarray,
    seed: float,
    sure_fg: np.ndarray,
    sure_bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(score.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    labels[score > seed] = cv2.GC_PR_FGD
    labels[sure_bg] = cv2.GC_BGD
    labels[sure_fg] = cv2.GC_FGD

    score_u8 = np.round(np.clip(score, 0, 1) * 255).astype(np.uint8)
    entropy_u8 = np.round(percentile_normalize(entropy_image) * 255).astype(np.uint8)
    grad_x = cv2.Sobel(score_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(score_u8, cv2.CV_32F, 0, 1, ksize=3)
    gradient_u8 = np.round(
        percentile_normalize(cv2.magnitude(grad_x, grad_y)) * 255
    ).astype(np.uint8)
    return labels, np.dstack([score_u8, entropy_u8, gradient_u8])


def _grabcut_foreground(labels: np.ndarray) -> np.ndarray:
    return (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)


def _run_grabcut(
    features: np.ndarray, labels: np.ndarray, config: EntropyGrabCutConfig
) -> tuple[np.ndarray, int, bool]:
    """Refine ``labels`` in place, returning the foreground mask, iterations run, and
    whether the loop actually converged.

    With ``grabcut_convergence_fraction`` unset this is a single fixed-length
    ``cv2.grabCut`` call, exactly as the frozen notebook baseline; no convergence
    criterion is evaluated in that mode, so ``converged`` is always ``False``. With the
    fraction set, the same work is issued one iteration at a time so the loop can stop
    once the mask settles: an initial ``GC_INIT_WITH_MASK`` pass followed by ``GC_EVAL``
    passes that reuse the same GMM state. Both forms are bitwise identical for a given
    iteration count, so the stopping rule changes when GrabCut stops, not what it
    computes.

    GrabCut's EM is not monotone, so hitting ``grabcut_max_iterations`` without ever
    satisfying the tolerance and converging exactly on the last allowed iteration both
    report that same iteration count - only the returned ``converged`` flag tells them
    apart. Callers must not infer convergence from ``iterations_run >= max_iterations``.
    """
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(config.rng_seed)
    if config.grabcut_convergence_fraction is None:
        cv2.grabCut(
            features,
            labels,
            None,
            background_model,
            foreground_model,
            config.grabcut_iterations,
            cv2.GC_INIT_WITH_MASK,
        )
        return _grabcut_foreground(labels), config.grabcut_iterations, False

    cv2.grabCut(
        features,
        labels,
        None,
        background_model,
        foreground_model,
        1,
        cv2.GC_INIT_WITH_MASK,
    )
    previous = _grabcut_foreground(labels)
    tolerance = config.grabcut_convergence_fraction * previous.size
    for iteration in range(2, config.grabcut_max_iterations + 1):
        cv2.grabCut(
            features, labels, None, background_model, foreground_model, 1, cv2.GC_EVAL
        )
        current = _grabcut_foreground(labels)
        changed = int(np.count_nonzero(current != previous))
        previous = current
        if changed <= tolerance:
            return current, iteration, True
    return previous, config.grabcut_max_iterations, False


def _threshold_fallback_result(
    score: np.ndarray,
    reason: str,
    *,
    kind: str,
    entropy_s: float,
    seed_features_s: float = 0.0,
) -> EntropyGrabCutCoreResult:
    """Fall back to the configured seed threshold alone, with no GrabCut refinement."""
    try:
        threshold, seed_classes_used = _seed_threshold(score, kind)
        raw = score > threshold
    except ValueError:
        threshold = float("nan")
        seed_classes_used = None
        raw = np.zeros_like(score, dtype=bool)
    return EntropyGrabCutCoreResult(
        raw_mask=np.asarray(raw, dtype=bool),
        seed_threshold=threshold,
        fallback=reason,
        timings_s={
            "entropy": entropy_s,
            "seed_features": seed_features_s,
            "grabcut": 0.0,
        },
        seed_classes_used=seed_classes_used,
    )


def compute_entropy_grabcut_from_score(
    score: np.ndarray,
    *,
    entropy_image: np.ndarray | None = None,
    config: EntropyGrabCutConfig | None = None,
) -> EntropyGrabCutCoreResult:
    """Run notebook GrabCut seeding and refinement on a normalized tissue score."""
    config = config or EntropyGrabCutConfig()
    # Validated before the try block below: the fallback handler catches ValueError,
    # so a configuration error raised inside it would be silently reported as a
    # Triangle fallback instead of surfacing to the caller. This validation must run
    # regardless of whether entropy_image is supplied - entropy_tiles otherwise only
    # gets checked on the path that computes entropy itself (local_entropy_image), so
    # a precomputed entropy_image would silently accept an invalid tile count.
    if config.grabcut_convergence_fraction is not None and config.grabcut_max_iterations < 1:
        raise ValueError("grabcut_max_iterations must be at least 1")
    if config.seed_threshold != "triangle":
        _parse_multiotsu_classes(config.seed_threshold)
    if config.gate_classes < 2:
        raise ValueError("gate_classes must be at least 2")
    if config.entropy_tiles is not None and config.entropy_tiles < 1:
        raise ValueError("entropy_tiles must be at least 1")
    score = _validate_score(score)
    entropy_started = perf_counter()
    entropy_image = (
        local_entropy_image(score, config.entropy_radius, tiles=config.entropy_tiles)
        if entropy_image is None
        else np.asarray(entropy_image, dtype=np.float32)
    )
    if entropy_image.shape != score.shape:
        raise ValueError("entropy_image must have the same shape as score")
    entropy_s = perf_counter() - entropy_started

    if not score.any():
        return EntropyGrabCutCoreResult(
            raw_mask=np.zeros_like(score, dtype=bool),
            seed_threshold=float("nan"),
            fallback="",
            timings_s={"entropy": entropy_s, "seed_features": 0.0, "grabcut": 0.0},
        )

    # Seeding and GrabCut are caught separately so the fallback reason names the stage
    # that actually raised. threshold_li, the gate's threshold_multiotsu, threshold_otsu,
    # and np.percentile can all raise before cv2.grabCut is ever called, and reporting
    # those as "GrabCut failed" would blame a stage that never ran.
    seed_started = perf_counter()
    try:
        seed, seed_classes_used = _seed_threshold(score, config.seed_threshold)
        seed_percentile = _percentile_rank(score, seed)
        # Reject a seed that has fallen into a background spike. The failure mode is a seed
        # threshold sitting *below* the rank the definite-background seed claims outright,
        # so the same pixels are proposed as probable foreground and asserted as definite
        # background at once. Stepping the class count down raises the threshold back out.
        #
        # Measured on the 21-case corpus: the two runaway configurations (B0039 sections
        # 220 and 260 at five classes, which take foreground from ~0.15 to ~0.85) sit at
        # percentile 15.21 and 15.23, while the lowest *good* four-class seed anywhere in
        # the corpus is 35.46 - a 20-point margin with no false positives. The guard never
        # fires at the default four classes, so it is insurance against unseen images
        # rather than a change to current behaviour.
        #
        # Deliberately not keyed on the size of the jump in foreground fraction: B0039 220's
        # runaway is +0.70 and Ctl1 DAPI's *correct* four-class behaviour is +0.50, so no
        # threshold on jump size can separate them.
        seed_guard_rejected: list[int] = []
        while (
            config.seed_background_guard
            and seed_classes_used is not None
            and seed_classes_used > 2
            and seed_percentile < _BACKGROUND_PERCENTILE
        ):
            seed_guard_rejected.append(seed_classes_used)
            seed, seed_classes_used = _seed_threshold(
                score, f"multiotsu{seed_classes_used - 1}"
            )
            seed_percentile = _percentile_rank(score, seed)
        li_threshold = float(threshold_li(score))
        gate_high, gate_classes_used = _gate_high_threshold(score, config.gate_classes)
        gate_li_dominated = li_threshold >= gate_high
        high = max(li_threshold, gate_high)
        entropy_cut = float(threshold_otsu(entropy_image))
        sure_fg = binary_erosion(
            (score >= high) & (entropy_image >= entropy_cut), iterations=2
        )
        sure_bg = (
            border_band(score.shape)
            & (score <= np.percentile(score, _BACKGROUND_PERCENTILE))
            & (entropy_image <= entropy_cut)
        )
    except (ValueError, RuntimeError, cv2.error) as exc:
        return _threshold_fallback_result(
            score,
            f"seeding failed ({type(exc).__name__}); used {config.seed_threshold}",
            kind=config.seed_threshold,
            entropy_s=entropy_s,
        )

    if not sure_fg.any() or not sure_bg.any():
        return _threshold_fallback_result(
            score,
            f"insufficient definite seeds; used {config.seed_threshold}",
            kind=config.seed_threshold,
            entropy_s=entropy_s,
            seed_features_s=perf_counter() - seed_started,
        )

    try:
        labels, features = _grabcut_labels_and_features(
            score, entropy_image, seed, sure_fg, sure_bg
        )
        seed_features_s = perf_counter() - seed_started
        grabcut_started = perf_counter()
        raw, iterations_run, converged = _run_grabcut(features, labels, config)
        grabcut_s = perf_counter() - grabcut_started
    except (ValueError, RuntimeError, cv2.error) as exc:
        return _threshold_fallback_result(
            score,
            f"GrabCut failed ({type(exc).__name__}); used {config.seed_threshold}",
            kind=config.seed_threshold,
            entropy_s=entropy_s,
        )

    return EntropyGrabCutCoreResult(
        raw_mask=raw,
        seed_threshold=seed,
        fallback="",
        timings_s={
            "entropy": entropy_s,
            "seed_features": seed_features_s,
            "grabcut": grabcut_s,
        },
        grabcut_iterations_run=iterations_run,
        converged=converged,
        seed_classes_used=seed_classes_used,
        seed_percentile=seed_percentile,
        gate_classes_used=gate_classes_used,
        gate_li_dominated=gate_li_dominated,
        seed_guard_rejected=tuple(seed_guard_rejected),
    )


def compute_entropy_grabcut(
    image: np.ndarray,
    *,
    polarity: str = "bright",
    config: EntropyGrabCutConfig | None = None,
) -> EntropyGrabCutResult:
    """Compute a full-resolution cleaned mask from a raw grayscale or RGB image."""
    config = config or EntropyGrabCutConfig()
    total_started = perf_counter()
    preprocess_started = perf_counter()
    _validate_polarity(polarity)
    image = _validate_raw_image(image)
    try:
        analysis_scale = float(config.analysis_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis_scale must be in (0, 1]") from exc
    if not 0.0 < analysis_scale <= 1.0:
        raise ValueError("analysis_scale must be in (0, 1]")
    score = tissue_score(
        image,
        polarity,
        low=config.low_percentile,
        high=config.high_percentile,
    )
    if analysis_scale < 1.0:
        height, width = score.shape
        analysis_score = cv2.resize(
            score,
            (
                max(1, int(round(width * analysis_scale))),
                max(1, int(round(height * analysis_scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis_score = score
    preprocess_s = perf_counter() - preprocess_started

    core = compute_entropy_grabcut_from_score(analysis_score, config=config)
    raw_mask = core.raw_mask
    if analysis_scale < 1.0:
        raw_mask = cv2.resize(
            raw_mask.astype(np.uint8),
            (score.shape[1], score.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    cleanup_started = perf_counter()
    mask = clean_mask(raw_mask, config=config)
    cleanup_s = perf_counter() - cleanup_started
    timings_s = {
        "preprocess": preprocess_s,
        "entropy": core.timings_s["entropy"],
        "seed_features": core.timings_s["seed_features"],
        "grabcut": core.timings_s["grabcut"],
        "cleanup": cleanup_s,
        "total": perf_counter() - total_started,
    }
    return EntropyGrabCutResult(
        mask=mask,
        raw_mask=raw_mask,
        seed_threshold=core.seed_threshold,
        fallback=core.fallback,
        timings_s=timings_s,
        grabcut_iterations_run=core.grabcut_iterations_run,
        converged=core.converged,
        seed_classes_used=core.seed_classes_used,
        seed_percentile=core.seed_percentile,
        gate_classes_used=core.gate_classes_used,
        gate_li_dominated=core.gate_li_dominated,
        seed_guard_rejected=core.seed_guard_rejected,
    )


def _initialize_worker(opencv_threads: int) -> None:
    """Configure OpenCV once in a spawned batch worker."""
    cv2.setNumThreads(opencv_threads)


def _compute_request(
    request: EntropyGrabCutRequest, config: EntropyGrabCutConfig
) -> EntropyGrabCutResult:
    """Run one picklable batch request in a worker process."""
    return compute_entropy_grabcut(
        request.image,
        polarity=request.polarity,
        config=config,
    )


def compute_entropy_grabcut_batch(
    requests: Sequence[EntropyGrabCutRequest],
    *,
    config: EntropyGrabCutConfig | None = None,
    max_workers: int | None = None,
    opencv_threads_per_worker: int | None = None,
) -> list[EntropyGrabCutResult]:
    """Compute requests in stable input order using Windows-spawn-safe workers."""
    requests = list(requests)
    if not requests:
        return []
    config = config or EntropyGrabCutConfig()
    workers, threads = resolve_worker_budget(
        len(requests),
        max_workers=max_workers,
        opencv_threads_per_worker=opencv_threads_per_worker,
    )
    results: list[EntropyGrabCutResult] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(threads,),
    ) as executor:
        result_iterator = executor.map(_compute_request, requests, repeat(config))
        for index, request in enumerate(requests):
            try:
                results.append(next(result_iterator))
            except Exception as exc:
                raise RuntimeError(
                    f"Entropy GrabCut batch request {index} ({request.label}) failed"
                ) from exc
    return results
