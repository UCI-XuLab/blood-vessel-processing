"""Prove `velazquez_rivera_2025` computes exactly what the notebooks used to compute.

For every helper, every distinct implementation found in the notebooks at the
pre-refactor commit is executed side by side with the shared version on the same
inputs, and the outputs must match bit for bit.

The notebooks themselves cannot be run here — they point at `/media/data/u01/...`
on the lab workstation — so equivalence is established on synthetic inputs
instead: random arrays, degenerate images, and the dtypes the pipeline actually
handles.
"""

import ast
import io as _io

import numpy as np
import pytest

import velazquez_rivera_2025 as archive
from tests import baseline

SEED = 20260729


def rng():
    return np.random.default_rng(SEED)


def assert_same(actual, expected, context):
    """Outputs must be identical, not merely close."""
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), context
        assert len(actual) == len(expected), context
        for a, e in zip(actual, expected):
            assert_same(a, e, context)
        return
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), context
        assert actual.dtype == expected.dtype, f"{context}: dtype {actual.dtype} != {expected.dtype}"
        assert actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}"
        assert np.array_equal(actual, expected, equal_nan=True), f"{context}: values differ"
        return
    assert actual == expected or (
        isinstance(expected, float) and np.isnan(expected) and np.isnan(actual)
    ), f"{context}: {actual!r} != {expected!r}"


def variants_of(name):
    """pytest params for each distinct baseline implementation of `name`."""
    found = baseline.variants(name)
    assert found, f"no baseline implementation of {name} found in git"
    return [pytest.param(v, id=f"{name}-{v[0]}") for v in found]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

METRIC_NAMES = ["dice_coefficient", "iou", "precision", "recall", "rand_index"]


def mask_pairs():
    r = rng()
    big_a = r.random((64, 64)) > 0.5
    big_b = r.random((64, 64)) > 0.5
    return [
        ("random", big_a, big_b),
        ("identical", big_a, big_a),
        ("disjoint", np.eye(32, dtype=bool), ~np.eye(32, dtype=bool)),
        ("both empty", np.zeros((16, 16), bool), np.zeros((16, 16), bool)),
        ("one empty", big_a[:16, :16], np.zeros((16, 16), bool)),
        ("both full", np.ones((16, 16), bool), np.ones((16, 16), bool)),
        ("float masks", big_a[:16, :16].astype(np.float64),
         big_b[:16, :16].astype(np.float64)),
    ]


@pytest.mark.parametrize("name", METRIC_NAMES)
def test_metrics_match_baseline(name):
    shared = getattr(archive, name)
    for digest, original, _source, path in baseline.variants(name):
        for label, a, b in mask_pairs():
            assert_same(shared(a, b), original(a, b),
                        f"{name} [{digest} from {path}] on {label}")


def test_iou_is_dtype_sensitive_and_stays_that_way():
    """`union` uses `+`: logical OR on bool, arithmetic sum on numeric dtypes.

    The pipeline passes boolean masks, so published values are a true IoU. The
    numeric path double-counts the intersection. Both behaviours are preserved
    deliberately — binarising inside `iou` would change reported numbers.
    """
    full = np.ones((8, 8), dtype=bool)
    assert archive.iou(full, full) == pytest.approx(1.0, abs=1e-9)
    # Same masks as float: intersection 64, "union" 128.
    assert archive.iou(full.astype(float), full.astype(float)) == \
        pytest.approx(0.5, abs=1e-9)

    # The dead `union == 0` branch still never fires; epsilon carries the case.
    empty = np.zeros((8, 8), dtype=bool)
    assert archive.iou(empty, empty) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# enhance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", variants_of("gamma_correction"))
def test_gamma_correction_matches_baseline(variant):
    digest, original, source, path = variant
    context = f"gamma_correction [{digest} from {path}]"
    r = rng()
    images = [
        ("uint16", (r.random((32, 32)) * 4000).astype(np.uint16)),
        ("float32", (r.random((32, 32)) * 1000).astype(np.float32)),
        ("constant", np.full((8, 8), 7.0)),
    ]
    for label, image in images:
        for gamma in (1.0, 2.0, 0.5):
            assert_same(
                archive.gamma_correction(image.copy(), gamma=gamma),
                original(image.copy(), gamma=gamma),
                f"{context} on {label} gamma={gamma}",
            )

    # The clamping parameters exist on only one variant; where they exist they
    # must agree, and where they do not the shared default must reduce to it.
    if "min_value" in ast.unparse(ast.parse(source).body[0].args):
        image = (rng().random((32, 32)) * 1000).astype(np.float32)
        for min_value, max_value in [(None, 2000.0), (100.0, None), (100.0, 800.0)]:
            assert_same(
                archive.gamma_correction(image.copy(), gamma=2.0,
                                              min_value=min_value, max_value=max_value),
                original(image.copy(), gamma=2.0,
                         min_value=min_value, max_value=max_value),
                f"{context} clamped min={min_value} max={max_value}",
            )


def test_gamma_correction_superset_reduces_to_short_variant():
    """With both bounds unset the merged signature must match the short form."""
    image = (rng().random((32, 32)) * 1000).astype(np.float32)
    short = [v for v in baseline.variants("gamma_correction")
             if "min_value" not in v[2]]
    assert short, "expected a baseline variant without min_value/max_value"
    for digest, original, _source, path in short:
        assert_same(
            archive.gamma_correction(image.copy(), gamma=2.0),
            original(image.copy(), gamma=2.0),
            f"gamma_correction short form [{digest} from {path}]",
        )


@pytest.mark.parametrize("variant", variants_of("auto_contrast"))
def test_auto_contrast_matches_baseline(variant):
    digest, original, _source, path = variant
    context = f"auto_contrast [{digest} from {path}]"
    r = rng()
    images = [
        ("uint8", (r.random((32, 32)) * 255).astype(np.uint8)),
        ("uint16", (r.random((32, 32)) * 60000).astype(np.uint16)),
    ]
    for label, image in images:
        # alpha=None exercises the dtype-derived scaling path.
        for alpha in (None, 0.5, 1.5, 0.02):
            assert_same(
                archive.auto_contrast(image.copy(), alpha=alpha),
                original(image.copy(), alpha=alpha),
                f"{context} on {label} alpha={alpha}",
            )


@pytest.mark.parametrize("variant", variants_of("histogram_equalization"))
def test_histogram_equalization_matches_baseline(variant):
    digest, original, _source, path = variant
    image = (rng().random((32, 32)) * 255).astype(np.uint8)
    assert_same(
        archive.histogram_equalization(image.copy()),
        original(image.copy()),
        f"histogram_equalization [{digest} from {path}]",
    )


@pytest.mark.parametrize("variant", variants_of("compute_average_image"))
def test_compute_average_image_matches_baseline(variant):
    digest, original, _source, path = variant
    r = rng()
    stack = [(r.random((16, 16)) * 500).astype(np.uint16) for _ in range(5)]
    assert_same(
        archive.compute_average_image(list(stack)),
        original(list(stack)),
        f"compute_average_image [{digest} from {path}]",
    )


@pytest.mark.parametrize("variant", variants_of("n4_bias_correction"))
def test_n4_bias_correction_matches_baseline(variant):
    digest, original, _source, path = variant
    context = f"n4_bias_correction [{digest} from {path}]"
    r = rng()
    image = (r.random((32, 32)) * 500 + 100).astype(np.float32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:28, 4:28] = True
    # shrink_factor 2 and 15 are the values the notebooks actually pass.
    for shrink_factor in (2, 15):
        assert_same(
            archive.n4_bias_correction(image.copy(), mask.copy(),
                                            shrink_factor=shrink_factor),
            original(image.copy(), mask.copy(), shrink_factor=shrink_factor),
            f"{context} shrink_factor={shrink_factor}",
        )


def test_n4_bias_correction_no_longer_raises_on_unshrunk_input():
    """The fixed branch: shrink_factor <= 1 used to leave maskImage unbound."""
    image = (rng().random((32, 32)) * 500 + 100).astype(np.float32)
    mask = np.ones((32, 32), dtype=bool)

    _digest, original, _source, _path = baseline.variants("n4_bias_correction")[0]
    with pytest.raises(NameError):
        original(image.copy(), mask.copy(), shrink_factor=1)

    result = archive.n4_bias_correction(image.copy(), mask.copy(), shrink_factor=1)
    assert result.shape == image.shape
    assert np.isfinite(result).all()


# --------------------------------------------------------------------------
# vessels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", variants_of("detect_vessels"))
def test_detect_vessels_matches_baseline(variant):
    """Each variant must match the shared function called with its own alpha/beta/gamma."""
    digest, original, source, path = variant
    namespace, _ = baseline.notebook_namespace(path)
    params = baseline.objectness_parameters(source, namespace)
    assert set(params) == {"alpha", "beta", "gamma"}, f"{digest}: {params}"

    r = rng()
    image = (r.random((48, 48)) * 255).astype(np.float32)
    for sigma_max, steps in [(10.0, 10), (4.0, 3)]:
        assert_same(
            archive.detect_vessels(image.copy(), 1.0, sigma_max, steps, **params),
            original(image.copy(), 1.0, sigma_max, steps),
            f"detect_vessels [{digest} from {path}] {params} sigma_max={sigma_max}",
        )


def test_detect_vessels_baseline_parameters_are_only_two_configurations():
    """Guards the claim the refactor rests on: alpha/gamma never vary, beta takes two values."""
    seen = set()
    for _digest, _fn, source, path in baseline.variants("detect_vessels"):
        namespace, _ = baseline.notebook_namespace(path)
        p = baseline.objectness_parameters(source, namespace)
        seen.add((p["alpha"], p["beta"], p["gamma"]))
    assert {a for a, _b, _g in seen} == {0.5}
    assert {g for _a, _b, g in seen} == {5.0}
    assert {b for _a, b, _g in seen} == {0.5, 1.0}


@pytest.mark.parametrize("variant", variants_of("process_vessels"))
def test_process_vessels_matches_baseline(variant):
    digest, original, _source, path = variant
    context = f"process_vessels [{digest} from {path}]"
    r = rng()
    vesselness = (r.random((64, 64)) * 255).astype(np.float32)
    cases = [
        ("random", vesselness, dict(thresh=230, min_size=10, area_threshold=2000, smoothing=1)),
        ("smoothing 3", vesselness, dict(thresh=230, min_size=100, area_threshold=2000, smoothing=3)),
        ("all above", np.full((32, 32), 255.0, np.float32), dict(thresh=230)),
        ("all below", np.zeros((32, 32), np.float32), dict(thresh=230)),
    ]
    for label, image, kwargs in cases:
        assert_same(
            archive.process_vessels(image.copy(), **kwargs),
            original(image.copy(), **kwargs),
            f"{context} on {label}",
        )


def test_process_vessels_inverts_because_objects_are_dark():
    """Polarity contract between detect_vessels and process_vessels."""
    below = np.zeros((32, 32), np.float32)
    assert archive.process_vessels(below, thresh=230, min_size=1,
                                        area_threshold=1, smoothing=1).all()


@pytest.mark.parametrize("variant", variants_of("get_brain_mask"))
def test_get_brain_mask_matches_baseline(variant):
    digest, original, _source, path = variant
    context = f"get_brain_mask [{digest} from {path}]"
    r = rng()
    image = np.zeros((128, 128), dtype=np.uint8)
    image[20:100, 20:100] = (r.random((80, 80)) * 120 + 130).astype(np.uint8)
    image[50:60, 50:60] = 5  # an interior hole
    for area_threshold, min_size in [(300000, 10000), (50000, 10000), (25000, 100)]:
        assert_same(
            archive.get_brain_mask(image.copy(), area_threshold=area_threshold,
                                        min_size=min_size),
            original(image.copy(), area_threshold=area_threshold, min_size=min_size),
            f"{context} area={area_threshold} min_size={min_size}",
        )


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def _write_stack(directory, count, shape=(12, 10)):
    import SimpleITK as sitk
    r = rng()
    for i in range(count):
        array = (r.random(shape) * 1000).astype(np.uint16)
        sitk.WriteImage(sitk.GetImageFromArray(array),
                        str(directory / f"slice_{i:03d}.tif"))
    return str(directory / "*.tif")


@pytest.mark.parametrize("variant", variants_of("read_tif"))
def test_read_tif_matches_baseline(variant, tmp_path):
    digest, original, _source, path = variant
    pattern = _write_stack(tmp_path, 1)
    target = pattern.replace("*", "slice_000")
    assert_same(archive.read_tif(target), original(target),
                f"read_tif [{digest} from {path}]")


@pytest.mark.parametrize("variant", variants_of("load_channel"))
def test_load_channel_matches_baseline(variant, tmp_path, capsys):
    digest, original, _source, path = variant
    pattern = _write_stack(tmp_path, 4)
    for idx in range(4):
        assert_same(archive.load_channel(pattern, idx), original(pattern, idx),
                    f"load_channel [{digest} from {path}] idx={idx}")
    capsys.readouterr()


@pytest.mark.parametrize("variant", variants_of("load_channels"))
def test_load_channels_matches_baseline(variant, tmp_path, capsys):
    digest, original, _source, path = variant
    pattern = _write_stack(tmp_path, 6)
    for idx in range(3):
        assert_same(archive.load_channels(pattern, idx), original(pattern, idx),
                    f"load_channels [{digest} from {path}] idx={idx}")
    capsys.readouterr()


@pytest.mark.parametrize("variant", variants_of("load_3_channels"))
def test_load_3_channels_matches_baseline(variant, tmp_path, capsys):
    digest, original, _source, path = variant
    pattern = _write_stack(tmp_path, 9)
    for idx in range(3):
        assert_same(archive.load_3_channels(pattern, idx), original(pattern, idx),
                    f"load_3_channels [{digest} from {path}] idx={idx}")
    capsys.readouterr()


# --------------------------------------------------------------------------
# viz  (compared by rendering, not by return value)
# --------------------------------------------------------------------------

def render(function, *args, **kwargs):
    """Call a plotting helper and capture what it drew, as pixels."""
    import matplotlib.pyplot as plt

    captured = {}
    real_show = plt.show

    def capture():
        buffer = _io.BytesIO()
        plt.gcf().savefig(buffer, format="png", dpi=50)
        captured["png"] = buffer.getvalue()

    plt.show = capture
    try:
        function(*args, **kwargs)
    finally:
        plt.show = real_show
        plt.close("all")
    assert "png" in captured, "helper never called plt.show()"
    return captured["png"]


def _viz_images():
    r = rng()
    image = (r.random((24, 24)) * 255).astype(np.uint8)
    contour = np.zeros((24, 24), dtype=bool)
    contour[6:18, 6:18] = True
    return image, contour


@pytest.mark.parametrize("variant", variants_of("show"))
def test_show_matches_baseline(variant):
    digest, original, _source, path = variant
    image, contour = _viz_images()
    context = f"show [{digest} from {path}]"
    cases = [
        dict(title="one"),
        dict(contour=contour, title="contoured", axis=False),
        dict(contour=contour, image2=image, contour2=contour,
             title="a", title2="b", xlim=(2, 20), ylim=(20, 2)),
    ]
    for kwargs in cases:
        assert render(archive.show, image, **kwargs) == \
               render(original, image, **kwargs), f"{context} {sorted(kwargs)}"


@pytest.mark.parametrize("variant", variants_of("show3"))
def test_show3_matches_baseline(variant):
    digest, original, _source, path = variant
    image, contour = _viz_images()
    kwargs = dict(contour=contour, image2=image, contour2=contour, image3=image,
                  title="a", title2="b", title3="c", axis=False)
    assert render(archive.show3, image, **kwargs) == \
           render(original, image, **kwargs), f"show3 [{digest} from {path}]"


@pytest.mark.parametrize("variant", variants_of("show_4"))
def test_show_4_matches_baseline(variant):
    digest, original, _source, path = variant
    image, contour = _viz_images()
    for kwargs in [dict(), dict(xlim=(2, 20), ylim=(2, 20))]:
        assert render(archive.show_4, image, image, contour, **kwargs) == \
               render(original, image, image, contour, **kwargs), \
               f"show_4 [{digest} from {path}] {sorted(kwargs)}"


@pytest.mark.parametrize("variant", variants_of("save_figure"))
def test_save_figure_matches_baseline(variant):
    """Compared structurally: it renders at dpi=600 on a 20x20 inch canvas."""
    digest, original, source, path = variant
    import inspect

    def normalise(text):
        tree = ast.parse(text)
        function = tree.body[0]
        if (function.body and isinstance(function.body[0], ast.Expr)
                and isinstance(function.body[0].value, ast.Constant)
                and isinstance(function.body[0].value.value, str)):
            function.body.pop(0)  # drop docstring
        return ast.dump(ast.parse(ast.unparse(function)))

    assert normalise(inspect.getsource(archive.save_figure)) == normalise(source), \
        f"save_figure [{digest} from {path}] body differs"


# --------------------------------------------------------------------------
# coverage of the extraction itself
# --------------------------------------------------------------------------

def test_every_baseline_helper_is_accounted_for():
    """No inline helper was silently left behind or silently dropped."""
    shared = set(archive.__all__)
    baseline_names = set(baseline.defined_function_names())
    unaccounted = baseline_names - shared - baseline.EXCLUDED
    assert not unaccounted, f"helpers neither shared nor explicitly excluded: {sorted(unaccounted)}"

    invented = shared - baseline_names
    assert not invented, f"velazquez_rivera_2025 exports names no notebook ever defined: {sorted(invented)}"


def test_dropped_helpers_are_really_gone():
    assert not hasattr(archive, "preprocess_image")
    assert "preprocess_image" not in archive.__all__


def test_every_exported_name_resolves():
    for name in archive.__all__:
        assert callable(getattr(archive, name)), name
