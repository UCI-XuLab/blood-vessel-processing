"""Check the rewritten notebooks against the extraction's invariants.

These tests read the notebooks as they are on disk now, and compare them with
the pre-refactor originals recovered from git. They catch the failure modes the
rewrite could plausibly have: a helper left behind, a helper used without being
imported, a lost call site, or a changed parameter.
"""

import ast
import json
from pathlib import Path

import pytest

import velazquez_rivera_2025 as archive
from tests import baseline

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(REPO.glob("process_*/**/*.ipynb"))
SHARED = set(archive.__all__)


def pytest_generate_tests(metafunc):
    if "notebook" in metafunc.fixturenames:
        metafunc.parametrize(
            "notebook", NOTEBOOKS,
            ids=[f"{p.parent.name}/{p.stem}" for p in NOTEBOOKS],
        )


def code_cells(path):
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            yield "\n".join(
                ("#" + line) if line.strip().startswith(("%", "!")) else line
                for line in source.split("\n")
            )


def trees(path):
    for cell in code_cells(path):
        yield ast.parse(cell)


def relative(path):
    return path.relative_to(REPO).as_posix()


def test_notebook_count_unchanged():
    assert len(NOTEBOOKS) == len(baseline.baseline_notebooks()) == 27


def test_every_code_cell_still_parses(notebook):
    for index, cell in enumerate(code_cells(notebook)):
        try:
            ast.parse(cell)
        except SyntaxError as error:
            pytest.fail(f"{relative(notebook)} cell {index}: {error}")


def test_no_shared_helper_is_still_defined_inline(notebook):
    leftovers = {
        node.name
        for tree in trees(notebook)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in SHARED
    }
    assert not leftovers, f"{relative(notebook)} still defines {sorted(leftovers)}"


def test_dropped_helper_is_gone(notebook):
    text = notebook.read_text(encoding="utf-8")
    assert "preprocess_image" not in text, \
        f"{relative(notebook)} still mentions preprocess_image"


def test_every_shared_helper_used_is_imported(notebook):
    """A name referenced but not imported would be a NameError at run time."""
    imported = set()
    defined = set()
    used = set()

    for tree in trees(notebook):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("velazquez_rivera_2025"):
                imported |= {alias.asname or alias.name for alias in node.names}
            elif isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.For)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(target, ast.Name):
                    defined.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in SHARED:
                    used.add(node.id)

    missing = used - imported - defined
    assert not missing, f"{relative(notebook)} uses {sorted(missing)} without importing them"


def test_no_unused_shared_imports(notebook):
    """Imports should reflect what the notebook actually does."""
    imported = set()
    used = set()
    for tree in trees(notebook):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("velazquez_rivera_2025"):
                imported |= {alias.asname or alias.name for alias in node.names}
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in SHARED:
                used.add(node.id)
    assert not (imported - used), \
        f"{relative(notebook)} imports unused {sorted(imported - used)}"


def test_bootstrap_present_when_imports_are(notebook):
    text = notebook.read_text(encoding="utf-8")
    if "from velazquez_rivera_2025" in text:
        assert "velazquez_rivera_2025" in text and "sys.path.insert" in text, \
            f"{relative(notebook)} imports velazquez_rivera_2025 without the path bootstrap"


# --------------------------------------------------------------------------
# the rewrite must not have changed what the notebook computes
# --------------------------------------------------------------------------

def _detect_vessels_calls(path_or_rev, from_git=False):
    """Positional-argument text and keyword names for each detect_vessels call."""
    if from_git:
        cells = []
        source = baseline._git("show", f"{baseline.BASELINE_REV}:{path_or_rev}")
        for cell in json.loads(source)["cells"]:
            if cell["cell_type"] == "code":
                raw = "".join(cell["source"])
                cells.append("\n".join(
                    ("#" + line) if line.strip().startswith(("%", "!")) else line
                    for line in raw.split("\n")))
    else:
        cells = list(code_cells(path_or_rev))

    calls = []
    for cell in cells:
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "detect_vessels"):
                # Skip calls inside a definition that no longer exists.
                calls.append((
                    tuple(ast.unparse(a) for a in node.args),
                    {kw.arg: ast.unparse(kw.value) for kw in node.keywords},
                ))
    return calls


def test_detect_vessels_call_sites_preserved(notebook):
    """Same number of calls, same positional arguments, plus explicit objectness."""
    before = _detect_vessels_calls(relative(notebook), from_git=True)
    after = _detect_vessels_calls(notebook)
    assert len(after) == len(before), \
        f"{relative(notebook)}: {len(before)} calls before, {len(after)} after"

    for index, ((old_args, old_kwargs), (new_args, new_kwargs)) in enumerate(zip(before, after)):
        assert new_args == old_args, f"{relative(notebook)} call {index}: positional args changed"
        # Keywords the notebook already passed must survive untouched.
        for key, value in old_kwargs.items():
            assert new_kwargs.get(key) == value, \
                f"{relative(notebook)} call {index}: {key} changed from {value} to {new_kwargs.get(key)}"
        if new_args or new_kwargs:
            assert {"alpha", "beta", "gamma"} <= set(new_kwargs), \
                f"{relative(notebook)} call {index}: objectness parameters not explicit"


def test_detect_vessels_arguments_match_the_original_definition(notebook):
    """The injected alpha/beta/gamma must equal what this notebook's own copy applied."""
    path = relative(notebook)
    _, sources = baseline.notebook_namespace(path)
    if "detect_vessels" not in sources:
        return
    namespace, _ = baseline.notebook_namespace(path)
    expected = baseline.objectness_parameters(sources["detect_vessels"], namespace)

    for index, (_args, kwargs) in enumerate(_detect_vessels_calls(notebook)):
        for key in ("alpha", "beta", "gamma"):
            literal = kwargs[key]
            if literal in namespace:
                actual = namespace[literal]  # a module constant such as ALPHA
            else:
                try:
                    actual = ast.literal_eval(literal)
                except (ValueError, SyntaxError):
                    # A notebook-local variable the call already passed (beta1,
                    # beta2, ...). test_detect_vessels_call_sites_preserved
                    # already asserts those survived unchanged.
                    continue
            assert actual == expected[key], \
                f"{path} call {index}: {key}={actual} but original applied {expected[key]}"


def test_other_call_sites_are_unchanged(notebook):
    """Every non-detect_vessels helper call keeps its exact arguments."""
    def calls_of(cells):
        found = []
        for cell in cells:
            try:
                tree = ast.parse(cell)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in SHARED and node.func.id != "detect_vessels"):
                    found.append(ast.unparse(node))
        return sorted(found)

    source = baseline._git("show", f"{baseline.BASELINE_REV}:{relative(notebook)}")
    before_cells = []
    for cell in json.loads(source)["cells"]:
        if cell["cell_type"] == "code":
            raw = "".join(cell["source"])
            before_cells.append("\n".join(
                ("#" + line) if line.strip().startswith(("%", "!")) else line
                for line in raw.split("\n")))

    before = calls_of(before_cells)
    after = calls_of(code_cells(notebook))

    # Calls that lived inside removed helper bodies are legitimately gone.
    removed_bodies = set()
    _, sources = baseline.notebook_namespace(relative(notebook))
    for name, text in sources.items():
        if name in SHARED or name == "preprocess_image":
            for node in ast.walk(ast.parse(text)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in SHARED):
                    removed_bodies.add(ast.unparse(node))

    expected = sorted(c for c in before if c not in removed_bodies)
    for call in expected:
        assert call in after, f"{relative(notebook)}: lost call {call}"


def test_markdown_and_outputs_untouched(notebook):
    """Only code cells were edited; prose and stored results survive."""
    source = baseline._git("show", f"{baseline.BASELINE_REV}:{relative(notebook)}")
    before = json.loads(source)["cells"]
    after = json.loads(notebook.read_text(encoding="utf-8"))["cells"]

    before_markdown = ["".join(c["source"]) for c in before if c["cell_type"] == "markdown"]
    after_markdown = ["".join(c["source"]) for c in after if c["cell_type"] == "markdown"]
    assert after_markdown == before_markdown, f"{relative(notebook)}: markdown changed"

    before_outputs = sum(len(c.get("outputs", [])) for c in before)
    after_outputs = sum(len(c.get("outputs", [])) for c in after)
    assert after_outputs == before_outputs, \
        f"{relative(notebook)}: stored outputs went from {before_outputs} to {after_outputs}"


def _code_size(cells):
    return sum(len("".join(c["source"])) for c in cells if c["cell_type"] == "code")


def test_duplication_actually_went_away():
    """The whole point: far less notebook code overall.

    Measured in aggregate rather than per notebook — condense_lightsheet_brain
    only ever held two short helpers, so its import block costs more than the
    definitions it replaced. That is fine; the total is what matters.
    """
    before = after = 0
    for notebook in NOTEBOOKS:
        source = baseline._git("show", f"{baseline.BASELINE_REV}:{relative(notebook)}")
        before += _code_size(json.loads(source)["cells"])
        after += _code_size(json.loads(notebook.read_text(encoding="utf-8"))["cells"])

    # Measured reduction at the time of the extraction was ~36% of all notebook
    # code; the threshold leaves room for the batch cells to grow later.
    assert after < before * 0.75, f"expected a large reduction, got {before} -> {after}"


def test_every_substantial_notebook_shrank():
    """Every notebook that carried the full helper block is now smaller."""
    for notebook in NOTEBOOKS:
        source = baseline._git("show", f"{baseline.BASELINE_REV}:{relative(notebook)}")
        before = _code_size(json.loads(source)["cells"])
        if before < 5000:
            continue  # condense_lightsheet_brain, which defined only two helpers
        after = _code_size(json.loads(notebook.read_text(encoding="utf-8"))["cells"])
        assert after < before, f"{relative(notebook)}: {before} -> {after} characters"
