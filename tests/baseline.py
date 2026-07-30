"""Recover the pre-refactor helper functions straight out of git history.

Every helper now living in `velazquez_rivera_2025` used to be defined inline in each
notebook. This module reads those notebooks as they were at
`BASELINE_REV` — the last commit before the extraction — and rebuilds one
namespace per notebook containing exactly the functions that notebook defined.

That gives the equivalence tests a ground truth that cannot drift: it comes from
git, not from a copy someone remembered to update.
"""

import ast
import functools
import hashlib
import json
import subprocess

# Last commit before the velazquez_rivera_2025 extraction ("Update README.md").
BASELINE_REV = "658304dddb64c35688277be05a7ac34517d3d156"

# Imports the notebooks make before defining their helpers. Recreated here so the
# recovered functions resolve the same globals they originally did.
PRELUDE = """
import csv
import glob
import os
import time
from pprint import pprint

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
import itk
from scipy import ndimage
from scipy.ndimage import median_filter
from scipy.spatial.distance import hamming
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import (binary_closing, disk, medial_axis,
                                remove_small_holes, remove_small_objects,
                                skeletonize)
from skimage.measure import label, regionprops
"""

# Defined inline in the notebooks but deliberately not carried into velazquez_rivera_2025.
EXCLUDED = {
    "preprocess_image",  # dead code in all 9 notebooks; would raise TypeError if called
    "run_test",          # notebook-local: closes over `filepath` and slice-specific thresholds
}


def _git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, check=True, text=True, encoding="utf-8"
    ).stdout


@functools.lru_cache(maxsize=1)
def baseline_notebooks():
    """Paths of every notebook present at the baseline revision."""
    listing = _git("ls-tree", "-r", "--name-only", BASELINE_REV).splitlines()
    return tuple(p for p in listing if p.endswith(".ipynb"))


def _baseline_source(path):
    return _git("show", f"{BASELINE_REV}:{path}")


def _code_cells(notebook_json):
    for cell in json.loads(notebook_json)["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Strip IPython magics, which are not valid Python.
        yield "\n".join(
            line for line in source.split("\n") if not line.strip().startswith("%")
        )


@functools.lru_cache(maxsize=None)
def notebook_namespace(path):
    """Rebuild `path`'s helper functions as they existed at the baseline revision.

    Returns (namespace, {function name: source text}). Only function definitions
    and upper-case constant assignments are executed — never the notebook's
    data-loading or batch cells, which reference paths that do not exist here.
    """
    namespace = {}
    exec(compile(PRELUDE, "<prelude>", "exec"), namespace)
    sources = {}

    for cell in _code_cells(_baseline_source(path)):
        try:
            tree = ast.parse(cell)
        except SyntaxError:
            continue
        lines = cell.split("\n")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                text = "\n".join(lines[node.lineno - 1:node.end_lineno])
                sources[node.name] = text
                exec(compile(text, f"<{path}:{node.name}>", "exec"), namespace)
            elif isinstance(node, ast.Assign):
                # ALPHA / BETA / GAMMA feed detect_vessels in most notebooks.
                targets = [t for t in node.targets
                           if isinstance(t, ast.Name) and t.id.isupper()]
                if targets and isinstance(node.value, ast.Constant):
                    for target in targets:
                        namespace[target.id] = node.value.value

    return namespace, sources


@functools.lru_cache(maxsize=None)
def variants(function_name):
    """Every distinct implementation of `function_name` across the notebooks.

    Returns a tuple of (digest, callable, source, first_notebook). Notebooks
    sharing byte-identical source collapse to a single entry, so a function
    copied unchanged into 27 notebooks is tested once.
    """
    found = {}
    for path in baseline_notebooks():
        namespace, sources = notebook_namespace(path)
        if function_name not in sources:
            continue
        digest = hashlib.md5(sources[function_name].encode()).hexdigest()[:8]
        found.setdefault(
            digest, (digest, namespace[function_name], sources[function_name], path)
        )
    return tuple(found.values())


@functools.lru_cache(maxsize=1)
def defined_function_names():
    """Names of every helper defined inline at the baseline revision."""
    names = set()
    for path in baseline_notebooks():
        _, sources = notebook_namespace(path)
        names |= set(sources)
    return frozenset(names)


def objectness_parameters(source, namespace=None):
    """Read the alpha/beta/gamma a `detect_vessels` variant actually applies.

    Variants differ only in how these reach the ITK objectness filter: some read
    module-level ALPHA/BETA/GAMMA constants, some hardcode literals, one takes
    them as keyword arguments. Recovering the effective values lets the
    equivalence test call the shared function with matching arguments.
    """
    tree = ast.parse(source)
    setters = {"SetAlpha": "alpha", "SetBeta": "beta", "SetGamma": "gamma"}
    resolved = {}

    # Defaults declared on the variant's own signature, if any.
    function = tree.body[0]
    defaults = dict(
        zip(
            [a.arg for a in function.args.args][-len(function.args.defaults):]
            if function.args.defaults else [],
            [d.value if isinstance(d, ast.Constant) else None
             for d in function.args.defaults],
        )
    )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        key = setters.get(node.func.attr)
        if key is None or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant):
            resolved[key] = argument.value
        elif isinstance(argument, ast.Name):
            # Either the variant's own parameter default, or a module-level
            # constant read from the notebook that defined it.
            if argument.id in defaults:
                resolved[key] = defaults[argument.id]
            elif namespace is not None and argument.id in namespace:
                resolved[key] = namespace[argument.id]
            else:
                raise AssertionError(
                    f"cannot resolve {argument.id} for {node.func.attr}"
                )
    return resolved
