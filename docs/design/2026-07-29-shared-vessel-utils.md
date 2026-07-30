# Extracting shared helpers into `vessel_utils`

**Date:** 2026-07-29
**Status:** Approved

## Problem

27 notebooks each carry an inline copy of the same helper functions — `read_tif`,
`auto_contrast`, `gamma_correction`, `detect_vessels`, `process_vessels`,
`get_brain_mask`, five segmentation metrics, and the display helpers. A fix to any
one of them has to be applied 27 times, and today the copies have already drifted.

## Evidence

Every top-level `def` in every notebook was parsed and its AST hashed
(docstrings included, then compared again with them stripped). The drift is far
smaller than the file count suggests:

| function | notebooks defining it | distinct variants | nature of the difference |
| --- | --- | --- | --- |
| `read_tif` | 27 | 1 | identical |
| `auto_contrast`, `get_brain_mask` | 26 | 1 | identical |
| `dice_coefficient`, `iou`, `precision`, `recall`, `rand_index` | 25 | 1 | identical |
| `load_channels`, `show` | 21 | 1 | identical |
| `n4_bias_correction`, `preprocess_image` | 9 | 1 | identical |
| `compute_average_image` | 6 | 1 | identical |
| `show_4` | 5 | 1 | identical |
| `load_channel` | 4 | 1 | identical |
| `save_figure` | 26 | 2 | docstring only |
| `process_vessels` | 26 | 2 | docstring only |
| `gamma_correction` | 26 | 2 | one is a strict superset of the other |
| `detect_vessels` | 26 | 5 | docstrings, plus how α/β/γ are supplied |

`detect_vessels` holds the only behavioural difference in the entire set. Where
`ALPHA`/`BETA`/`GAMMA` constants exist they are `0.5`/`0.5`/`5.0` in all 20
notebooks. The notebooks that hardcode instead use `beta=1.0` (`M7`, `M12`,
`process_blood_vessel_brain`) or `beta=1` (`M74`); `M14` uses `beta=0.5`;
`3-M13 run 2` already exposes α/β/γ as keyword arguments. Two effective
configurations, both expressible at the call site.

A behaviour-preserving extraction is therefore achievable exactly, not
approximately.

## Design

### Layout

```
vessel_utils/
  __init__.py     re-exports the common names
  io.py           read_tif, load_channel, load_channels, load_3_channels
  enhance.py      auto_contrast, gamma_correction, histogram_equalization,
                  n4_bias_correction, compute_average_image
  vessels.py      detect_vessels, process_vessels, get_brain_mask
  metrics.py      dice_coefficient, iou, precision, recall, rand_index
  viz.py          show, show3, show_4, save_figure
pyproject.toml
tests/
  test_equivalence.py
  test_notebooks.py
```

Modules are split by pipeline stage, so a notebook's import block reads as a
description of what that notebook does.

### How notebooks import

`pyproject.toml` supports `pip install -e .`. Each notebook also carries a
`sys.path` bootstrap so it still runs on a machine where nobody performed the
install — the lab workstation runs these notebooks directly and must not acquire
a setup step it can silently skip.

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

from vessel_utils.io import read_tif, load_channels
from vessel_utils.vessels import detect_vessels, process_vessels, get_brain_mask
```

The bootstrap is a no-op when the package is installed.

### Signature reconciliation

`detect_vessels(input_image, min_sigma=1.0, max_sigma=10.0, num_steps=10, alpha=0.5, beta=0.5, gamma=5.0)`
— α/β/γ become keyword arguments, matching what `3-M13 run 2` already does.
Notebooks holding module-level constants keep them and pass
`alpha=ALPHA, beta=BETA, gamma=GAMMA`. Notebooks that hardcoded pass their
literal. Every notebook keeps its current numbers.

`gamma_correction(image, gamma=2.0, min_value=None, max_value=None)` — the
superset signature. With both bounds `None` it reduces arithmetically to the
shorter variant: `max_value` becomes `image.max()`, which is what the short
variant divides and re-multiplies by.

`save_figure` and `process_vessels` differ only in docstrings; the documented
variant is kept.

### What stays in the notebooks

Batch loops, all parameters, per-slice overrides (`if i == 0: THRESH = 3200`),
and `ALPHA`/`BETA`/`GAMMA` constants. These are per-specimen tuning and belong
where a reviewer reading the figure's notebook will see them.

`run_test` in `3-M13 run 2` stays local: it closes over the notebook's
`filepath` variable and embeds slice-specific thresholds.

### Deliberate behaviour changes

Two, both flagged and approved:

1. **`preprocess_image` is dropped.** Defined in 9 notebooks, called in none, and
   would raise `TypeError` if called — it passes `alpha=` to
   `n4_bias_correction`, which no longer accepts that parameter. Dead code.

2. **`n4_bias_correction`'s unbound-mask bug is fixed.** `maskImage` is currently
   assigned only inside `if shrink_factor > 1`, so any call with
   `shrink_factor <= 1` raises `NameError` at `bias_corrector.Execute`. Every
   existing call site uses 2 or 15, so the fix cannot alter a published result.
   The fix binds `maskImage = mask_img` before the branch.

### Deliberately preserved

`iou` computes `union = np.sum(binary_image1 + binary_image2)`. On boolean
arrays `+` is logical OR, so for the boolean masks the pipeline actually
produces this is a true IoU and the published numbers are sound; on int or float
masks the same line double-counts the intersection. Its `if union == 0` guard is
dead either way, immediately overwritten by the line below. This is the metric
the paper reports. It stays byte-equivalent, with the dtype sensitivity recorded
in a comment and pinned by a test, so the next reader does not "fix" it and
silently invalidate the published numbers.

## Verification

The notebooks cannot execute here: paths are hardcoded to `/media/data/u01/...`
on the lab's Linux workstation and the imaging data is not in the repository.
Equivalence is therefore established structurally.

**`tests/test_equivalence.py`** — for each helper, retrieve *every* original
variant from git at the pre-refactor commit, `exec` it in an isolated namespace,
and assert output identical to `vessel_utils` on synthetic inputs: random arrays,
constant images, all-zero and all-one masks, and dtype boundaries (`uint8`,
`uint16`, `float32`). Functions wrapping ITK/SimpleITK are exercised on small
real arrays. This is the check that the refactor changed nothing.

**`tests/test_notebooks.py`** — parses all 27 notebooks and asserts that none
still defines a now-shared helper, that every notebook using a shared name
imports it, and that every notebook remains valid JSON with parseable code cells.

The two intentional changes are asserted explicitly rather than by equivalence:
`preprocess_image` is absent from the package and from every notebook, and
`n4_bias_correction` succeeds with `shrink_factor=1` where the original raised
`NameError`.

## Execution

Notebook edits are applied by a script operating on the `.ipynb` JSON. Within the
`## Functions` cells it removes only `FunctionDef` nodes, preserving the
`import itk` and `from skimage.morphology import ...` statements and the constants
that share those cells, and leaves stored outputs untouched.

## Documentation

`CLAUDE.md` currently instructs against extracting these helpers — guidance
inferred from the README before this decision. Both that section and the README
sentence it derives from are rewritten to describe the shared package.
