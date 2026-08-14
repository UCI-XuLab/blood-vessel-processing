# An interactive viewer for tuning the spinal-cord vessel segmentation

**Date:** 2026-08-14
**Status:** Approved

## Problem

The spinal-cord specificity pipeline in [scripts/analyse_spinal_cord.py](../../scripts/analyse_spinal_cord.py)
has five knobs — `SIGMAS`, `VESSEL_HIGH`, `VESSEL_LOW`, `MIN_VESSEL_UM2`, `VIRUS_K` —
plus a sixth, `reference_lambda`, that is coupled to the thresholds. Today the only
way to see what a knob does is to edit a module constant, re-run the batch over 34–63
sections, and read a CSV. The visual consequence of a setting, which is the thing a
biologist can actually judge, is never on screen next to the setting.

[scripts/sweep_spinal_cord.py](../../scripts/sweep_spinal_cord.py) exists because of
this: its docstring records that tuning moved the vessel area fraction across 0.004,
0.085 and 0.35 through choices that are "individually defensible and jointly
under-determined". A sweep answers *whether the ordering survives the range*. It does
not help anyone see **where in that range this section actually is**, which is the
question a viewer answers.

The specific trap the viewer has to make visible is in that same docstring:

> In 2D, Jerman saturates everything above `tau * reference_lambda / 2` to exactly 1,
> so the reference and the threshold are coupled and neither can be tuned alone.

A tuning GUI that hides the vesselness response would let a user tune thresholds
against a saturated field, where the threshold is inert and the picture still looks
plausible. So the response is a first-class layer, not a debug extra.

## Non-goals

- The archive pipeline in [velazquez_rivera_2025/](../../velazquez_rivera_2025/). It is
  frozen and its numbers are published; a tuning GUI for it would invite exactly the
  edits the freeze exists to prevent.
- Whole-brain lightsheet volumes. At ~500 GB per channel a live re-run is not
  physically possible, and the tuning that matters happens on 2D sections.
- Editing or replacing any published CSV.
- Undo / parameter history. napari has no built-in for it, and it is speculative until
  someone finds themselves wanting the last five settings back.

## Framework choice

Requirement: runs locally on Windows machines belonging to several lab members, 2D
sections, layered overlays, pan and zoom, live parameter tuning.

| Option | Verdict |
| --- | --- |
| **napari + magicgui** | **Chosen.** Supplies the entire viewer half: pan/zoom, a layer list with per-layer visibility, opacity, blending and colormap, a `Labels` layer built for masks, and `magicgui`, which derives a parameter panel from a function signature. Bioimage-standard, so the audience may already know it. |
| Panel / Bokeh + datashader | The only credible runner-up, and the better choice had the requirement been "browser, nothing installed". It is not: everyone runs locally, so a server buys nothing, and the layer model, blending and mask overlays all become our code. |
| Streamlit | Re-runs the script on every widget change and has no real pan/zoom. Wrong for a pipeline with a multi-second stage. |
| Jupyter + ipywidgets / stackview | Lightest, and the lab already runs notebooks. Weak pan/zoom, awkward multi-file browsing, and the result is a notebook rather than an application. |
| pyqtgraph + hand-written Qt | Reimplements most of napari. Justified only if napari could not be installed. |
| Fiji / QuPath | Cannot call `vessel_utils` without a pyimagej bridge. |
| itkwidgets | Tempting since `itk` is already a dependency, but volume-oriented with a weaker parameter-widget story. |

Installability was verified rather than assumed. On the project's Python 3.14 `.venv`,
`pip install --dry-run "napari[all]"` resolves to napari 0.8.0 with PyQt6 6.10.2 —
PyQt6 publishes `abi3` wheels, so the new interpreter is not a barrier and no second
virtual environment is needed.

## Architecture

### The cost cascade

The design is a caching problem, not a widget problem. The knobs have sharply tiered
costs, and "live" means respecting the tiers:

| Knob | Invalidates | Cost |
| --- | --- | --- |
| file | everything | disk read + GrabCut (already disk-cached) |
| `sigmas` | reference, response, mask, metrics | seconds to tens of seconds |
| `reference_lambda` | response, mask, metrics | seconds |
| `high`, `low`, `min_vessel_um2` | mask, metrics | ~100 ms |
| `virus_k` | virus-positive mask, coverage, off-target | ~10 ms |

Two `functools.lru_cache`d functions implement it, and that is the whole performance
design:

```
load(path)                         -> green, cd31, tissue
response(path, sigmas, reference)  -> normalised, vesselness
```

`maxsize=4` on each — large arrays, but enough to flip between a few sections without
re-filtering. Everything downstream of `response` re-runs on every widget change,
because at ~100 ms it is free.

So `high`, `low`, `min_vessel_um2` and `virus_k` use magicgui's `auto_call` and update
on release; `sigmas` and `reference_lambda` sit behind an explicit **Recompute**
button, because by construction they miss the cache.

### `reference_lambda` is computed once

`calibrate_reference()` runs once at startup and its result is then an *editable
number*, never silently recomputed per file. Its docstring is explicit that the value
is dataset-wide on purpose: computed per image, "a fixed threshold would mean a
different thing in every section — and the comparison being made here is precisely
across sections, regions and mice." Per-section recalibration inside a GUI would
quietly destroy the comparability the analysis depends on.

Exposing it as editable is deliberate: it lets a user compare the calibrated median
against the fixed `2.5` that `sweep_spinal_cord.py` uses, which is the only honest way
to reason about a knob coupled to the thresholds.

### Components

Three files touched, one created.

**[scripts/analyse_spinal_cord.py](../../scripts/analyse_spinal_cord.py)** — refactor
only. `analyse()` currently computes `tissue`, `vessels` and `virus_positive` and then
discards them, returning scalars. Split it:

```python
def stages(path, reference, *, sigmas=SIGMAS, low=VESSEL_LOW, high=VESSEL_HIGH,
           min_vessel_um2=MIN_VESSEL_UM2, virus_k=VIRUS_K):
    """Every intermediate array, so a viewer can show them as layers."""
    # -> {green, cd31, normalised, response, tissue, vessels, virus_positive, cut}

def summarise(st):
    """The scalars, from the arrays."""   # the existing analyse() return dict

def analyse(path, reference, **kw):
    return summarise(stages(path, reference, **kw))
```

Plain keyword arguments whose defaults *are* the existing module constants — not a
`Params` dataclass, because magicgui derives its widget panel from a function
signature and a dataclass would be a wrapper to unwrap. The constants stay at module
level where a reviewer reading the figure finds them, per CLAUDE.md.

This module is already the shared library for this pipeline:
`sweep_spinal_cord.py` imports `tissue_mask`, `normalise_for_segmentation`,
`curated_paths`, `SIGMAS`, `UM_PER_PX` and `NAME` from it, and its `prepare()` is
already documented as "per-section work that does not depend on the threshold" — the
same split, half-built. The GUI becomes a third consumer, not a new architecture.

**`scripts/tune_spinal_cord.py`** (new, ~180 lines) — the viewer. Holds no pipeline
logic; it wires `stages`/`summarise` to napari layers and magicgui widgets.

**[pyproject.toml](../../pyproject.toml)** — a new `gui` extra:

```toml
gui = ["napari[all]"]
```

Kept out of `dev` so `pip install -e ".[dev]"` continues not to pull Qt and ~45
packages into a test environment.

### Layers

Bottom to top, using napari's built-in per-layer visibility, opacity and blending:

| Layer | Type | Default |
| --- | --- | --- |
| CD31 | Image, magenta, additive | visible |
| virus reporter | Image, green, additive | visible |
| vesselness response | Image, `turbo`, clim pinned 0–1 | hidden, one click away |
| tissue | Labels | hidden |
| vessels | Labels | visible, opacity 0.4 |
| virus-positive | Labels | hidden |

The response layer's contrast limits are pinned to 0–1 rather than auto-scaled,
because auto-scaling would hide saturation — the exact failure mode described in the
Problem section.

### Dock

Top to bottom: a file combobox built from `curated_paths()` and labelled
`mouse region slice`; the magicgui parameter panel; **Recompute**; a read-only metrics
readout (`enrichment`, `coverage`, `off_target`, `dice_virus_vs_cd31`,
`vessel_area_fraction`, `virus_cut`); and **Run all → CSV**.

The readout flags `vessel_area_fraction` when it falls outside
`PLAUSIBLE = (0.01, 0.10)`, the CNS-plausible band `sweep_spinal_cord.py` already
defines. One line, and it is what stops someone tuning to an attractive picture at a
physically impossible vessel density.

**Run all → CSV** applies the current parameters across all curated sections and
writes `results/spinal_cord_specificity_tuned.csv`. The distinct filename is the point:
the GUI can never overwrite the published `spinal_cord_specificity.csv`.

### Data location

`analyse_spinal_cord.DATA` is hardcoded to `Z:\Lab\Eric V\BEC Spinal Cords\composites_EV`,
which no lab member other than the author reliably has mounted, while a local copy sits
in [data/composites_EV/](../../data/composites_EV/). The GUI resolves, in order:

1. `--data <dir>`
2. `$BVP_DATA`
3. the `Z:` path
4. `data/composites_EV`

If none exists it exits naming all four. `analyse_spinal_cord.DATA` itself is left
alone — changing it would alter what the batch script reads.

## Error handling

Every failure mode already exists and is already handled by the batch loop as a `SKIP`
line. The GUI must not promote any of them to a crash.

- `tissue_mask` raises `ValueError` on a near-blank section; `stages` raises on an empty
  vessel or parenchyma mask. Catch both, show the message in the readout, and **leave
  the previous layers on screen**. Selecting a bad section must not blank the display.
- napari not installed: print the `pip install -e ".[gui]"` line and exit 1, rather than
  an `ImportError` traceback.
- Failures during **Run all** are collected and reported at the end, matching `main()`.

## Testing

The GUI is not the risky part. The refactor is, because it touches code that produced
numbers already handed off.

- **Acceptance check.** Run `analyse_spinal_cord.py --full` before and after the
  refactor and diff `results/spinal_cord_specificity.csv`. Byte-identical, or the
  refactor is wrong. This is run, not assumed.
- **`tests/test_spinal_stages.py`.** Asserts the new keyword defaults are the module
  constants — pure, no data, runs anywhere — and that
  `analyse(p, r) == summarise(stages(p, r))` on one section, skipped when the data
  directory is absent. That mirrors the repo's existing position that the notebooks
  cannot run from a fresh clone.
- **`python scripts/tune_spinal_cord.py --selftest`.** Headless, no Qt: runs `stages`
  on one section and asserts layer shapes, dtypes, and that the metrics match
  `analyse`. One runnable check that fails if the wiring breaks.

No Qt-in-CI test. Widget plumbing a human exercises on every launch does not need a
test that needs a display server.
