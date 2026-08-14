# An interactive viewer for tuning the spinal-cord vessel segmentation

**Date:** 2026-08-14
**Status:** Approved

## Problem

The spinal-cord follow-up has accumulated **three** vessel definitions, each with its
own operating point, living in three scripts:

| Produced by | reference | thresholds | vessel area fraction | status |
| --- | --- | --- | --- | --- |
| [analyse_spinal_cord.py](../../scripts/analyse_spinal_cord.py) → `spinal_cord_specificity.csv` | `calibrate_reference()` ≈ 0.5 | 0.02 / 0.15, symmetric across channels | **0.41–0.43** | superseded |
| [dice_between_channels.py](../../scripts/dice_between_channels.py) → contour JPGs, handoff mask TIFs | `REFERENCE = 2.0` | CD31 0.03/0.09, virus 0.04/0.12, asymmetric by design | ~0.2–0.3 | the shipped segmentations |
| [enrichment_by_cd31_percentile.py](../../scripts/enrichment_by_cd31_percentile.py) → the headline numbers | — | top-10% CD31 intensity, no vesselness at all | — | the primary measure |

The 0.41–0.43 is the `vessel_area_fraction` column of the committed CSV, and
[export_handoff.py](../../scripts/export_handoff.py) says why it is superseded:
*"its calibrated reference over-segmented ~45% of tissue"* — four times the top of the
CNS-plausible band that [sweep_spinal_cord.py](../../scripts/sweep_spinal_cord.py)
defines as `PLAUSIBLE = (0.01, 0.10)`.

That table is the actual problem. Nobody can see these three definitions at once. Each
lives behind a batch run over 34–63 sections that emits a CSV, so comparing them means
running three scripts and reading three files, and the visual consequence of a setting —
the thing a biologist can actually judge — is never on screen next to the setting. It
took reading four scripts and a CSV column to establish that the headline mask-based
number came from a saturated operating point.

The trap is documented in `sweep_spinal_cord.py`'s docstring:

> In 2D, Jerman saturates everything above `tau * reference_lambda / 2` to exactly 1,
> so the reference and the threshold are coupled and neither can be tuned alone.

A viewer that hid the vesselness response would let someone tune thresholds against a
saturated field, where the threshold is inert and the picture still looks plausible —
which is how the ≈0.5 reference survived as long as it did. So the responses are
first-class layers with contrast limits pinned to 0–1, never auto-scaled.

## Non-goals

- The archive pipeline in [velazquez_rivera_2025/](../../velazquez_rivera_2025/). Frozen,
  published; a tuning GUI for it would invite the edits the freeze prevents.
- Whole-brain lightsheet volumes. At ~500 GB per channel a live re-run is impossible, and
  the tuning that matters happens on 2D sections.
- Editing or replacing any published CSV or figure.
- Undo / parameter history. napari has no built-in and it is speculative until someone
  wants the last five settings back.

## Framework choice

Requirement: runs locally on Windows machines belonging to several lab members, 2D
sections, layered overlays, pan and zoom, live parameter tuning.

| Option | Verdict |
| --- | --- |
| **napari + magicgui** | **Chosen.** Supplies the whole viewer half: pan/zoom, a layer list with per-layer visibility, opacity, blending and colormap, a `Labels` layer built for masks, and `magicgui`, which derives a parameter panel from a function signature. Bioimage-standard, so the audience may already know it. |
| Panel / Bokeh + datashader | The credible runner-up, and better had the requirement been "browser, nothing installed". It is not — everyone runs locally, so a server buys nothing, and the layer model, blending and mask overlays all become our code. |
| Streamlit | Re-runs the script on every widget change, no real pan/zoom. Wrong for a pipeline with a multi-second stage. |
| Jupyter + ipywidgets / stackview | Lightest, and the lab already runs notebooks. Weak pan/zoom, awkward multi-file browsing, and the result is a notebook, not an application. |
| pyqtgraph + hand-written Qt | Reimplements most of napari. Justified only if napari could not be installed. |
| Fiji / QuPath | Cannot call `vessel_utils` without a pyimagej bridge. |
| itkwidgets | Tempting since `itk` is already a dependency, but volume-oriented with a weaker parameter-widget story. |

Installability was verified, not assumed: on the project's Python 3.14 `.venv`,
`pip install --dry-run "napari[all]"` resolves to napari 0.8.0 with PyQt6 6.10.2. PyQt6
publishes `abi3` wheels, so the new interpreter is no barrier and **no second virtual
environment is needed**.

## What already exists

Most of the backend is written. The GUI is a fourth consumer of helpers that six
visualisation scripts already share, not new machinery:

| Needed | Already in the repo |
| --- | --- |
| arrays for one section, both masks | `dice_between_channels.channel_masks(path)` — docstring already says "also used for visualisation" |
| response computed once, thresholds reused | `sweep_spinal_cord.prepare()`, documented as "per-section work that does not depend on the threshold" |
| load + tissue-mask + skip bad sections | `analyse_spinal_cord.load_sections(paths)` — "the same behaviour each script implemented separately" |
| the curated file list | `analyse_spinal_cord.section_paths(full=True)` |
| virus MAD threshold | `analyse_spinal_cord.virus_cut(virus, parenchyma, k=VIRUS_K)` — already takes `k` |
| percentile vessel mask | `enrichment_by_cd31_percentile.top_q_mask(cd31, tissue, q)` — already takes `q` |
| the metrics | `vessel_utils.metrics` |

`_segment(channel, tissue, low, high)` already takes its thresholds as arguments. Only
`REFERENCE`, `SIGMAS` and `MIN_VESSEL_PX` are still module-level.

**`analyse_spinal_cord.py` is therefore not refactored.** The earlier plan to split
`analyse()` into `stages()`/`summarise()` is dropped: `channel_masks` already returns the
arrays, `virus_cut` is already parameterised, and the enrichment scalars are ~8 lines
computed in the GUI from arrays it already holds. Not touching that module also removes
any risk to the script that produced the published numbers.

## Architecture

### The cost cascade

The design is a caching problem, not a widget problem:

| Knob | Invalidates | Cost |
| --- | --- | --- |
| file | everything | disk read + GrabCut (already disk-cached in `results/tissue_masks/`) |
| `sigmas`, `reference` | both responses, all masks, all metrics | seconds |
| `cd31_low/high`, `virus_low/high`, `min_vessel_um2` | masks, metrics | ~100 ms |
| `virus_k`, `q` | one mask, its metrics | ~10 ms |

Two `functools.lru_cache`d functions implement it, and that is the entire performance
design:

```
load(path)                                  -> virus, cd31, tissue
response(path, which_channel, sigmas, reference) -> vesselness in [0, 1]
```

`maxsize=8` on `response` — two channels across four sections, enough to flip back and
forth without re-filtering. Everything below `response` re-runs on every widget change,
because at ~100 ms it is free.

So the thresholds, `virus_k` and `q` use magicgui's `auto_call` and update on release;
`sigmas` and `reference` sit behind an explicit **Recompute** button, because by
construction they miss the cache.

`load()` delegates to `next(load_sections([path]), None)`, which reuses the existing
load-mask-skip behaviour exactly; `None` means the section is unusable and the reason has
been printed to the console.

### Both virus definitions are always shown

The two mask-based operating points differ *structurally*, not just numerically:
`dice_between_channels` segments the virus with vesselness, `analyse_spinal_cord`
thresholds it with parenchyma median + `k`·MAD. Rather than make a preset switch between
two different pipelines, the GUI computes **both** every time — each is cheap once the
response is cached, both are real definitions in use, and having them on screen together
is the comparison the repo currently cannot make.

A preset is then only a set of numbers, never a different code path.

```python
PRESETS = {
    "shipped":     dict(reference=2.0, cd31=(0.03, 0.09), virus=(0.04, 0.12)),
    "superseded":  dict(reference=None, cd31=(0.02, 0.15), virus=(0.02, 0.15)),
}
```

`reference=None` on the superseded preset means "call `calibrate_reference()`", which
runs once on selection and is then editable — never silently recomputed per section,
because its docstring is explicit that the value is dataset-wide on purpose: computed per
image, "a fixed threshold would mean a different thing in every section — and the
comparison being made here is precisely across sections, regions and mice."

Default preset is **shipped**. It is what produced the contour JPGs and handoff TIFs, and
it is the one not marked superseded. Selecting `superseded` renders its 43%-of-tissue
mask over grey matter with the plausible-band warning lit, which is a more convincing
account of why it was retired than the sentence in `export_handoff.py`.

### Layers

Bottom to top, using napari's built-in per-layer visibility, opacity and blending:

| Layer | Type | Default |
| --- | --- | --- |
| CD31 | Image, magenta, additive | visible |
| virus reporter | Image, green, additive | visible |
| CD31 vesselness | Image, `turbo`, clim pinned 0–1 | hidden |
| virus vesselness | Image, `turbo`, clim pinned 0–1 | hidden |
| tissue | Labels | hidden |
| CD31 vessels | Labels | visible, opacity 0.4 |
| virus vessels (Jerman) | Labels | visible, opacity 0.4 |
| virus⁺ (MAD, `k`) | Labels | hidden |
| CD31 top-q% | Labels | hidden |

Magenta for CD31, green for virus — the convention already used across the merge views,
notebooks and handoff. Response clims are **pinned**, not auto-scaled, for the reason in
the Problem section.

### Dock

Top to bottom: a preset combobox; a file combobox from `section_paths(full=True)`,
labelled `mouse region slice`; the magicgui parameter panel (`sigmas`, `reference`,
`cd31_low/high`, `virus_low/high`, `min_vessel_um2`, `virus_k`, `q`); **Recompute**; a
read-only metrics readout; and **Run all → CSV**.

Readout, all cheap given the cached responses:

- `dice`, `jaccard`, `precision`, `recall` — virus-Jerman vs CD31, i.e. `score_section`
- `cd31_af`, `virus_af` — area fractions within tissue
- `enrichment`, `coverage`, `off_target` — the MAD definition, from `virus_cut`
- `enrichment_q` — the percentile measure at the current `q`

`cd31_af` is flagged when it falls outside `PLAUSIBLE = (0.01, 0.10)`. One line, and it is
what stops someone tuning to an attractive picture at a physically impossible vessel
density — the failure that produced the superseded operating point.

`top_q_mask` selection is guarded the way `enrichment_curve` guards it: if the achieved
fraction exceeds 1.5× the nominal `q` (ties at the cutoff, from quantised or saturated
CD31), `enrichment_q` reads `n/a` rather than a misleading number.

**Run all → CSV** applies the current parameters across all curated sections and writes
`results/spinal_cord_tuned.csv`. The distinct filename is the point: the GUI can never
overwrite `dice_between_channels_full.csv` or `spinal_cord_specificity.csv`.

### Data location

`analyse_spinal_cord.DATA` is hardcoded to
`Z:\Lab\Eric V\BEC Spinal Cords\composites_EV`, which no lab member other than the author
reliably has mounted, while a local copy sits in
[data/composites_EV/](../../data/composites_EV/). This breaks all eight scripts that
import from that module, not just the GUI, so the fix goes there rather than being
monkeypatched from the viewer:

```python
DATA = Path(os.environ.get("BVP_DATA", r"Z:\Lab\Eric V\BEC Spinal Cords\composites_EV"))
```

Behaviour is unchanged when the variable is unset. `tune_spinal_cord.py --data <dir>` sets
it before importing; if neither the variable, the `Z:` path, nor `data/composites_EV`
exists, exit naming all three.

### Files touched

| File | Change |
| --- | --- |
| `scripts/dice_between_channels.py` | split `_response()` out of `_segment()` so the response can be cached independently; `_segment`/`channel_masks` gain `reference`, `sigmas`, `min_size` kwargs defaulting to the module constants |
| `scripts/analyse_spinal_cord.py` | one line: `DATA` reads `$BVP_DATA` |
| `scripts/tune_spinal_cord.py` | new, ~180 lines, no pipeline logic |
| `pyproject.toml` | `gui = ["napari[all]"]`, kept out of `dev` so the test install stays free of Qt and ~45 packages |
| `tests/test_dice_params.py` | new |

`enrichment_by_cd31_percentile.py` is untouched — `top_q_mask` already takes `q`.

## Error handling

Every failure mode already exists and is already handled by the batch loops as a `SKIP`
line. The GUI must not promote any to a crash.

- `tissue_mask` raises `ValueError` on a near-blank section; `score_section` raises when a
  channel yields an empty mask. `load()` returns `None` and the metrics readout shows the
  failure, **leaving the previous layers on screen** — selecting a bad section must not
  blank the display.
- napari not installed: print the `pip install -e ".[gui]"` line and exit 1, not an
  `ImportError` traceback.
- Failures during **Run all** are collected and reported at the end, matching `main()`.

## Testing

The GUI is not the risky part. The `dice_between_channels` refactor is, because it
produced the shipped segmentations and the handoff TIFs.

- **Acceptance check.** Run `dice_between_channels.py --full` before and after and diff
  `results/dice_between_channels_full.csv`. Byte-identical, or the refactor is wrong.
  Run, not assumed.
- **`tests/test_dice_params.py`.** Asserts the new keyword defaults are the module
  constants — pure, no data, runs anywhere — and that
  `channel_masks(p) == channel_masks(p, reference=REFERENCE, sigmas=SIGMAS, min_size=MIN_VESSEL_PX)`
  on one section, skipped when the data directory is absent. That mirrors the repo's
  existing position that the notebooks cannot run from a fresh clone.
- **`python scripts/tune_spinal_cord.py --selftest`.** Headless, no Qt: builds every layer
  for one section and asserts shapes, dtypes, and that the readout matches
  `score_section`. One runnable check that fails if the wiring breaks.

No Qt-in-CI test. Widget plumbing a human exercises on every launch does not need a test
that needs a display server.
