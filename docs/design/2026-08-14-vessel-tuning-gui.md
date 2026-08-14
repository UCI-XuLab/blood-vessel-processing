# An interactive viewer for tuning vessel segmentation

**Date:** 2026-08-14
**Status:** Approved

A napari-based viewer that runs the `vessel_utils` segmentation pipeline live on any
2D vessel image, shows every intermediate as a toggleable layer, and lets the
parameters be tuned with the result on screen. Datasets are described by presets, not
by code.

## Problem

Every parameter in this repository is tuned by editing a constant, running a batch over
tens of images, and reading a CSV. The visual consequence of a setting — the thing a
biologist can actually judge — is never on screen next to the setting. Two costs follow.

**Nobody can see what a setting does.** The spinal-cord follow-up alone has accumulated
three vessel definitions with three operating points, in three scripts:

| Produced by | reference | thresholds | vessel area fraction | status |
| --- | --- | --- | --- | --- |
| [analyse_spinal_cord.py](../../scripts/analyse_spinal_cord.py) | `calibrate_reference()` ≈ 0.5 | 0.02 / 0.15, symmetric | **0.41–0.43** | superseded |
| [dice_between_channels.py](../../scripts/dice_between_channels.py) | `REFERENCE = 2.0` | ref 0.03/0.09, test 0.04/0.12 | ~0.2–0.3 | the shipped segmentations |
| [enrichment_by_cd31_percentile.py](../../scripts/enrichment_by_cd31_percentile.py) | — | top-10% intensity, no vesselness | — | the primary measure |

The 0.41–0.43 is the `vessel_area_fraction` column of the committed
`spinal_cord_specificity.csv`, and [export_handoff.py](../../scripts/export_handoff.py)
records why it was retired: *"its calibrated reference over-segmented ~45% of tissue"* —
four times the top of the plausible band that
[sweep_spinal_cord.py](../../scripts/sweep_spinal_cord.py) defines as
`PLAUSIBLE = (0.01, 0.10)`. Establishing that took reading four scripts and a CSV column.
A viewer would have shown it in one glance: a "vessel" mask covering the grey matter.

**And the tuning does not transfer.** Each new dataset — brain slices, condensed
lightsheet, the next cohort — starts this loop again from scratch, because there is
nowhere to put a starting point and no way to look at one.

The specific trap is documented in `sweep_spinal_cord.py`:

> In 2D, Jerman saturates everything above `tau * reference_lambda / 2` to exactly 1,
> so the reference and the threshold are coupled and neither can be tuned alone.

A viewer that hid the vesselness response would let someone tune thresholds against a
saturated field, where the threshold is inert and the picture still looks plausible —
which is how the ≈0.5 reference survived. So the responses are first-class layers with
contrast limits **pinned** to 0–1, never auto-scaled.

## Scope

In: any 2D single- or two-channel vessel image, the `vessel_utils` pipeline, live tuning,
per-layer toggling, pan and zoom, dataset presets, batch re-run to a CSV.

Out:

- **The archive pipeline** in [velazquez_rivera_2025/](../../velazquez_rivera_2025/).
  Excluded on principle, not just for scope: `itk.RescaleIntensityImageFilter` stretches
  each image's own min/max across 0–255, so `thresh=230` is a *relative* criterion that
  means something different in every image. In a viewer whose purpose is comparing a
  setting across files, a backend whose settings are incomparable across files would be
  actively misleading. Add later behind a clear warning if reproducing a brain-slice
  notebook in the viewer becomes worth it.
- **3D volumes**, for now. The structure is kept 3D-ready (below) so this is an addition
  rather than a rewrite.
- **Editing or replacing any published CSV or figure.**
- **Undo / parameter history.** napari has no built-in and it is speculative until
  someone wants the last five settings back.

## Framework choice

Requirement: runs locally on Windows machines belonging to several lab members, 2D,
layered overlays, pan and zoom, live parameter tuning.

| Option | Verdict |
| --- | --- |
| **napari + magicgui** | **Chosen.** Supplies the entire viewer half: pan/zoom, a layer list with per-layer visibility, opacity, blending and colormap, a `Labels` layer built for masks, and `magicgui`, which derives a parameter panel from a function signature. Natively nD, which is most of the 3D-readiness for free. Bioimage-standard, so the audience may already know it. |
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

## Architecture

### The pipeline is already generic; only the dataset description is not

Sorting the spinal-cord scripts into "pipeline" and "description of this dataset" splits
cleanly, and everything in the second column is a value, not code:

| Dataset description | Spinal-cord value | Varies because |
| --- | --- | --- |
| where the files are | `Z:\...\composites_EV`, `Fig N_..._sliceK.tif` | every dataset names files differently |
| spacing | `0.650193` µm/px | brain slices run at implicit 1×1; condensed lightsheet at ~3 µm |
| which channel is which | ch1 = CD31 reference, ch0 = virus test | CLAUDE.md warns layout varies: `[0]/[1]`, `[:, :, 1]` for M74, alternating files via `load_channels()` |
| how tissue is found | entropy-GrabCut on the channel sum | brain slices use a Triangle threshold; lightsheet needs something else |
| plausible area fraction | `(0.01, 0.10)` | a CNS-section figure, not a universal one |

What remains is the invariant pipeline, which is exactly `vessel_utils`' decomposition:

```
image, spacing
  → tissue/ROI mask
  → normalise within ROI          (percentile clip, so response scales are comparable)
  → jerman_vesselness             (bounded [0,1], sigmas in µm)
  → segment                       (hysteresis, ROI, min_size, holes, closing)
  → metrics                       (per-channel area fraction; pairwise if two channels)
```

So the GUI drives `vessel_utils` directly and a dataset becomes a **preset**: a set of
numbers plus two enum choices. There are no adapter classes and no extension points,
because there is nothing per-dataset left that is code.

### Presets

```python
PRESETS = {
    "generic":               dict(spacing=None, roles=(1, 0), mask="percentile",
                                  reference=None, ref_thr=(0.03, 0.09),
                                  test_thr=(0.03, 0.09), min_vessel_um2=6.0,
                                  plausible=(0.0, 1.0)),
    "spinal-cord shipped":   dict(spacing=0.650193, roles=(1, 0), mask="grabcut",
                                  reference=2.0, ref_thr=(0.03, 0.09),
                                  test_thr=(0.04, 0.12), min_vessel_um2=6.0,
                                  plausible=(0.01, 0.10)),
    "spinal-cord superseded": dict(..., reference=None, ref_thr=(0.02, 0.15),
                                  test_thr=(0.02, 0.15)),
    "brain-slice starting point": dict(spacing=1.0, mask="brain", reference=None, ...),
}
```

`spacing=None` means read it from the TIFF resolution tags and fall back to 1.0 with the
field flagged. `reference=None` means calibrate over a sample of the loaded directory once
on preset selection, then leave it editable — never silently per-image, for the reason
below.

Default preset is **generic**. `spinal-cord shipped` reproduces the operating point behind
the contour JPGs and handoff TIFs. `spinal-cord superseded` exists so its 43%-of-tissue
mask can be looked at; that is a more convincing account of why it was retired than the
sentence in `export_handoff.py`.

`brain-slice starting point` is labelled exactly that, in the UI as well as here. It
reproduces no published figure — those came from the archive pipeline, which this viewer
deliberately does not run — and must not be presented as if it did.

### `reference_lambda` is per-dataset, never per-image

This is the one piece of pipeline the generalization makes *more* important, not less.
`max_eigenvalue`'s own docstring: run it once over a representative subset and pass the
result to every `jerman_vesselness` call, because that "is what lets a single threshold
mean the same thing in both channels and across slices". `vessel_utils.chunked.apply_vesselness`
refuses to run without it. `calibrate_reference` in the spinal-cord script spells out the
consequence of getting it wrong: computed per image, "a fixed threshold would mean a
different thing in every section — and the comparison being made here is precisely across
sections, regions and mice."

A cross-dataset viewer must not weaken that. So:

- A new `vessel_utils.vesselness.reference_lambda(images, sigmas, spacing, masks=None, percentile=99.9)`
  — the median of per-image `max_eigenvalue` over a sample. Twelve lines, next to the
  function whose docstring already prescribes the procedure, and unit-testable. It goes in
  `vessel_utils` rather than the GUI because CLAUDE.md puts active improvements there, and
  because it is a vesselness concern, not a widget one.
- The GUI computes it **once per (dataset, sigmas)**, displays it, and lets it be edited so
  the calibrated value can be compared against a fixed one — the honest way to handle a
  knob coupled to the thresholds.
- Changing it invalidates the response cache, which is correct.

### The cost cascade

The design is a caching problem, not a widget problem:

| Knob | Invalidates | Cost |
| --- | --- | --- |
| file, tissue-mask method | everything | disk read + masking (GrabCut ~7–15 s, hence cached) |
| `spacing`, `sigmas`, `reference` | both responses, all masks, all metrics | seconds |
| `ref_thr`, `test_thr`, `min_vessel` | masks, metrics | ~100 ms |
| `k` (MAD cut), `q` (percentile) | one mask, its metrics | ~10 ms |

Two `functools.lru_cache`d functions implement it, and that is the entire performance
design:

```
load(path, mask_method)                       -> channels, tissue
response(path, channel_index, sigmas, spacing, reference) -> vesselness in [0, 1]
```

`maxsize=8` on `response` — two channels across four files, enough to flip back and forth
without re-filtering. Everything below `response` re-runs on every widget change, because
at ~100 ms it is free.

So the thresholds, `k` and `q` use magicgui's `auto_call` and update on release;
`spacing`, `sigmas` and `reference` sit behind an explicit **Recompute**, because by
construction they miss the cache. `cl_dice` sits behind a checkbox — it skeletonises, so
it is the one metric not free at slider rates.

### Layers

Bottom to top, using napari's built-in per-layer visibility, opacity and blending. Names
come from the preset when it supplies them (`CD31 (reference)`), otherwise the channel
index (`ch1 (reference)`):

| Layer | Type | Default |
| --- | --- | --- |
| reference channel | Image, magenta, additive | visible |
| test channel | Image, green, additive | visible (absent in single-channel mode) |
| reference vesselness | Image, `turbo`, clim pinned 0–1 | hidden |
| test vesselness | Image, `turbo`, clim pinned 0–1 | hidden |
| tissue | Labels | hidden |
| reference vessels | Labels | visible, opacity 0.4 |
| test vessels | Labels | visible, opacity 0.4 |
| test⁺ (intensity, median + k·MAD) | Labels | hidden |
| reference top-q% | Labels | hidden |

Magenta reference / green test follows the convention already used across the merge views,
notebooks and handoff. Response clims are pinned, not auto-scaled, for the reason in the
Problem section.

The last two layers are the two *non-vesselness* vessel definitions this repo uses — the
MAD intensity cut and the top-q% percentile mask, which is the spinal-cord project's
actual primary measure. Both are cheap, both are real definitions in use, and rendering
them alongside the vesselness masks is the comparison nothing in the repo can currently
make. Consequently a preset never selects between pipelines; it only sets numbers.

### Dock

Preset combobox; source directory and file combobox; the magicgui parameter panel
(`spacing`, `sigmas`, `reference`, `reference`/`test` channel indices, tissue-mask method,
`ref_thr`, `test_thr`, `min_vessel`, `k`, `q`, `cl_dice` checkbox); **Recompute**; a
read-only metrics readout; **Run all → CSV**.

Readout — the generic core always, the pairwise extras only with two channels:

- `ref_af`, `test_af` — area fraction within tissue
- `dice`, `jaccard`, `precision`, `recall` — test vs reference, and `cl_dice` when enabled
- `enrichment`, `coverage`, `off_target` — from the MAD cut
- `enrichment_q` — the percentile measure at the current `q`

`ref_af` is flagged when it falls outside the preset's plausible band, with the band shown
and editable and labelled with its provenance. That single line is what stops someone
tuning to an attractive picture at a physically impossible vessel density — the failure
that produced the superseded operating point. The band is per-preset because it is a
property of the tissue, not of the software.

`enrichment_q` is guarded the way `enrichment_curve` guards it: if the achieved fraction
exceeds 1.5× the nominal `q` — ties at the cutoff, from quantised or saturated intensities
— it reads `n/a` rather than a misleading number.

**Run all → CSV** applies the current parameters across the loaded directory and writes
`<dir>/../results/tuned_<preset>_<timestamp>.csv`. No fixed filename, so it can never
collide with `dice_between_channels_full.csv`, `spinal_cord_specificity.csv`, or anything
else already produced.

### 3D-readiness

Cheap now, expensive later, so it is done now:

- `spacing` is a per-axis tuple throughout, never a scalar. The UI shows one box for 2D and
  would show N for a volume.
- `min_vessel` is entered in µm² (2D) / µm³ (3D) and converted with `prod(spacing)`, not in
  pixels.
- No 2D-only calls in `gui.py`. `jerman_vesselness` already takes 2D or 3D, `clean_mask`
  already branches `disk`/`ball` on `ndim`, `remove_small_objects` is nD.

What is left for 3D is the loader (`storage.open_volume` and level switching) and tuning
on a crop, since a live whole-volume re-run is not physically possible. Those are
additions, not rewrites.

### Files

| File | Change |
| --- | --- |
| `vessel_utils/gui.py` | new, ~320 lines. Imports napari lazily so `import vessel_utils` stays cheap, matching the existing lazy re-export convention. |
| `vessel_utils/vesselness.py` | new `reference_lambda()`, ~12 lines |
| `pyproject.toml` | `gui = ["napari[all]"]` kept out of `dev` so the test install stays free of Qt and ~45 packages; `[project.scripts] bvp-tune = "vessel_utils.gui:main"` |
| `scripts/analyse_spinal_cord.py` | one line: `DATA = Path(os.environ.get("BVP_DATA", r"Z:\..."))`. Separable from the GUI and worth it independently — that hardcoded share breaks all eight scripts importing from this module for any lab member without `Z:` mounted. Behaviour unchanged when unset. |
| `tests/test_gui_pipeline.py` | new |

**Nothing that produced a published number is modified.** Driving `vessel_utils` directly
means the GUI needs neither `dice_between_channels.channel_masks` nor a split of
`analyse_spinal_cord.analyse`, both of which earlier drafts of this design called for.

The tissue-mask dropdown reuses existing implementations rather than adding a fifth:
`percentile` → `vessel_utils.correct.tissue_mask`, `grabcut` →
`vessel_utils._vendor.compute_entropy_grabcut`, `brain` →
`velazquez_rivera_2025.vessels.get_brain_mask`, plus `none`. The `brain` option is labelled
**8-bit input** in the UI: it calls `cv2.threshold(..., THRESH_TRIANGLE)`, which does not
accept uint16 or float, so on 16-bit microscopy it must be given a converted copy.
Importing the frozen archive from the active package is a read-only import and does not
touch the freeze; it beats duplicating four lines.

## Error handling

- Tissue masking raises `ValueError` on a near-blank image; segmentation can yield an
  empty mask. Show the failure in the readout and **leave the previous layers on screen** —
  selecting a bad file must not blank the display. This mirrors the `SKIP` lines the batch
  scripts already print.
- napari not installed: print the `pip install -e ".[gui]"` line and exit 1, not an
  `ImportError` traceback.
- No readable images in the source directory: exit naming the directory and the patterns
  tried.
- Spacing absent from the TIFF tags: fall back to 1.0 and flag the field, rather than
  silently proceeding — with sigmas in µm, a wrong spacing silently searches the wrong
  vessel calibres.
- Failures during **Run all** are collected and reported at the end, matching the batch
  scripts.

## Testing

The GUI's own widget plumbing is exercised by a human on every launch and does not need a
test that needs a display server. What needs testing is the pipeline wiring and the one new
library function — and both can be tested with **no data and no Qt**, because
`vessel_utils.synth.phantom(shape, spacing, seed)` returns a simulated acquisition with its
exact ground-truth mask.

- **`tests/test_gui_pipeline.py`** — on a 2D slice of `phantom()`: build every layer,
  assert shapes, dtypes and that each mask is a subset of the tissue mask; assert the
  readout equals direct `vessel_utils.metrics` calls on the same arrays; assert the
  phantom's Dice against its known ground truth clears a floor, which catches a pipeline
  wired together wrongly rather than merely running. Seeded, so it is deterministic.
- **`reference_lambda`** — on several phantoms: the result is the median of the
  per-image values, is invariant to input order, and is unchanged by appending a duplicate
  image. That last one is the property that matters: it is what makes the value a dataset
  constant rather than a sample artefact.
- **`python -m vessel_utils.gui --selftest`** — the same wiring check as a runnable
  command, so the check exists where someone debugging the GUI will find it.
- **No regression check needed on the existing scripts**, because none of their behaviour
  changes. The `$BVP_DATA` line is verified by confirming `DATA` is unchanged when the
  variable is unset.
