# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the paper *"Specific targeting of brain endothelial cells using enhancer AAV vectors"* (Zenodo DOI 10.5281/zenodo.14876892). It quantifies blood vessel segmentation in mouse brain microscopy, comparing an AAV reporter channel against a ground-truth vessel-stain channel.

Two parts: a small shared package, [vessel_utils/](vessel_utils/), and 27 notebooks that drive it. Each notebook corresponds to one specimen/figure and holds that specimen's hand-tuned parameters — which are the actual scientific content.

## Commands

```bash
pip install -e .                    # optional; notebooks bootstrap sys.path without it
pip install -e ".[dev]" && pytest   # run the test suite
pytest tests/test_equivalence.py -k detect_vessels    # a single test
jupyter lab                         # run the notebooks
```

The notebooks cannot be executed from a fresh clone: paths are hardcoded to `/media/data/u01/...` on the lab's Linux workstation and the imaging data is not in the repository. Anything you change must be verified through the test suite instead.

## Architecture

`vessel_utils` is split by pipeline stage, so a notebook's import block reads as a description of what that notebook does:

| module | contents |
| --- | --- |
| [io.py](vessel_utils/io.py) | `read_tif`, `load_channel`, `load_channels`, `load_3_channels` |
| [enhance.py](vessel_utils/enhance.py) | `auto_contrast`, `gamma_correction`, `histogram_equalization`, `n4_bias_correction`, `compute_average_image` |
| [vessels.py](vessel_utils/vessels.py) | `detect_vessels`, `process_vessels`, `get_brain_mask` |
| [metrics.py](vessel_utils/metrics.py) | `dice_coefficient`, `iou`, `precision`, `recall`, `rand_index` |
| [viz.py](vessel_utils/viz.py) | `show`, `show3`, `show_4`, `save_figure` |

`__init__.py` re-exports every name but resolves it lazily, so importing `metrics` does not drag in `itk`.

Each notebook opens with a bootstrap cell that walks up from the working directory to find the repo root and adds it to `sys.path`. That is why `pip install -e .` is optional — do not remove the bootstrap, the lab runs these notebooks directly and must not acquire a setup step it can silently skip.

## What lives in the notebooks, not the package

Batch loops, all parameter values, per-slice overrides (`if i == 0: THRESH = 3200`), and the `ALPHA`/`BETA`/`GAMMA` constants. These are per-specimen tuning and belong where a reviewer reading the figure's notebook will see them. `run_test` in `3-M13 run 2` also stays local — it closes over the notebook's `filepath` and embeds slice-specific thresholds.

**Do not "clean up" hand-tuned constants or commented-out contrast lines.** They record what was tried and rejected for that specimen.

Notebook naming: `process_enhanced_rgb <figure>-<mouse>.ipynb`, e.g. `3-M1` = Figure 3, mouse M1; `Supple-4a-M2` = Supplemental Figure 4a, mouse M2. Suffixes like `run 2` / `newcode` are reprocessing attempts kept alongside the original.

## Things that will bite you

- **Vesselness polarity.** `detect_vessels` runs the objectness filter with `SetBrightObject(False)`, so its output is *dark* where vessels are — which is why `process_vessels` does `np.invert(vessel_image > thresh)`. Changing either without the other silently inverts every mask.
- **`detect_vessels` objectness parameters are passed explicitly at every call site.** Notebooks with `ALPHA`/`BETA`/`GAMMA` constants pass `alpha=ALPHA, beta=BETA, gamma=GAMMA`; the rest pass literals. `beta` is `0.5` in most notebooks and `1.0` in `M7`, `M12`, `M74` and `process_blood_vessel_brain`. Never let a call fall back to the signature defaults — the tuning has to stay visible.
- **`iou` is dtype-sensitive.** `union` is computed with `+`, which is logical OR on boolean arrays but arithmetic addition on numeric ones. The pipeline passes boolean masks, so published values are a true IoU; passing float masks silently halves the score. Preserved deliberately, pinned by a test.
- **`precision` and `recall` are the only asymmetric metrics.** Several notebooks deliberately place the ground-truth channel second, and some report both directions.
- **Channel layout varies per dataset** — check before touching a batch loop: `curr_img[0]`/`curr_img[1]` (channel-first stack), `curr_img[:, :, 1]` (RGB planes, e.g. `M74`), or `load_channels()` which treats *alternating files* as ch1/ch2 pairs, so a stray file in the glob shifts every pairing.
- **Notebook anatomy.** Tuning cells operate on a single slice selected by a module-level `IDX`; the final "Run the whole thing" cell *re-declares* its own parameters. A change made while tuning does not reach the batch loop unless copied down manually.

## The three pipelines

**[process_brain_slices/](process_brain_slices/)** (21 notebooks) — per slice, per channel: `gamma_correction` → `auto_contrast` → `get_brain_mask` → mask the channel → `detect_vessels` → `process_vessels` → re-mask → write `.tif` → ch0-vs-ch1 metrics into a CSV. Newer notebooks (`M46`, `M49`, `Supple-4a-*`) add `n4_bias_correction` first; older ones do not. Both variants are current.

**[process_thickness/](process_thickness/)** (4 notebooks) — different algorithm despite the shared filename prefix, and no Hessian filter at all: `n4_bias_correction` → median filter → Triangle threshold **multiplied by** an adaptive-mean threshold → `remove_small_objects` → optional `binary_fill_holes` → `medial_axis(..., return_distance=True)`. Per connected skeleton component it emits `thickness = mean(2 * distance)` and `length = pixel count`.

**[process_lightsheet_full_brain/](process_lightsheet_full_brain/)** (2 notebooks) — run [condense_lightsheet_brain.ipynb](process_lightsheet_full_brain/condense_lightsheet_brain.ipynb) first; it averages the ~2600-plane z-stack in chunks of 20 (3 µm/slice → 60 µm/output) to make the volume tractable. Then [process_blood_vessel_brain.ipynb](process_lightsheet_full_brain/process_blood_vessel_brain.ipynb) applies the brain-slices pipeline to the condensed tiffs. Images are ~10k px per side, so preview cells zoom via `xlim`/`ylim` rather than downsampling.

## Tests

[tests/baseline.py](tests/baseline.py) recovers the pre-extraction helper definitions straight out of git at `BASELINE_REV`, so the ground truth cannot drift.

- [tests/test_equivalence.py](tests/test_equivalence.py) — runs every historical variant of every helper side by side with the shared version on synthetic inputs and requires bit-identical output. This is what makes the extraction trustworthy; if you change a `vessel_utils` function, this is the test that will tell you whether you changed a published result.
- [tests/test_notebooks.py](tests/test_notebooks.py) — asserts no notebook still defines a shared helper, every used name is imported, no import is unused, call sites and their arguments survived, and markdown and stored outputs were untouched.

Two behaviours are deliberately *not* equivalent to the originals, and are asserted separately: `preprocess_image` was dropped (dead in all 9 notebooks, would have raised `TypeError` if called), and `n4_bias_correction` no longer raises `NameError` when `shrink_factor <= 1` (`maskImage` was only bound inside the `> 1` branch; no call site ever hit it).
