# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the paper *"Specific targeting of brain endothelial cells using enhancer AAV vectors"* (Zenodo DOI 10.5281/zenodo.14876892). It quantifies blood vessel segmentation in mouse brain microscopy, comparing an AAV reporter channel against a ground-truth vessel-stain channel.

Three parts:

- **[velazquez_rivera_2025/](velazquez_rivera_2025/)** — the published implementation. **Frozen.** It produced the numbers in the paper, and [tests/test_archive_frozen.py](tests/test_archive_frozen.py) fails if any file in it changes. Improvements do not go here.
- **[vessel_utils/](vessel_utils/)** — active development for follow-up work. Free to differ from the archive; the point is to do better, not to reproduce.
- **27 notebooks** driving the archive. Each corresponds to one specimen/figure and holds that specimen's hand-tuned parameters — the actual scientific content. They stay pinned to the archive so they keep reproducing the published figures.

## Commands

Everything runs off a project virtual environment. Create it once, then install
the `[dev]` extra — that pulls the runtime deps, the test tooling, and the
notebook tooling (pandas, jupyterlab, nbconvert/nbclient/ipykernel), so a fresh
`.venv` can run the scripts, the tests, and the notebooks:

```bash
python -m venv .venv                          # create the project venv (.venv is gitignored)
.venv\Scripts\activate                        # Windows  (source .venv/bin/activate on Unix)
pip install -e ".[dev]"                       # install everything: deps, tests, notebooks
pytest                                         # run the test suite
pytest tests/test_equivalence.py -k detect_vessels    # a single test
jupyter lab                                    # run the notebooks
```

`pip install -e .` (without `[dev]`) is enough just to import `vessel_utils`; the
notebooks also bootstrap `sys.path`, so they import the packages without any
install.

The notebooks cannot be executed from a fresh clone: paths are hardcoded to `/media/data/u01/...` on the lab's Linux workstation and the imaging data is not in the repository. Anything you change must be verified through the test suite instead.

## Architecture

Both packages are split by pipeline stage, so an import block reads as a description of what the caller does. Importing `metrics` does not drag in `itk` either way, but by different means: the frozen archive re-exports every name lazily through a `__getattr__`, while `vessel_utils/__init__.py` imports nothing at all and callers name the submodule (`from vessel_utils.threshold import segment`). Do not add imports to `vessel_utils/__init__.py` — a test pins the fact that it stays empty.

**Archive — [velazquez_rivera_2025/](velazquez_rivera_2025/)** (frozen):

| module | contents |
| --- | --- |
| [io.py](velazquez_rivera_2025/io.py) | `read_tif`, `load_channel`, `load_channels`, `load_3_channels` |
| [enhance.py](velazquez_rivera_2025/enhance.py) | `auto_contrast`, `gamma_correction`, `histogram_equalization`, `n4_bias_correction`, `compute_average_image` |
| [vessels.py](velazquez_rivera_2025/vessels.py) | `detect_vessels`, `process_vessels`, `get_brain_mask` |
| [metrics.py](velazquez_rivera_2025/metrics.py) | `dice_coefficient`, `iou`, `precision`, `recall`, `rand_index` |
| [viz.py](velazquez_rivera_2025/viz.py) | `show`, `show3`, `show_4`, `save_figure` |

**Active — [vessel_utils/](vessel_utils/)**:

| module | contents |
| --- | --- |
| [vesselness.py](vessel_utils/vesselness.py) | `jerman_vesselness`, `hessian_eigenvalues`, `max_eigenvalue` |
| [threshold.py](vessel_utils/threshold.py) | `hysteresis_threshold`, `otsu_threshold`, `clean_mask`, `segment` |
| [metrics.py](vessel_utils/metrics.py) | `dice`, `jaccard`, `precision`, `recall`, `cl_dice`, `area_fraction`, `agreement`, `agreement_by_calibre` |
| [synth.py](vessel_utils/synth.py) | `vascular_tree`, `render_tree`, `simulate_acquisition`, `phantom` |
| [benchmark.py](vessel_utils/benchmark.py) | `score_segmentation`, `run_benchmark`, `sweep_condition`, `summarise` |
| [gui.py](vessel_utils/gui.py) | `main`, `build_viewer`, `stages`, `readout`, `reference_lambda`, `PRESETS` — the interactive tuning viewer, run as `bvp-tune <dir>`. The one module here that is an application rather than a pipeline stage: it drives the others and adds no algorithm. Needs the `[gui]` extra (`napari[all]`), imported inside functions so `import vessel_utils.gui` stays cheap. A dataset is a `PRESETS` entry — numbers plus two enum choices — never a code path; `python -m vessel_utils.gui --selftest` checks the pipeline half headlessly. |
| [_vendor/](vessel_utils/_vendor/) | `compute_entropy_grabcut`, `EntropyGrabCutConfig`, `EntropyGrabCutResult` — the tissue masker. **Vendored, not ours: do not edit it here**, re-sync from UCI-XuLab-RegTools per [_vendor/README.md](vessel_utils/_vendor/README.md). A local edit diverges from upstream and invalidates every cached mask under `results/tissue_masks/` |

**Removed, and why it matters if you go looking for it.** `storage`, `chunked` and `correct` (chunked OME-Zarr conversion, halo-sized blockwise filtering, lightsheet destriping and depth-attenuation correction), plus `qc`, `validate`, `ensemble` and `sweep`, were built for whole-brain lightsheet volumes and validation designs that no analysis here ever ran — every one of their callers was its own test. They were deleted rather than left to read as a description of work that happened; `git log` has them if that work restarts, and the design reasoning is preserved in the sections below. That also dropped `zarr`, `dask` and `PyWavelets` from the dependencies. **Do not re-add a module here speculatively** — the last round cost ~2,000 lines and three dependencies carrying nothing.

### Why the active package differs

Three deliberate departures, each addressing a consistency problem in the archive:

- **Bounded response instead of per-image rescale.** `itk.RescaleIntensityImageFilter` stretches each image's own min/max across 0–255, so the archive's `thresh=230` is a *relative* criterion — it means something different in every image, and in particular something different in each of the two channels being compared, which lands directly on precision vs recall as a methodological asymmetry. Jerman's response is bounded in [0,1] by construction. Pass `reference_lambda` (see `max_eigenvalue`) to remove the last per-image dependence in the tau regularisation.
- **Physical scales.** Sigmas are in µm when `spacing` is supplied. The archive runs on raw arrays at implicit 1×1 spacing, which on anisotropic lightsheet data searches different vessel calibres along different axes.
- **A trimmed metric set.** Of the archive's eight metrics, `mean_squared_error` and `hamming` are provably the same number on binary masks, `rand_index` is pixel accuracy (two *unrelated* masks still score ~0.91), and `ssim` on binary masks reflects spatial structure more than agreement — unrelated mask pairs measure anywhere from near 0 for uncorrelated speckle to above 0.8 for realistically structured vasculature. The active package keeps Dice, Jaccard, precision and recall, and adds `cl_dice`, which compares skeletons and so notices connectivity disagreements that voxel overlap misses.

`vessel_utils.metrics.jaccard` is named differently from the archive's `iou` on purpose — it binarises first, so it agrees with the archive on boolean input and is simply correct on numeric input.

### If whole-brain lightsheet work restarts

None of this is in the tree — it was removed with `storage`/`chunked`/`correct`. To get a deleted file back, find the commit that removed it and read the parent: `git log --oneline --diff-filter=D -- vessel_utils/storage.py` then `git show <rev>^:vessel_utils/storage.py` (or `git checkout <rev>^ -- vessel_utils/` for the lot). `git log --` alone lists commits, it does not produce the source. The reasoning below is recorded because each point cost real debugging and none of it is obvious:

- **A raw acquisition is ~2600 planes of ~10k x 10k, ~500 GB per channel.** Convert to a chunked store *before* anything else. One TIFF per plane is the worst layout for 3D work: a 64-voxel-deep column means opening 64 files and decoding 64 full planes. Chunked blocks plus a pyramid fixes it — and coarse levels matter, because tissue masks and depth profiles need no capillary detail.
- **Halos must match the filter's *actual* reach, in physical units.** `scipy.ndimage.gaussian_filter` truncates at **4.0σ**, not the 3σ conventionally quoted. Sizing a halo at 3σ leaves a real seam at every chunk boundary — small in absolute terms, but concentrated exactly where it severs vessels and changes the topology every skeleton-based measure depends on. Anisotropic spacing means a different halo per axis.
- **Never regularise per block.** A per-block `reference_lambda` makes the response mean something different in every chunk, turning seams into genuine discontinuities rather than edge artefacts.
- **Both lightsheet artefacts are channel-asymmetric**, which is why they corrupt a comparison rather than merely degrading it. Stripes are linear structures — exactly what a Hessian filter fires on — so they are false vessels, not background noise, and no method can separate a stripe from a vessel running the full width parallel to the illumination axis. Depth attenuation differs by wavelength (488/561/640), so **correct each channel with its own profile**; sharing one reintroduces the asymmetry. Estimate the profile with a median, not a mean: vessels are a bright minority, and a mean tracks vessel density as much as illumination.

Each notebook opens with a bootstrap cell that walks up from the working directory to find the repo root (by looking for `pyproject.toml`) and adds it to `sys.path`. That is why `pip install -e .` is optional — do not remove the bootstrap, the lab runs these notebooks directly and must not acquire a setup step it can silently skip.

### The spinal-cord scripts share one section loader

[scripts/](scripts/) holds the follow-up spinal-cord analysis. Every script there needs the same preamble — pick the pilot or `--full` path list, parse the filename, check the file really is a two-channel composite, cast to float32, build the tissue mask, skip-with-a-reason anything that fails. That block was copied into eight scripts and drifted. It now lives in [analyse_spinal_cord.py](scripts/analyse_spinal_cord.py), and seven scripts use it:

```python
paths = section_paths(full, slices_per_region=3)     # pilot subset unless full
for s in load_sections(paths):                       # s.virus, s.cd31, s.tissue
    ...                                              # s.label, s.stem, s.counter
```

`section_paths(full=True, ...)` ignores the pilot arguments — `--full` means every section. Use the loader for anything new. Related shared pieces, same reason: `short_reporter`, `virus_cut` (the per-image `median + k*MAD` rule, which takes the already-extracted parenchyma values), `REGION_NAME`, `visualize_sections.contact_sheet`, and `write_csv` for the per-section result CSVs.

**Three loaders remain outside it, deliberately.** If you change `load_sections`, check whether the change belongs in these too — they are what produce two of the three result CSVs:

| loader | why it stays |
| --- | --- |
| `analyse_spinal_cord.analyse()` | takes a path so its own `main()` and the notebooks can drive it per section; writes `spinal_cord_specificity.csv` |
| `dice_between_channels.channels()` / `channel_masks()` | `notebooks/dice_between_channels.ipynb` calls `channel_masks(path)` and `score_section(path)` directly; changing the signature breaks a stored notebook |
| `export_handoff.clean_stem()` + its loop | `clean_stem` omits the `_s0` suffix `Section.stem` always emits, so converting it renames every exported mask TIF its README references. A real inconsistency, just not one worth a rename |

`plot_results.py`, `regional_stats.py` and `summarise_spinal_cord.py` read only CSVs and import none of this — pulling in the analysis module would drag `tifffile` and the vendored GrabCut along for a four-entry dict, so they keep their own `LONG`.

## What lives in the notebooks, not the archive

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
