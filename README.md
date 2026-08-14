# blood-vessel-processing

[![DOI](https://zenodo.org/badge/921053214.svg)](https://doi.org/10.5281/zenodo.14876892)

Blood vessel processing code for quantifying sample brain slices

For the paper: Specific targeting of brain endothelial cells using enhancer AAV vectors

## Layout

### `velazquez_rivera_2025/` — the published implementation

The routines that produced the figures in the paper, split by pipeline stage:

| module | contents |
| --- | --- |
| `io` | reading TIFF slices and unpacking their channels |
| `enhance` | contrast, gamma, histogram equalization, N4 bias correction |
| `vessels` | Hessian vesselness detection, mask post-processing, brain masking |
| `metrics` | agreement metrics between two binary masks |
| `viz` | inline previews and figure export |

**This package is frozen.** It is the reproducible record of the published analysis, and the test suite fails if any of it changes. Improvements go in `vessel_utils/`.

### `vessel_utils/` — active development

Where follow-up work happens. It is free to differ from the archive:

| module | contents |
| --- | --- |
| `vesselness` | Jerman vesselness — physical scales, 2D and 3D, response bounded in [0,1] |
| `threshold` | hysteresis thresholding and mask clean-up |
| `metrics` | Dice, Jaccard, precision, recall, clDice, area fractions, agreement by vessel calibre |
| `synth` | synthetic vasculature with a simulated lightsheet acquisition |
| `benchmark` | scoring a segmenter against phantoms with known ground truth |

`vessel_utils/_vendor/` holds the entropy-guided GrabCut tissue masker, copied from UCI-XuLab-RegTools. It is *not* part of the above — it is a vendored copy, and unlike the rest of this package it must **not** be edited locally: re-sync it from upstream instead, per [`_vendor/README.md`](vessel_utils/_vendor/README.md). Editing it in place silently diverges from RegTools and invalidates every cached tissue mask.

The main differences: sigmas are in physical units rather than voxels, so anisotropic data is handled correctly; the vesselness response has an absolute scale rather than being rescaled per image, so one threshold means the same thing in both channels and across slices; and the metric set drops four measures that carry little information on binary masks in favour of `clDice`, which compares centrelines and so registers connectivity disagreements that voxel overlap misses.

### Notebooks

The analysis itself is stored as Jupyter notebooks, whose contents can be run and modified for any set of images. Each notebook is a workspace for one specimen, holding that specimen's parameters, batch loop, and stored outputs. They import from `velazquez_rivera_2025` and stay there, so they keep reproducing the published figures.

- `process_brain_slices/` — vessel segmentation per slice, and agreement between the two channels
- `process_thickness/` — vessel caliber and length via medial-axis skeletonization
- `process_lightsheet_full_brain/` — whole-brain lightsheet: condense the z-stack, then segment

## Setup

```bash
pip install -e ".[dev]"
jupyter lab
```

`[dev]` is what the notebooks need — `pip install -e .` alone covers `import vessel_utils` but not the notebook stack (`tqdm`, `pandas`, `jupyterlab`), which no module imports.

The install is optional — each notebook locates the repository root and adds it to `sys.path` on its own, so the notebooks also run from a plain checkout.

## Benchmarks

```bash
python scripts/benchmark_pipeline.py
```

Reproduces every accuracy figure quoted for the active pipeline, against phantom
ground truth. Deterministic, so a change in the numbers means a change in the
pipeline.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

384 tests. The suite reconstructs the helper functions as they were defined inline in the notebooks before they were extracted, and requires `velazquez_rivera_2025` to produce bit-identical results; it also hashes that package against a manifest so accidental edits to the published implementation fail loudly. The active package is checked against the properties it claims rather than against history.
