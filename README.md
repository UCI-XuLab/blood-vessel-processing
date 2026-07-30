# blood-vessel-processing

[![DOI](https://zenodo.org/badge/921053214.svg)](https://doi.org/10.5281/zenodo.14876892)

Blood vessel processing code for quantifying sample brain slices

For the paper: Specific targeting of brain endothelial cells using enhancer AAV vectors

## Layout

The image processing routines live in `vessel_utils/`, split by pipeline stage:

| module | contents |
| --- | --- |
| `io` | reading TIFF slices and unpacking their channels |
| `enhance` | contrast, gamma, histogram equalization, N4 bias correction |
| `vessels` | Hessian vesselness detection, mask post-processing, brain masking |
| `metrics` | agreement metrics between two binary masks |
| `viz` | inline previews and figure export |

The analysis itself is stored as Jupyter notebooks, whose contents can be run and modified for any set of images. Each notebook is a workspace for one specimen, holding that specimen's parameters, batch loop, and stored outputs; the shared routines they call are identical, so a change in `vessel_utils` reaches every notebook at once.

- `process_brain_slices/` — vessel segmentation per slice, and agreement between the two channels
- `process_thickness/` — vessel caliber and length via medial-axis skeletonization
- `process_lightsheet_full_brain/` — whole-brain lightsheet: condense the z-stack, then segment

## Setup

```bash
pip install -e .
jupyter lab
```

The install is optional — each notebook locates the repository root and adds it to `sys.path` on its own, so the notebooks also run from a plain checkout.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite reconstructs the helper functions as they were defined inline in the notebooks before they were extracted, and requires the shared versions to produce bit-identical results.
