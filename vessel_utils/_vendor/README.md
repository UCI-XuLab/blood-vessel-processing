# Vendored modules

Third-party code copied into this repo rather than depended on as a package.

## `grabcut.py` + `masking_thresholds.py` — entropy-guided GrabCut tissue masker

- **Source:** [UCI-XuLab-RegTools](https://github.com/UCI-XuLab/UCI-XuLab-RegTools),
  `regtools/tissue_masking/core/grabcut.py` and `regtools/utils/masking_thresholds.py`.
- **Pinned commit:** `9f2e5fb` (the commit that made seeded GrabCut the default
  foreground masker there, and added the vendoring note this copy follows).
- **Why vendored:** the follow-up spinal-cord pipeline uses it as the tissue mask
  (`scripts/analyse_spinal_cord.py::tissue_mask`). RegTools is a large GUI/registration
  package; vendoring these two self-contained files avoids depending on all of it.
- **Runtime deps:** `numpy`, `opencv-python` (`cv2`), `scipy`, `scikit-image` — all
  already in this project's `pyproject.toml`.

### The two files ship together

`grabcut.py` imports `BACKGROUND_PERCENTILE` and `_percentile_rank` from
`masking_thresholds.py`. Upstream keeps that constant shared so the GrabCut seed and
the 2D-registration foreground guard cannot drift apart; copying only one file would
silently break that guarantee. Keep them paired.

### Local modifications (the only edits to the copied files)

1. `grabcut.py`: the import was rewritten from
   `from regtools.utils.masking_thresholds import ...` to
   `from .masking_thresholds import ...` (relative, since `regtools` is not installed here).
2. A three-line provenance header at the top of each file.

Nothing else is changed. The default `EntropyGrabCutConfig()` is the upstream
benchmark configuration (percentile 1–99.5 normalize + Multi-Otsu-4 seed).

### Re-syncing to a newer RegTools

Re-copy both files from the source repo, then re-apply the two modifications above
(the relative import and the provenance headers). Bump the pinned commit here and, if
the masker's output changes, delete `results/tissue_masks/` so cached masks recompute.
