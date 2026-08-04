# notebooks/

Interactive analysis notebooks for the **follow-up** spinal-cord work (the BEC
spinal cords in `Z:\Lab\Eric V\BEC Spinal Cords\composites_EV`). They wrap the
functions in [`scripts/`](../scripts) with markdown that explains each metric, so
the method is readable alongside the numbers.

These are **not** part of the frozen `velazquez_rivera_2025` archive. The archive
notebooks live in `process_brain_slices/`, `process_thickness/` and
`process_lightsheet_full_brain/`, are pinned to the published implementation, and
are covered by `tests/test_notebooks.py` (which only globs `process_*/`). Nothing
here touches them.

| notebook | what it covers |
| --- | --- |
| [dice_between_channels.ipynb](dice_between_channels.ipynb) | Segment vessels in **both** channels and score overlap — Dice, Jaccard, specificity (precision), coverage (recall). The "paper-1 style" comparison, with an explanation of why segmenting the virus channel flatters the vector. |
| [enrichment_metrics.ipynb](enrichment_metrics.ipynb) | The **enrichment** measures — threshold-free CD31-percentile enrichment (primary) plus the mask-based enrichment / coverage / off-target — with the over-segmentation caveat spelled out. |

## Running them

Each notebook bootstraps `sys.path` by walking up to the repo root (the folder
with `pyproject.toml`), so a plain clone runs them with no install step — the same
pattern the archive notebooks use. They read `Z:` **read-only** and are executed
here on a small **pilot** subset (both reporters, every region, one slice each);
set `RUN_FULL = True` in the config cell to run every section. Nothing is written
outside the (gitignored) `results/`.

They import the merged `scripts/` functions rather than re-implementing anything,
so the numbers match the command-line tools exactly.
