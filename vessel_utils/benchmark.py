"""Scoring a segmenter against phantoms with known ground truth.

This is the closest thing to an accuracy measurement available without hand
annotation. Everything else in the package compares one channel against another
or one threshold against another; here there is a correct answer.

Two things to keep in view when reading a number from this:

  - It measures performance on the phantom, and the phantom's realism is set by
    the degradation model in `synth`. A pipeline tuned until it wins here has
    been tuned to the simulator.
  - Aggregate Dice hides where the failures are. `by_calibre` exists because the
    interesting question is almost never "how good is it" but "at what vessel
    radius does it stop working", and the answer to the second changes what you
    can claim.
"""

import numpy as np

from . import metrics as metric_module
from .synth import phantom

__all__ = ["score_segmentation", "run_benchmark", "sweep_condition", "summarise"]


def score_segmentation(predicted, truth, spacing=None, calibre_edges=None):
    """Compare one predicted mask against a known-correct one."""
    predicted = np.asarray(predicted).astype(bool)
    truth = np.asarray(truth).astype(bool)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")

    result = {
        "dice": metric_module.dice(predicted, truth),
        "jaccard": metric_module.jaccard(predicted, truth),
        # Argument order fixed deliberately: precision asks what fraction of the
        # prediction is real, recall what fraction of the truth was found.
        "precision": metric_module.precision(predicted, truth),
        "recall": metric_module.recall(predicted, truth),
        "cl_dice": metric_module.cl_dice(predicted, truth),
        "area_fraction_predicted": metric_module.area_fraction(predicted),
        "area_fraction_truth": metric_module.area_fraction(truth),
    }
    if calibre_edges is not None:
        result["by_calibre"] = metric_module.agreement_by_calibre(
            truth, predicted, edges=calibre_edges, spacing=spacing)
    return result


def run_benchmark(segmenter, n_phantoms=5, shape=(48, 96, 96), spacing=(3.0, 0.75, 0.75),
              seed=0, calibre_edges=None, tree_kwargs=None, acquisition_kwargs=None,
              progress=None):
    """Run a segmenter over several independent phantoms.

    Args:
        segmenter: callable taking (volume, spacing) and returning a boolean mask.
        n_phantoms: how many independent trees to average over. Several is
            necessary — a single tree's radius distribution is a lucky draw, and
            calibre is what dominates the score.
        calibre_edges: radius bins in micrometres for the per-calibre breakdown.

    Returns:
        List of per-phantom score dicts, each with its seed.
    """
    rows = []
    for index in range(n_phantoms):
        phantom_seed = seed + index
        volume, truth, _segments = phantom(
            shape=shape, spacing=spacing, seed=phantom_seed,
            tree_kwargs=tree_kwargs, acquisition_kwargs=acquisition_kwargs)
        predicted = segmenter(volume, spacing)
        row = {"seed": phantom_seed}
        row.update(score_segmentation(predicted, truth, spacing, calibre_edges))
        rows.append(row)
        if progress is not None:
            progress(index + 1, n_phantoms)
    return rows


def sweep_condition(segmenter, parameter, values, base=None, n_phantoms=3,
                    shape=(48, 96, 96), spacing=(3.0, 0.75, 0.75), seed=0,
                    **kwargs):
    """Vary one acquisition parameter and watch the score respond.

    This is the attribution tool: hold the segmenter fixed, sweep `gain` or
    `psf_sigma` or `attenuation`, and see which one actually costs recall. If
    performance collapses with photon count rather than with filter choice, the
    useful change is to the microscope, not the code.

    Args:
        parameter: name of a `simulate_acquisition` argument.
        values: values to try.
        base: other acquisition settings held constant.

    Returns:
        List of dicts, each with the parameter value and the mean scores.
    """
    rows = []
    for value in values:
        acquisition = dict(base or {})
        acquisition[parameter] = value
        scores = run_benchmark(segmenter, n_phantoms=n_phantoms, shape=shape,
                           spacing=spacing, seed=seed,
                           acquisition_kwargs=acquisition, **kwargs)
        summary = summarise(scores)
        rows.append({parameter: value, **summary})
    return rows


def summarise(rows, keys=("dice", "jaccard", "precision", "recall", "cl_dice")):
    """Mean and spread across phantoms, with a bootstrap interval on the mean.

    The interval reflects variation between phantoms only. It says nothing about
    how well the phantom stands in for a real brain, which is the larger and
    unquantified uncertainty.
    """
    if not rows:
        raise ValueError("no rows to summarise")
    rng = np.random.default_rng(0)
    summary = {"n_phantoms": len(rows)}

    for key in keys:
        values = np.array([row[key] for row in rows if key in row], dtype=float)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if values.size > 1:
            draws = rng.choice(values, size=(2000, values.size), replace=True)
            means = draws.mean(axis=1)
            summary[f"{key}_ci_low"] = float(np.percentile(means, 2.5))
            summary[f"{key}_ci_high"] = float(np.percentile(means, 97.5))
    return summary
