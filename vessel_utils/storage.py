"""Chunked storage for volumes too large to hold in memory.

A whole-brain lightsheet acquisition here is roughly 2600 planes of 10k x 10k,
about 500 GB per channel. Nothing downstream can open that as an array, and a
directory of one TIFF per plane is the worst possible layout for 3D work: reading
a 64-voxel-deep column means touching 64 separate files and decoding 64 full
planes to keep a few hundred pixels from each.

Converting once to a chunked store fixes that. A chunk of (64, 512, 512) is a
compact 3D block on disk, so a filter that needs a neighbourhood in z reads one
block instead of 64 files. That single change is what makes 3D segmentation
possible at this scale; it is not an optimisation.

The layout follows OME-NGFF: a multiscale group whose datasets are successive
2x downsamplings, with physical voxel size recorded in the metadata. The coarse
levels exist so you can look at the whole brain interactively without reading the
full-resolution data, and so that tissue masks and depth profiles — which do not
need capillary detail — can be computed cheaply.
"""

import json
import math
from pathlib import Path

import numpy as np
import zarr

__all__ = ["plane_series_to_zarr", "open_volume", "write_volume", "pyramid_levels",
           "read_spacing"]

DEFAULT_CHUNKS = (64, 512, 512)


def _sorted_planes(pattern):
    import glob
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(f"no files matched {pattern!r}")
    return paths


def _multiscale_metadata(n_levels, spacing, axes=("z", "y", "x")):
    """OME-NGFF multiscales metadata, with each level's physical voxel size."""
    datasets = []
    for level in range(n_levels):
        factor = 2 ** level
        # Only y and x are downsampled; z stays put because it is already the
        # coarsest axis and halving it destroys capillary continuity.
        scale = [float(spacing[0]), float(spacing[1] * factor), float(spacing[2] * factor)]
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })
    return [{
        "version": "0.4",
        "name": "volume",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in axes],
        "datasets": datasets,
    }]


def pyramid_levels(shape, min_size=512):
    """How many 2x downsamplings before the xy plane drops below `min_size`."""
    smallest = min(shape[1], shape[2])
    if smallest <= min_size:
        return 1
    return int(math.floor(math.log2(smallest / min_size))) + 1


def plane_series_to_zarr(pattern, store_path, spacing, chunks=DEFAULT_CHUNKS,
                         n_levels=None, dtype=None, compressor="default",
                         progress=None):
    """Convert a directory of one-TIFF-per-plane into a chunked multiscale store.

    Planes are read one at a time and written straight into the store, so peak
    memory is one chunk-row rather than the volume. A 500 GB channel converts
    without ever holding more than a few GB.

    Args:
        pattern: glob for the plane series, e.g. "raw/ch0/*_C0.tif". Sorted
            lexicographically, so zero-padded plane numbers are required — an
            unpadded series orders Z10 before Z9 and silently scrambles the volume.
            `plane_series_to_zarr` checks for this and refuses.
        store_path: output .zarr directory.
        spacing: (z, y, x) voxel size in micrometres. Recorded in the metadata and
            relied on by every physical-units operation downstream.
        chunks: chunk shape. The default is a compromise: deep enough in z for a
            3D Hessian neighbourhood, small enough in xy that a worker holds
            several chunks at once.
        n_levels: pyramid depth. None picks enough levels to reach ~512 px.
        progress: optional callable taking (plane_index, total).

    Returns:
        Path to the created store.
    """
    import tifffile

    paths = _sorted_planes(pattern)
    _reject_unpadded_ordering(paths)

    first = tifffile.imread(paths[0])
    if first.ndim != 2:
        raise ValueError(f"expected 2D planes, got shape {first.shape} from {paths[0]}")
    shape = (len(paths), *first.shape)
    dtype = np.dtype(dtype) if dtype is not None else first.dtype
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"spacing must be three positive values, got {spacing}")

    if n_levels is None:
        n_levels = pyramid_levels(shape)

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="w")

    full = root.create_array(
        "0", shape=shape, chunks=tuple(chunks), dtype=dtype,
        compressors=compressor if compressor != "default" else None,
    )

    for index, path in enumerate(paths):
        plane = tifffile.imread(path)
        if plane.shape != shape[1:]:
            raise ValueError(
                f"plane {index} ({path}) has shape {plane.shape}, expected {shape[1:]}; "
                f"the series is not a single consistent volume"
            )
        full[index] = plane.astype(dtype, copy=False)
        if progress is not None:
            progress(index + 1, len(paths))

    _build_pyramid(root, n_levels, chunks, dtype)

    root.attrs["multiscales"] = _multiscale_metadata(n_levels, spacing)
    root.attrs["source_pattern"] = str(pattern)
    root.attrs["n_planes"] = len(paths)
    return store_path


def _reject_unpadded_ordering(paths):
    """Refuse a series whose lexicographic order is not its numeric order.

    `sorted()` puts Z1000 before Z999. On a 2607-plane acquisition that silently
    interleaves the volume, and the result looks plausible enough in a single
    plane view to go unnoticed.
    """
    import re
    numbers = []
    for path in paths:
        found = re.findall(r"Z(\d+)", Path(path).name)
        if not found:
            return  # no recognisable plane index; nothing to check
        numbers.append(int(found[-1]))
    if numbers != sorted(numbers):
        raise ValueError(
            "plane files do not sort into numeric order — the Z index is not "
            "zero-padded, so sorted() would scramble the volume. Rename the "
            "series with a fixed-width index before converting."
        )


def _build_pyramid(root, n_levels, chunks, dtype):
    """Successive 2x downsamplings in xy, computed chunk-row by chunk-row."""
    for level in range(1, n_levels):
        source = root[str(level - 1)]
        shape = (source.shape[0], source.shape[1] // 2, source.shape[2] // 2)
        target = root.create_array(
            str(level), shape=shape,
            chunks=tuple(min(c, s) for c, s in zip(chunks, shape)), dtype=dtype,
        )
        step = chunks[0]
        for start in range(0, shape[0], step):
            stop = min(start + step, shape[0])
            block = source[start:stop, :shape[1] * 2, :shape[2] * 2]
            # Mean of each 2x2 xy neighbourhood; z is left alone.
            reduced = block.reshape(block.shape[0], shape[1], 2, shape[2], 2)
            target[start:stop] = reduced.mean(axis=(2, 4)).astype(dtype, copy=False)


def open_volume(store_path, level=0):
    """Open one pyramid level as a dask array, with its physical spacing.

    Returns:
        (array, spacing) where spacing is the (z, y, x) voxel size in micrometres
        *for that level* — coarse levels have larger xy spacing, and every
        physical-units function downstream needs the level's own value, not the
        full-resolution one.
    """
    import dask.array as da

    root = zarr.open_group(str(store_path), mode="r")
    array = da.from_zarr(root[str(level)])
    return array, read_spacing(store_path, level)


def read_spacing(store_path, level=0):
    """Physical voxel size (z, y, x) in micrometres for a pyramid level."""
    root = zarr.open_group(str(store_path), mode="r")
    multiscales = root.attrs["multiscales"]
    if isinstance(multiscales, str):
        multiscales = json.loads(multiscales)
    datasets = multiscales[0]["datasets"]
    for dataset in datasets:
        if dataset["path"] == str(level):
            return tuple(dataset["coordinateTransformations"][0]["scale"])
    raise KeyError(f"level {level} not found in {store_path}")


def write_volume(array, store_path, spacing, chunks=DEFAULT_CHUNKS, n_levels=1):
    """Write an in-memory or dask array to a store, preserving spacing metadata.

    Used for intermediate products — corrected volumes, vesselness responses,
    masks — so that each stage reads and writes the same layout and nothing has
    to be held whole.
    """
    import dask.array as da

    store_path = Path(store_path)
    root = zarr.open_group(str(store_path), mode="w")
    chunks = tuple(min(c, s) for c, s in zip(chunks, array.shape))

    if isinstance(array, da.Array):
        array.rechunk(chunks).to_zarr(str(store_path), component="0",
                                      overwrite=True)
        root = zarr.open_group(str(store_path), mode="a")
    else:
        target = root.create_array("0", shape=array.shape, chunks=chunks,
                                   dtype=array.dtype)
        target[:] = array

    root.attrs["multiscales"] = _multiscale_metadata(n_levels, spacing)
    return store_path
