"""Feature calculations for integer-valued histology label maps."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
import skimage.io
import skimage.measure
import skimage.morphology
from PIL import Image as PILImage

DEFAULT_LABELS = {
    "lumen": 1,
    "epithelium": 2,
    "stroma": 4,
    "epithelial_cells": 5,
    "stromal_cells": 6,
}
DEFAULT_MIN_AREA = 2048
MAX_DEFAULT_JOBS = 12
PARALLEL_REGION_AREA_THRESHOLD = 100_000
PARALLEL_LABEL_PIXEL_THRESHOLD = 4_000_000
PARALLEL_HISTOGRAM_MEMORY_BYTES = 128 * 1024 * 1024
LABEL_CHUNK_ROWS = 512
MAT_CHUNK_BLOCK_ROWS = 32
PARALLEL_CHUNKS_PER_WORKER = 2


def _default_job_count() -> int:
    return min(MAX_DEFAULT_JOBS, os.cpu_count() or 1)


def _make_tissue_mask(
    task: tuple[np.ndarray, str, Mapping[str, int]],
) -> tuple[str, np.ndarray]:
    palette, name, labels = task
    if name == "lumen":
        mask = palette == labels["lumen"]
    elif name == "epithelial_cells":
        mask = palette == labels["epithelial_cells"]
    elif name == "epithelium":
        mask = (palette == labels["epithelium"]) | (
            palette == labels["epithelial_cells"]
        )
    else:
        mask = (palette == labels["stroma"]) | (palette == labels["stromal_cells"])
    return name, mask


def _make_tissue_mask_chunk(
    task: tuple[np.ndarray, np.ndarray, str, Mapping[str, int]],
) -> None:
    palette, output, name, labels = task
    output[:] = _make_tissue_mask((palette, name, labels))[1]


def _create_tissue_masks(
    palette: np.ndarray,
    labels: Mapping[str, int],
    n_jobs: int | None,
    names: tuple[str, ...] = (
        "lumen",
        "epithelium",
        "stroma",
        "epithelial_cells",
    ),
) -> dict[str, np.ndarray]:
    tasks = [(palette, name, labels) for name in names]
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers > 1 and palette.size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        outputs = {name: np.empty(palette.shape, dtype=np.bool_) for name in names}
        chunk_count = min(workers * PARALLEL_CHUNKS_PER_WORKER, palette.shape[0])
        row_bounds = np.linspace(0, palette.shape[0], chunk_count + 1, dtype=np.int64)
        chunk_tasks = [
            (
                palette[start:stop],
                outputs[name][start:stop],
                name,
                labels,
            )
            for name in names
            for start, stop in pairwise(row_bounds)
        ]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk_tasks))) as pool:
            list(pool.map(_make_tissue_mask_chunk, chunk_tasks))
        return outputs
    return dict(map(_make_tissue_mask, tasks))


def _summify_item(
    task: tuple[str, np.ndarray, tuple[int, int]],
) -> tuple[str, np.ndarray]:
    name, mask, block_size = task
    return name, summify(mask, block_size)


def _summify_density_chunk(
    task: tuple[str, np.ndarray, tuple[int, int], int],
) -> tuple[str, int, np.ndarray]:
    name, mask, block_size, block_start = task
    return name, block_start, summify(mask, block_size)


def _calculate_densities(
    masks: Mapping[str, np.ndarray],
    block_size: tuple[int, int],
    n_jobs: int | None,
) -> dict[str, np.ndarray]:
    tasks = [(f"{name}_density", mask, block_size) for name, mask in masks.items()]
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers > 1 and tasks[0][1].size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        block_height = block_size[0]
        block_rows = (tasks[0][1].shape[0] + block_height - 1) // block_height
        chunk_count = min(workers * PARALLEL_CHUNKS_PER_WORKER, block_rows)
        block_bounds = np.linspace(0, block_rows, chunk_count + 1, dtype=np.int64)
        output_shape = (
            block_rows,
            (tasks[0][1].shape[1] + block_size[1] - 1) // block_size[1],
        )
        outputs = {
            name: np.empty(output_shape, dtype=np.float32) for name, _, _ in tasks
        }
        chunk_tasks = []
        for name, mask, size in tasks:
            for block_start, block_stop in pairwise(block_bounds):
                row_start = int(block_start * block_height)
                row_stop = min(int(block_stop * block_height), mask.shape[0])
                chunk_tasks.append(
                    (name, mask[row_start:row_stop], size, int(block_start))
                )
        with ThreadPoolExecutor(max_workers=min(workers, len(chunk_tasks))) as pool:
            for name, block_start, density_chunk in pool.map(
                _summify_density_chunk, chunk_tasks
            ):
                outputs[name][block_start : block_start + density_chunk.shape[0]] = (
                    density_chunk
                )
        return outputs
    return dict(map(_summify_item, tasks))


def _label_row_chunk(task: tuple[np.ndarray, np.ndarray]) -> int:
    mask, output = task
    count = scipy.ndimage.label(mask, output=output)
    return cast(int, count)


def _find_root(parent: np.ndarray, label_id: int) -> int:
    while parent[label_id] != label_id:
        parent[label_id] = parent[parent[label_id]]
        label_id = int(parent[label_id])
    return label_id


def _compact_component_labels(parent: np.ndarray) -> np.ndarray:
    while True:
        compressed = parent[parent]
        if np.array_equal(compressed, parent):
            break
        parent = compressed
    roots = parent == np.arange(parent.size)
    roots[0] = False
    compacted = np.cumsum(roots, dtype=np.int32)
    compacted[0] = 0
    return compacted[parent]


def _remap_label_chunk(task: tuple[np.ndarray, np.ndarray]) -> None:
    labels, lookup = task
    labels[:] = lookup[labels]


def _remap_compacted_label_chunk(
    task: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    source, output, lookup = task
    output[:] = lookup[source]


def _count_label_chunk(task: tuple[np.ndarray, int]) -> np.ndarray:
    labels, bin_count = task
    return np.bincount(labels.ravel(), minlength=bin_count)


def _label_components(mask: np.ndarray, n_jobs: int | None = None) -> np.ndarray:
    """Label a large mask in parallel row strips, then merge strip-edge regions."""
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if (
        workers <= 1
        or mask.size < PARALLEL_LABEL_PIXEL_THRESHOLD
        or mask.shape[0] <= LABEL_CHUNK_ROWS
    ):
        return cast(tuple[np.ndarray, int], scipy.ndimage.label(mask))[0]

    output = np.empty(mask.shape, dtype=np.int32)
    bounds = [
        (start, min(start + LABEL_CHUNK_ROWS, mask.shape[0]))
        for start in range(0, mask.shape[0], LABEL_CHUNK_ROWS)
    ]
    tasks = [(mask[start:stop], output[start:stop]) for start, stop in bounds]
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        counts = list(pool.map(_label_row_chunk, tasks))

    offset = 0
    for (start, stop), count in zip(bounds, counts, strict=True):
        if offset:
            tile = output[start:stop]
            np.add(tile, offset, out=tile, where=tile > 0)
        offset += count
    total_labels = int(sum(counts))
    parent = np.arange(total_labels + 1, dtype=np.int32)

    for (_, stop), (next_start, _) in pairwise(bounds):
        upper = output[stop - 1]
        lower = output[next_start]
        overlaps = (upper > 0) & (lower > 0)
        if not np.any(overlaps):
            continue
        pairs = np.unique(np.column_stack((upper[overlaps], lower[overlaps])), axis=0)
        for upper_id, lower_id in pairs.tolist():
            upper_root = _find_root(parent, upper_id)
            lower_root = _find_root(parent, lower_id)
            if upper_root != lower_root:
                child = max(upper_root, lower_root)
                parent[child] = min(upper_root, lower_root)

    if np.all(parent == np.arange(parent.size)):
        return output
    lookup = _compact_component_labels(parent)
    remap_tasks = [(output[start:stop], lookup) for start, stop in bounds]
    with ThreadPoolExecutor(max_workers=min(workers, len(remap_tasks))) as pool:
        list(pool.map(_remap_label_chunk, remap_tasks))
    return output


def load_label_map(path: str | Path, n_jobs: int | None = None) -> np.ndarray:
    """Read a label map, using threaded Glymur/OpenJPEG decoding for JPEG2000."""
    path = Path(path)
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers < 1:
        raise ValueError("n_jobs must be positive")
    if path.suffix.lower() in {".jp2", ".j2k", ".j2c"}:
        try:
            import glymur
        except ModuleNotFoundError as exc:
            if exc.name != "glymur":
                raise
        else:
            previous_threads = glymur.get_option("lib.num_threads")
            try:
                glymur.set_option("lib.num_threads", workers)
                image = np.asarray(glymur.Jp2k(path)[:])
            finally:
                glymur.set_option("lib.num_threads", previous_threads)
            return _validate_label_map(image)
    max_image_pixels = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = None
    try:
        image = np.asarray(skimage.io.imread(str(path)))
    finally:
        PILImage.MAX_IMAGE_PIXELS = max_image_pixels
    return _validate_label_map(image)


def _validate_label_map(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[-1] not in (3, 4) or not np.all(
            image[..., :3] == image[..., :1]
        ):
            raise ValueError(
                "Expected a single-channel integer label map, not a color image"
            )
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected integer label values, got dtype {image.dtype}")
    return image


def summify(image: np.ndarray, block_size: tuple[int, int] = (20, 20)) -> np.ndarray:
    """Sum image pixels in padded, non-overlapping blocks."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("summify expects a 2-D image")
    if not arr.size:
        return np.empty((0, 0), dtype=np.float32)
    bh, bw = block_size
    if bh <= 0 or bw <= 0:
        raise ValueError("block_size values must be positive")
    columns = np.add.reduceat(
        arr, np.arange(0, arr.shape[1], bw), axis=1, dtype=np.float32
    )
    return np.add.reduceat(
        columns, np.arange(0, arr.shape[0], bh), axis=0, dtype=np.float32
    )


def calculate_tortuosity(area: float, perimeter: float) -> float:
    """Return circularity (1 for an ideal circle).

    skimage's pixel-based perimeter underestimates small objects, which can push
    the raw ratio above 1, so the result is clipped to [0, 1].
    """
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float(min(1.0, 4 * np.pi * area / perimeter**2))


def average_thickness(region_mask: np.ndarray) -> float:
    area = int(np.sum(region_mask))
    if area == 0:
        return 0.0
    length = int(np.sum(skimage.morphology.skeletonize(region_mask)))
    return float(area / length) if length else 0.0


def _thickness_of_mask(mask: np.ndarray) -> float:
    return average_thickness(mask)


def _compact_regions(
    labeled: np.ndarray, min_area: int, n_jobs: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not labeled.size:
        return (
            np.zeros_like(labeled, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    workers = n_jobs if n_jobs is not None else _default_job_count()
    bin_count = int(labeled.max()) + 1
    max_histogram_workers = PARALLEL_HISTOGRAM_MEMORY_BYTES // (
        bin_count * np.dtype(np.int64).itemsize
    )
    histogram_workers = min(workers, max_histogram_workers, labeled.shape[0])
    if histogram_workers > 1 and labeled.size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        row_step = (labeled.shape[0] + histogram_workers - 1) // histogram_workers
        tasks = [
            (labeled[start : start + row_step], bin_count)
            for start in range(0, labeled.shape[0], row_step)
        ]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            partial_areas = pool.map(_count_label_chunk, tasks)
            areas = np.zeros(bin_count, dtype=np.int64)
            for partial in partial_areas:
                areas += partial
    else:
        areas = np.bincount(labeled.ravel(), minlength=bin_count)
    original_ids = np.flatnonzero(areas >= min_area)
    original_ids = original_ids[original_ids > 0]
    selected_areas = areas[original_ids]
    if not original_ids.size:
        return labeled, original_ids, selected_areas
    lookup = np.zeros(len(areas), dtype=np.int32)
    lookup[original_ids] = np.arange(1, len(original_ids) + 1, dtype=np.int32)
    if workers > 1 and labeled.size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        output = np.empty(labeled.shape, dtype=np.int32)
        bounds = [
            (start, min(start + LABEL_CHUNK_ROWS, labeled.shape[0]))
            for start in range(0, labeled.shape[0], LABEL_CHUNK_ROWS)
        ]
        tasks = [
            (labeled[start:stop], output[start:stop], lookup) for start, stop in bounds
        ]
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            list(pool.map(_remap_compacted_label_chunk, tasks))
        return output, original_ids, selected_areas
    return lookup[labeled], original_ids, selected_areas


def _measure_lumen_region(
    task: tuple[int, np.ndarray],
) -> tuple[int, int, float]:
    label_id, mask = task
    return label_id, int(np.sum(mask)), float(skimage.measure.perimeter(mask))


def _measure_epithelium_region(
    task: tuple[int, np.ndarray, np.ndarray],
) -> tuple[int, int, float, float, float]:
    label_id, mask, cell_values = task
    area = int(np.sum(mask))
    return (
        label_id,
        area,
        float(skimage.measure.perimeter(mask)),
        average_thickness(mask),
        float(np.mean(cell_values)),
    )


def _region_slices(
    labeled: np.ndarray,
) -> list[tuple[slice, ...] | None]:
    return scipy.ndimage.find_objects(labeled)


def _get_region_slice(
    slices: list[tuple[slice, ...] | None], index: int
) -> tuple[slice, ...]:
    bounds = slices[index]
    if bounds is None:
        raise RuntimeError(f"Missing bounding box for region {index + 1}")
    return bounds


def _parallel_thickness(
    labeled: np.ndarray,
    label_ids: np.ndarray,
    n_jobs: int | None,
    total_area: int,
) -> np.ndarray:
    """Calculate thickness on bounding-box crops, using threads for large workloads."""
    if not label_ids.size:
        return np.empty(0, dtype=np.float64)
    workers = n_jobs if n_jobs is not None else _default_job_count()
    slices = _region_slices(labeled)
    if (
        workers <= 1
        or len(label_ids) < 2
        or total_area < PARALLEL_REGION_AREA_THRESHOLD
    ):
        return np.array(
            [
                _thickness_of_mask(labeled[_get_region_slice(slices, i - 1)] == i)
                for i in label_ids
            ],
            dtype=np.float64,
        )
    masks = [labeled[_get_region_slice(slices, i - 1)] == i for i in label_ids]
    with ThreadPoolExecutor(max_workers=min(workers, len(masks))) as pool:
        return np.fromiter(
            pool.map(_thickness_of_mask, masks),
            dtype=np.float64,
            count=len(masks),
        )


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def calculate_lumen_features(
    labeled_lumen: np.ndarray,
    min_area: int = DEFAULT_MIN_AREA,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    columns = ["label", "area", "roundness"]
    if not np.any(labeled_lumen):
        return _empty(columns)
    compact, original_ids, areas = _compact_regions(labeled_lumen, min_area, n_jobs)
    if not original_ids.size:
        return _empty(columns)
    total_area = int(np.sum(areas))
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if (
        workers > 1
        and len(original_ids) > 1
        and total_area >= PARALLEL_REGION_AREA_THRESHOLD
    ):
        slices = _region_slices(compact)
        tasks = [
            (
                int(label_id),
                compact[_get_region_slice(slices, index)] == index + 1,
            )
            for index, label_id in enumerate(original_ids)
        ]
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            rows = list(pool.map(_measure_lumen_region, tasks))
        frame = pd.DataFrame(rows, columns=["label", "area", "perimeter"])
        frame["area"] = frame["area"].astype(np.float64)
        frame["roundness"] = [
            calculate_tortuosity(a, p)
            for a, p in zip(frame.area, frame.perimeter, strict=False)
        ]
        return frame[columns].reset_index(drop=True)
    props = skimage.measure.regionprops_table(
        compact, properties=("label", "area", "perimeter")
    )
    frame = pd.DataFrame(props)
    frame["label"] = original_ids[frame["label"].to_numpy(dtype=np.int64) - 1]
    frame["roundness"] = [
        calculate_tortuosity(a, p)
        for a, p in zip(frame.area, frame.perimeter, strict=False)
    ]
    frame = frame.loc[frame.area >= min_area, columns]
    return frame.reset_index(drop=True)


def calculate_epithelium_features(
    labeled_epithelium: np.ndarray,
    epithelial_cells: np.ndarray,
    min_area: int = DEFAULT_MIN_AREA,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    columns = ["label", "area", "roundness", "average_thickness", "cell_fraction"]
    if not np.any(labeled_epithelium):
        return _empty(columns)
    compact, original_ids, areas = _compact_regions(
        labeled_epithelium, min_area, n_jobs
    )
    if not original_ids.size:
        return _empty(columns)
    total_area = int(np.sum(areas))
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if (
        workers > 1
        and len(original_ids) > 1
        and total_area >= PARALLEL_REGION_AREA_THRESHOLD
    ):
        slices = _region_slices(compact)
        tasks = []
        for index, label_id in enumerate(original_ids):
            bounds = _get_region_slice(slices, index)
            mask = compact[bounds] == index + 1
            tasks.append((int(label_id), mask, epithelial_cells[bounds][mask]))
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            rows = list(pool.map(_measure_epithelium_region, tasks))
        frame = pd.DataFrame(
            rows,
            columns=[
                "label",
                "area",
                "perimeter",
                "average_thickness",
                "cell_fraction",
            ],
        )
        frame["area"] = frame["area"].astype(np.float64)
        frame["roundness"] = [
            calculate_tortuosity(a, p)
            for a, p in zip(frame.area, frame.perimeter, strict=False)
        ]
        return frame[columns].reset_index(drop=True)
    props = skimage.measure.regionprops_table(
        compact,
        intensity_image=epithelial_cells,
        properties=("label", "area", "perimeter", "intensity_mean"),
    )
    frame = pd.DataFrame(props)
    frame["label"] = original_ids[frame["label"].to_numpy(dtype=np.int64) - 1]
    frame["average_thickness"] = _parallel_thickness(
        compact,
        np.arange(1, len(original_ids) + 1),
        n_jobs,
        total_area,
    )
    frame["roundness"] = [
        calculate_tortuosity(a, p)
        for a, p in zip(frame.area, frame.perimeter, strict=False)
    ]
    frame["cell_fraction"] = frame["intensity_mean"]
    frame = frame[columns]
    return frame.reset_index(drop=True)


def extract_features(
    label_map: np.ndarray,
    labels: Mapping[str, int] | None = None,
    min_area: int = DEFAULT_MIN_AREA,
    block_size: tuple[int, int] = (20, 20),
    n_jobs: int | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Calculate density maps and per-object lumen/epithelium measurements.

    Epithelium includes both ``epithelium`` and ``epithelial_cells`` labels;
    stromal cells are included in the stroma density. All feature areas and
    thresholds are in source-map pixels.
    """
    palette = np.asarray(label_map)
    if palette.ndim != 2 or not np.issubdtype(palette.dtype, np.integer):
        raise ValueError("label_map must be a 2-D integer array")
    label_values = dict(DEFAULT_LABELS)
    if labels:
        unknown = set(labels) - set(label_values)
        if unknown:
            raise ValueError(f"Unknown label names: {', '.join(sorted(unknown))}")
        label_values.update(labels)
    if len(set(label_values.values())) != len(label_values):
        raise ValueError("label values must be distinct")
    masks = _create_tissue_masks(palette, label_values, n_jobs)
    densities = _calculate_densities(masks, block_size, n_jobs)
    labeled_lumen = _label_components(masks["lumen"], n_jobs)
    labeled_epithelium = _label_components(masks["epithelium"], n_jobs)
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers > 1 and palette.size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        lumen_workers = workers // 2
        epithelium_workers = workers - lumen_workers
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = (
                pool.submit(
                    calculate_lumen_features,
                    labeled_lumen,
                    min_area,
                    lumen_workers,
                ),
                pool.submit(
                    calculate_epithelium_features,
                    labeled_epithelium,
                    masks["epithelial_cells"],
                    min_area,
                    epithelium_workers,
                ),
            )
            wait(futures)
            errors = [future.exception() for future in futures]
            errors = [error for error in errors if error is not None]
            if errors:
                raise errors[0]
            lumen_features, epithelium_features = (
                future.result() for future in futures
            )
    else:
        lumen_features = calculate_lumen_features(labeled_lumen, min_area, n_jobs)
        epithelium_features = calculate_epithelium_features(
            labeled_epithelium, masks["epithelial_cells"], min_area, n_jobs
        )
    logging.info(
        "Found %d lumens and %d epithelial regions",
        len(lumen_features),
        len(epithelium_features),
    )
    return densities, lumen_features, epithelium_features


def _paint_feature_map(
    task: tuple[
        str,
        np.ndarray,
        pd.DataFrame,
        str,
        int,
        tuple[int, int],
        bool,
    ],
) -> tuple[str, np.ndarray]:
    key, labeled, features, column, max_label, block_size, full_resolution = task
    if column not in features.columns:
        raise ValueError(f"Feature column {column!r} not found")
    lookup = np.zeros(max_label + 1, dtype=np.float32)
    included = np.zeros(max_label + 1, dtype=np.bool_)
    if not features.empty:
        ids = features["label"].to_numpy(dtype=np.int64)
        values = np.nan_to_num(features[column].to_numpy(dtype=np.float32), nan=0.0)
        valid = (ids > 0) & (ids <= max_label)
        lookup[ids[valid]] = values[valid]
        included[ids[valid]] = True
    if full_resolution:
        return key, lookup[labeled]

    bh, bw = block_size
    output = np.zeros(
        (
            (labeled.shape[0] + bh - 1) // bh,
            (labeled.shape[1] + bw - 1) // bw,
        ),
        dtype=np.float32,
    )
    row_step = bh * MAT_CHUNK_BLOCK_ROWS
    for row_start in range(0, labeled.shape[0], row_step):
        row_stop = min(row_start + row_step, labeled.shape[0])
        chunk_labels = labeled[row_start:row_stop]
        painted = lookup[chunk_labels]
        present = included[chunk_labels]
        block_start = row_start // bh
        block_count = (painted.shape[0] + bh - 1) // bh
        sums = summify(painted, block_size)
        counts = summify(present, block_size)
        np.divide(
            sums,
            counts,
            out=output[block_start : block_start + block_count],
            where=counts > 0,
        )
    return key, output


def write_mat(
    label_map: np.ndarray,
    lumen_features: pd.DataFrame,
    epithelium_features: pd.DataFrame,
    path: str | Path,
    labels: Mapping[str, int] | None = None,
    block_size: tuple[int, int] = (20, 20),
    full_resolution: bool = False,
    n_jobs: int | None = None,
    densities: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write a MAT feature map from a label map and per-object feature tables.

    Per-object features are pixel-area-weighted means at density resolution
    unless ``full_resolution`` is set. Pixels belonging to filtered-out objects
    and background do not contribute to the block mean.
    """
    values = dict(DEFAULT_LABELS)
    if labels:
        values.update(labels)
    required_density_names = (
        "lumen_density",
        "stroma_density",
        "epithelium_density",
        "epithelial_cells_density",
    )
    if densities is None:
        masks = _create_tissue_masks(label_map, values, n_jobs)
        maps = _calculate_densities(masks, block_size, n_jobs)
    else:
        masks = _create_tissue_masks(
            label_map,
            values,
            n_jobs,
            ("lumen", "epithelium", "epithelial_cells"),
        )
        maps = {name: densities[name] for name in required_density_names}
    labeled_lumen = _label_components(masks["lumen"], n_jobs)
    labeled_epithelium = _label_components(masks["epithelium"], n_jobs)
    lumen_max = int(labeled_lumen.max()) if labeled_lumen.size else 0
    epithelium_max = int(labeled_epithelium.max()) if labeled_epithelium.size else 0
    painted = [
        ("lumen_roundness", labeled_lumen, lumen_features, "roundness", lumen_max),
        ("lumen_area", labeled_lumen, lumen_features, "area", lumen_max),
        (
            "epithelium_roundness",
            labeled_epithelium,
            epithelium_features,
            "roundness",
            epithelium_max,
        ),
        (
            "epithelium_area",
            labeled_epithelium,
            epithelium_features,
            "area",
            epithelium_max,
        ),
        (
            "epithelium_thickness",
            labeled_epithelium,
            epithelium_features,
            "average_thickness",
            epithelium_max,
        ),
        (
            "cell_fraction",
            labeled_epithelium,
            epithelium_features,
            "cell_fraction",
            epithelium_max,
        ),
    ]
    tasks = [(*item, block_size, full_resolution) for item in painted]
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers > 1 and label_map.size >= PARALLEL_LABEL_PIXEL_THRESHOLD:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            maps.update(dict(pool.map(_paint_feature_map, tasks)))
    else:
        maps.update(map(_paint_feature_map, tasks))
    scipy.io.savemat(path, maps)


def save_outputs(
    label_map: np.ndarray,
    densities: Mapping[str, np.ndarray],
    lumen_features: pd.DataFrame,
    epithelium_features: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    save_mat: bool = False,
    labels: Mapping[str, int] | None = None,
    block_size: tuple[int, int] = (20, 20),
    full_resolution_mat: bool = False,
    n_jobs: int | None = None,
) -> None:
    """Write per-object Parquet tables, density arrays, and optionally a MAT map."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lumen_features.to_parquet(output / f"{stem}_lumen.parquet", index=False)
    epithelium_features.to_parquet(output / f"{stem}_epithelium.parquet", index=False)
    np.savez_compressed(output / f"{stem}_densities.npz", **densities)
    if save_mat:
        write_mat(
            label_map,
            lumen_features,
            epithelium_features,
            output / f"{stem}_features.mat",
            labels,
            block_size,
            full_resolution_mat,
            n_jobs,
            densities,
        )
