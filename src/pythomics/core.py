"""Feature calculations for integer-valued histology label maps."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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


def _default_job_count() -> int:
    return min(MAX_DEFAULT_JOBS, os.cpu_count() or 1)


def load_label_map(path: str | Path) -> np.ndarray:
    """Read a 2-D label map; RGB representations are accepted only if channels match."""
    max_image_pixels = PILImage.MAX_IMAGE_PIXELS
    PILImage.MAX_IMAGE_PIXELS = None
    try:
        image = np.asarray(skimage.io.imread(str(path)))
    finally:
        PILImage.MAX_IMAGE_PIXELS = max_image_pixels
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


def _parallel_thickness(
    labeled: np.ndarray, label_ids: np.ndarray, n_jobs: int | None
) -> np.ndarray:
    """Average thickness per label, computed on bounding-box crops across processes."""
    slices = scipy.ndimage.find_objects(labeled)
    masks = [labeled[slices[i - 1]] == i for i in label_ids]
    workers = n_jobs if n_jobs is not None else _default_job_count()
    if workers <= 1 or len(masks) < 2:
        return np.array([_thickness_of_mask(m) for m in masks], dtype=np.float64)
    chunk = max(1, len(masks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=min(workers, len(masks))) as pool:
        return np.fromiter(
            pool.map(_thickness_of_mask, masks, chunksize=chunk),
            dtype=np.float64,
            count=len(masks),
        )


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def calculate_lumen_features(
    labeled_lumen: np.ndarray, min_area: int = DEFAULT_MIN_AREA
) -> pd.DataFrame:
    columns = ["label", "area", "roundness"]
    if not np.any(labeled_lumen):
        return _empty(columns)
    props = skimage.measure.regionprops_table(
        labeled_lumen, properties=("label", "area", "perimeter")
    )
    frame = pd.DataFrame(props)
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
    props = skimage.measure.regionprops_table(
        labeled_epithelium,
        intensity_image=epithelial_cells,
        properties=("label", "area", "perimeter", "intensity_mean"),
    )
    frame = pd.DataFrame(props)
    frame = frame.loc[frame.area >= min_area].reset_index(drop=True)
    frame["average_thickness"] = _parallel_thickness(
        labeled_epithelium, frame["label"].to_numpy(), n_jobs
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
    lumen = palette == label_values["lumen"]
    epithelial_cells = palette == label_values["epithelial_cells"]
    epithelium = (palette == label_values["epithelium"]) | epithelial_cells
    stroma = (palette == label_values["stroma"]) | (
        palette == label_values["stromal_cells"]
    )

    densities = {
        "lumen_density": summify(lumen, block_size),
        "stroma_density": summify(stroma, block_size),
        "epithelium_density": summify(epithelium, block_size),
        "epithelial_cells_density": summify(epithelial_cells, block_size),
    }
    labeled_lumen = scipy.ndimage.label(lumen)[0]
    labeled_epithelium = scipy.ndimage.label(epithelium)[0]
    lumen_features = calculate_lumen_features(labeled_lumen, min_area)
    epithelium_features = calculate_epithelium_features(
        labeled_epithelium, epithelial_cells, min_area, n_jobs
    )
    logging.info(
        "Found %d lumens and %d epithelial regions",
        len(lumen_features),
        len(epithelium_features),
    )
    return densities, lumen_features, epithelium_features


def _paint_features(
    labeled: np.ndarray, features: pd.DataFrame, column: str
) -> np.ndarray:
    if column not in features.columns:
        raise ValueError(f"Feature column {column!r} not found")
    max_label = int(labeled.max()) if labeled.size else 0
    lookup = np.zeros(max_label + 1, dtype=np.float32)
    if not features.empty:
        ids = features["label"].to_numpy(dtype=np.int64)
        values = features[column].to_numpy(dtype=np.float32)
        valid = (ids > 0) & (ids <= max_label)
        values = np.nan_to_num(values, nan=0.0)
        lookup[ids[valid]] = values[valid]
    return lookup[labeled]


def _block_max(image: np.ndarray, block_size: tuple[int, int]) -> np.ndarray:
    bh, bw = block_size
    columns = np.maximum.reduceat(image, np.arange(0, image.shape[1], bw), axis=1)
    return np.maximum.reduceat(columns, np.arange(0, image.shape[0], bh), axis=0)


def write_mat(
    label_map: np.ndarray,
    lumen_features: pd.DataFrame,
    epithelium_features: pd.DataFrame,
    path: str | Path,
    labels: Mapping[str, int] | None = None,
    block_size: tuple[int, int] = (20, 20),
    full_resolution: bool = False,
) -> None:
    """Write a MAT feature map from a label map and per-object feature tables.

    Maps are at density resolution (one value per block, the block maximum for
    per-object features) unless ``full_resolution`` is set.
    """
    values = dict(DEFAULT_LABELS)
    if labels:
        values.update(labels)
    lumen = label_map == values["lumen"]
    epithelial_cells = label_map == values["epithelial_cells"]
    epithelium = (label_map == values["epithelium"]) | epithelial_cells
    stroma = (label_map == values["stroma"]) | (label_map == values["stromal_cells"])
    labeled_lumen = scipy.ndimage.label(lumen)[0]
    labeled_epithelium = scipy.ndimage.label(epithelium)[0]
    maps = {
        "lumen_density": summify(lumen, block_size),
        "stroma_density": summify(stroma, block_size),
        "epithelium_density": summify(epithelium, block_size),
        "epithelial_cells_density": summify(epithelial_cells, block_size),
    }
    painted = {
        "lumen_roundness": (labeled_lumen, lumen_features, "roundness"),
        "lumen_area": (labeled_lumen, lumen_features, "area"),
        "epithelium_roundness": (labeled_epithelium, epithelium_features, "roundness"),
        "epithelium_area": (labeled_epithelium, epithelium_features, "area"),
        "epithelium_thickness": (
            labeled_epithelium,
            epithelium_features,
            "average_thickness",
        ),
        "cell_fraction": (labeled_epithelium, epithelium_features, "cell_fraction"),
    }
    for key, (labeled, features, column) in painted.items():
        full = _paint_features(labeled, features, column)
        maps[key] = full if full_resolution else _block_max(full, block_size)
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
        )
