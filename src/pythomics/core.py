"""Feature calculations for integer-valued histology label maps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.io
import skimage.io
import skimage.measure
import skimage.morphology

DEFAULT_LABELS = {
    "lumen": 1,
    "epithelium": 2,
    "stroma": 4,
    "epithelial_cells": 5,
    "stromal_cells": 6,
}


def load_label_map(path: str | Path) -> np.ndarray:
    """Read a 2-D label map; RGB representations are accepted only if channels match."""
    image = np.asarray(skimage.io.imread(str(path)))
    if image.ndim == 3:
        if image.shape[-1] not in (3, 4) or not np.all(image[..., :3] == image[..., :1]):
            raise ValueError("Expected a single-channel integer label map, not a color image")
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError(f"Expected integer label values, got dtype {image.dtype}")
    return image


def summify(image: np.ndarray, block_size: tuple[int, int] = (20, 20)) -> np.ndarray:
    """Sum image pixels in padded, non-overlapping blocks."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("summify expects a 2-D image")
    if not arr.size:
        return np.empty((0, 0), dtype=np.float32)
    bh, bw = block_size
    if bh <= 0 or bw <= 0:
        raise ValueError("block_size values must be positive")
    pad_h, pad_w = (-arr.shape[0]) % bh, (-arr.shape[1]) % bw
    arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
    return skimage.measure.block_reduce(arr, block_size=block_size, func=np.sum).astype(np.float32)


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


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def calculate_lumen_features(labeled_lumen: np.ndarray, min_area: int = 16) -> pd.DataFrame:
    columns = ["label", "area", "roundness"]
    if not np.any(labeled_lumen):
        return _empty(columns)
    props = skimage.measure.regionprops_table(labeled_lumen, properties=("label", "area", "perimeter"))
    frame = pd.DataFrame(props)
    frame["roundness"] = [calculate_tortuosity(a, p) for a, p in zip(frame.area, frame.perimeter)]
    frame = frame.loc[frame.area >= min_area, columns]
    return frame.reset_index(drop=True)


def calculate_epithelium_features(
    labeled_epithelium: np.ndarray, epithelial_cells: np.ndarray, min_area: int = 16
) -> pd.DataFrame:
    columns = ["label", "area", "roundness", "average_thickness", "cell_fraction"]
    if not np.any(labeled_epithelium):
        return _empty(columns)
    props = skimage.measure.regionprops_table(
        labeled_epithelium,
        intensity_image=epithelial_cells.astype(np.uint8),
        properties=("label", "area", "perimeter"),
        extra_properties=[average_thickness],
    )
    frame = pd.DataFrame(props)
    frame["roundness"] = [calculate_tortuosity(a, p) for a, p in zip(frame.area, frame.perimeter)]
    cell_counts = np.bincount(
        labeled_epithelium.ravel(), weights=epithelial_cells.astype(np.uint8).ravel()
    )
    counts = cell_counts[frame.label.to_numpy(dtype=np.int64)]
    frame["cell_fraction"] = np.divide(counts, frame.area, out=np.zeros_like(counts, dtype=float), where=frame.area.to_numpy() > 0)
    frame = frame.loc[frame.area >= min_area, columns]
    return frame.reset_index(drop=True)


def extract_features(
    label_map: np.ndarray,
    labels: Mapping[str, int] | None = None,
    min_area: int = 16,
    block_size: tuple[int, int] = (20, 20),
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
    stroma = (palette == label_values["stroma"]) | (palette == label_values["stromal_cells"])

    densities = {
        "lumen_density": summify(lumen, block_size),
        "stroma_density": summify(stroma, block_size),
        "epithelium_density": summify(epithelium, block_size),
        "epithelial_cells_density": summify(epithelial_cells, block_size),
    }
    labeled_lumen = scipy.ndimage.label(lumen)[0]
    labeled_epithelium = scipy.ndimage.label(epithelium)[0]
    lumen_features = calculate_lumen_features(labeled_lumen, min_area)
    epithelium_features = calculate_epithelium_features(labeled_epithelium, epithelial_cells, min_area)
    logging.info("Found %d lumens and %d epithelial regions", len(lumen_features), len(epithelium_features))
    return densities, lumen_features, epithelium_features


def _paint_features(labeled: np.ndarray, features: pd.DataFrame, column: str) -> np.ndarray:
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
) -> None:
    """Write per-object CSV tables, density arrays, and optionally a MAT feature map."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lumen_features.to_csv(output / f"{stem}_lumen_features.csv", index=False)
    epithelium_features.to_csv(output / f"{stem}_epithelium_features.csv", index=False)
    np.savez_compressed(output / f"{stem}_densities.npz", **densities)
    if not save_mat:
        return

    values = dict(DEFAULT_LABELS)
    if labels:
        values.update(labels)
    lumen = label_map == values["lumen"]
    epithelial_cells = label_map == values["epithelial_cells"]
    epithelium = (label_map == values["epithelium"]) | epithelial_cells
    labeled_lumen = scipy.ndimage.label(lumen)[0]
    labeled_epithelium = scipy.ndimage.label(epithelium)[0]
    maps = {
        **{key: summify(mask, block_size) for key, mask in {
            "lumen_density": lumen,
            "stroma_density": (label_map == values["stroma"]) | (label_map == values["stromal_cells"]),
            "epithelium_density": epithelium,
            "epithelial_cells_density": epithelial_cells,
        }.items()},
        "lumen_roundness": _paint_features(labeled_lumen, lumen_features, "roundness"),
        "lumen_area": _paint_features(labeled_lumen, lumen_features, "area"),
        "epithelium_roundness": _paint_features(labeled_epithelium, epithelium_features, "roundness"),
        "epithelium_area": _paint_features(labeled_epithelium, epithelium_features, "area"),
        "epithelium_thickness": _paint_features(labeled_epithelium, epithelium_features, "average_thickness"),
        "cell_fraction": _paint_features(labeled_epithelium, epithelium_features, "cell_fraction"),
    }
    scipy.io.savemat(output / f"{stem}_features.mat", maps)
