"""Tests for label-map feature extraction and output generation."""

import numpy as np
import pandas as pd
import pytest
import scipy.io

from pythomics import core


def test_load_label_map_accepts_grayscale_and_matching_rgb(monkeypatch, tmp_path):
    """Grayscale input and repeated RGB channels both yield a 2-D label map."""
    labels = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    path = tmp_path / "labels.png"

    monkeypatch.setattr(core.skimage.io, "imread", lambda _: labels)
    np.testing.assert_array_equal(core.load_label_map(path), labels)

    rgb_labels = np.repeat(labels[..., np.newaxis], 3, axis=2)
    monkeypatch.setattr(core.skimage.io, "imread", lambda _: rgb_labels)
    np.testing.assert_array_equal(core.load_label_map(path), labels)


def test_load_label_map_disables_pillow_pixel_limit_during_read(monkeypatch, tmp_path):
    labels = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    max_image_pixels = core.PILImage.MAX_IMAGE_PIXELS

    def imread(_):
        assert core.PILImage.MAX_IMAGE_PIXELS is None
        return labels

    monkeypatch.setattr(core.skimage.io, "imread", imread)

    np.testing.assert_array_equal(core.load_label_map(tmp_path / "labels.png"), labels)
    assert core.PILImage.MAX_IMAGE_PIXELS == max_image_pixels


def test_load_label_map_restores_pillow_pixel_limit_after_read_error(
    monkeypatch, tmp_path
):
    max_image_pixels = core.PILImage.MAX_IMAGE_PIXELS

    def imread(_):
        assert core.PILImage.MAX_IMAGE_PIXELS is None
        raise OSError("could not read image")

    monkeypatch.setattr(core.skimage.io, "imread", imread)

    with pytest.raises(OSError, match="could not read image"):
        core.load_label_map(tmp_path / "labels.png")
    assert core.PILImage.MAX_IMAGE_PIXELS == max_image_pixels


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.array([[[0, 1, 2]]], dtype=np.uint8), "single-channel"),
        (np.zeros((2, 2, 2), dtype=np.uint8), "single-channel"),
        (np.zeros((2, 2), dtype=np.float32), "integer label values"),
        (np.zeros((2, 2, 2, 2), dtype=np.uint8), "2-D label map"),
    ],
)
def test_load_label_map_rejects_invalid_input(monkeypatch, tmp_path, image, message):
    monkeypatch.setattr(core.skimage.io, "imread", lambda _: image)

    with pytest.raises(ValueError, match=message):
        core.load_label_map(tmp_path / "labels.png")


def test_summify_pads_partial_edge_blocks():
    image = np.arange(15, dtype=np.uint8).reshape(3, 5)

    result = core.summify(image, block_size=(2, 3))

    np.testing.assert_array_equal(result, [[21, 24], [33, 27]])
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("image", "block_size", "message"),
    [
        (np.zeros((2, 2, 2)), (1, 1), "2-D"),
        (np.zeros((2, 2)), (0, 1), "positive"),
        (np.zeros((2, 2)), (1, -1), "positive"),
    ],
)
def test_summify_rejects_invalid_arguments(image, block_size, message):
    with pytest.raises(ValueError, match=message):
        core.summify(image, block_size)


def test_summify_empty_image_returns_empty_float_array():
    result = core.summify(np.empty((0, 3)))

    assert result.shape == (0, 0)
    assert result.dtype == np.float32


@pytest.mark.parametrize(
    ("area", "perimeter", "expected"),
    [(0, 5, 0.0), (5, 0, 0.0), (1, 1, 1.0), (1, 4, np.pi / 4)],
)
def test_calculate_tortuosity_handles_nonpositive_and_caps_roundness(
    area, perimeter, expected
):
    assert core.calculate_tortuosity(area, perimeter) == pytest.approx(expected)


def test_average_thickness_is_zero_for_empty_mask_and_measures_filled_width():
    assert core.average_thickness(np.zeros((3, 3), dtype=bool)) == 0.0
    assert core.average_thickness(np.ones((3, 3), dtype=bool)) > 1.0


def test_extract_features_counts_combined_tissues_and_calculates_objects():
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    labels[0:2, 4:6] = 2
    labels[1, 4] = 5
    labels[4:6, 0:2] = 4
    labels[6, 0] = 6

    densities, lumen, epithelium = core.extract_features(
        labels, min_area=1, block_size=(4, 4)
    )

    np.testing.assert_array_equal(densities["lumen_density"], [[4, 0], [0, 0]])
    np.testing.assert_array_equal(densities["epithelium_density"], [[0, 4], [0, 0]])
    np.testing.assert_array_equal(
        densities["epithelial_cells_density"], [[0, 1], [0, 0]]
    )
    np.testing.assert_array_equal(densities["stroma_density"], [[0, 0], [5, 0]])
    assert lumen[["area", "roundness"]].to_dict("records") == [
        {"area": 4, "roundness": 1.0}
    ]
    assert epithelium["area"].tolist() == [4]
    assert epithelium["cell_fraction"].tolist() == [0.25]
    assert epithelium["average_thickness"].iloc[0] > 0


def test_region_area_filtering_keeps_original_labels():
    labeled = np.zeros((4, 6), dtype=np.int32)
    labeled[0, 0] = 1
    labeled[2:4, 3:5] = 2

    features = core.calculate_lumen_features(labeled, min_area=2, n_jobs=1)

    assert features[["label", "area"]].to_dict("records") == [{"label": 2, "area": 4}]


def test_parallel_region_measurements_match_serial(monkeypatch):
    labels = np.zeros((12, 20), dtype=np.uint8)
    labels[1:4, 1:5] = 1
    labels[7:10, 1:5] = 1
    labels[1:4, 11:15] = 2
    labels[7:10, 11:15] = 2
    labels[2, 12] = 5
    labels[8, 13] = 5
    monkeypatch.setattr(core, "PARALLEL_REGION_AREA_THRESHOLD", 0)
    monkeypatch.setattr(core, "PARALLEL_LABEL_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(core, "LABEL_CHUNK_ROWS", 3)

    serial = core.extract_features(labels, min_area=1, n_jobs=1)
    parallel = core.extract_features(labels, min_area=1, n_jobs=2)

    for name in serial[0]:
        np.testing.assert_array_equal(serial[0][name], parallel[0][name])
    for serial_frame, parallel_frame in zip(serial[1:], parallel[1:], strict=True):
        pd.testing.assert_frame_equal(serial_frame, parallel_frame)


def test_parallel_component_labels_match_serial_across_chunk_boundaries(monkeypatch):
    mask = np.zeros((11, 12), dtype=bool)
    mask[1:9, 2] = True
    mask[2:5, 7] = True
    mask[5, 8] = True
    mask[6, 9] = True
    mask[10, 10] = True
    monkeypatch.setattr(core, "PARALLEL_LABEL_PIXEL_THRESHOLD", 0)
    monkeypatch.setattr(core, "LABEL_CHUNK_ROWS", 3)

    serial = core.scipy.ndimage.label(mask)[0]
    parallel = core._label_components(mask, n_jobs=3)

    np.testing.assert_array_equal(parallel, serial)


def test_component_label_compaction_resolves_transitive_merges():
    parent = np.array([0, 1, 1, 2, 3, 4], dtype=np.int32)

    lookup = core._compact_component_labels(parent)

    np.testing.assert_array_equal(lookup, [0, 1, 1, 1, 1, 1])


def test_extract_features_filters_small_regions_and_returns_empty_tables():
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[0, 0] = 1
    labels[3:5, 3:5] = 2

    _, lumen, epithelium = core.extract_features(labels, min_area=5)

    assert lumen.empty
    assert list(lumen.columns) == ["label", "area", "roundness"]
    assert epithelium.empty
    assert list(epithelium.columns) == [
        "label",
        "area",
        "roundness",
        "average_thickness",
        "cell_fraction",
    ]


def test_extract_features_uses_default_minimum_area():
    labels = np.ones((10, 10), dtype=np.uint8)

    _, lumen, epithelium = core.extract_features(labels)

    assert lumen.empty
    assert epithelium.empty


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.zeros((2, 2, 2), dtype=np.uint8), "2-D integer"),
        (np.zeros((2, 2), dtype=np.float32), "2-D integer"),
    ],
)
def test_extract_features_rejects_non_integer_2d_maps(labels, message):
    with pytest.raises(ValueError, match=message):
        core.extract_features(labels)


def test_extract_features_rejects_unknown_and_duplicate_labels():
    labels = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown label names"):
        core.extract_features(labels, labels={"unknown": 9})
    with pytest.raises(ValueError, match="distinct"):
        core.extract_features(labels, labels={"lumen": 2})


def test_paint_features_projects_region_values_and_validates_column():
    labeled = np.array([[0, 1], [2, 2]], dtype=np.int32)
    features = pd.DataFrame({"label": [1], "area": [3]})

    np.testing.assert_array_equal(
        core._paint_feature_map(("area", labeled, features, "area", 2, (1, 1), True))[
            1
        ],
        [[0, 3], [0, 0]],
    )
    with pytest.raises(ValueError, match="not found"):
        core._paint_feature_map(
            ("missing", labeled, features, "missing", 2, (1, 1), True)
        )


@pytest.mark.parametrize("column", ["roundness", "average_thickness", "cell_fraction"])
def test_painted_mat_features_are_area_weighted_means(column, monkeypatch):
    labeled = np.array([[1, 1, 0, 2, 2, 0], [1, 1, 0, 2, 2, 0]], dtype=np.int32)
    features = pd.DataFrame(
        {
            "label": [1, 2],
            "roundness": [0.25, 0.75],
            "average_thickness": [2.0, 6.0],
            "cell_fraction": [0.2, 0.8],
        }
    )
    monkeypatch.setattr(core, "MAT_CHUNK_BLOCK_ROWS", 1)

    actual = core._paint_feature_map(
        (column, labeled, features, column, 2, (2, 6), False)
    )[1]
    expected = {
        "roundness": (0.25 * 4 + 0.75 * 4) / 8,
        "average_thickness": (2.0 * 4 + 6.0 * 4) / 8,
        "cell_fraction": (0.2 * 4 + 0.8 * 4) / 8,
    }[column]
    np.testing.assert_allclose(actual, [[expected]])


def test_save_outputs_writes_parquet_density_and_optional_mat_files(
    tmp_path, monkeypatch
):
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    densities, lumen, epithelium = core.extract_features(
        labels, min_area=1, block_size=(2, 2)
    )

    monkeypatch.setattr(
        core,
        "_calculate_densities",
        lambda *args: pytest.fail("MAT output should reuse extracted densities"),
    )
    core.save_outputs(
        labels,
        densities,
        lumen,
        epithelium,
        tmp_path / "csv-output",
        "sample",
    )

    output = tmp_path / "csv-output"
    assert pd.read_parquet(output / "sample_lumen.parquet")["area"].tolist() == [4]
    assert pd.read_parquet(output / "sample_epithelium.parquet").empty
    with np.load(output / "sample_densities.npz") as saved:
        np.testing.assert_array_equal(
            saved["lumen_density"], densities["lumen_density"]
        )
    assert not (output / "sample_features.mat").exists()

    core.save_outputs(
        labels,
        densities,
        lumen,
        epithelium,
        tmp_path / "mat-output",
        "sample",
        save_mat=True,
        block_size=(2, 2),
    )
    maps = scipy.io.loadmat(tmp_path / "mat-output" / "sample_features.mat")
    np.testing.assert_array_equal(
        maps["lumen_area"],
        np.array([[4, 0], [0, 0]]),
    )
    np.testing.assert_array_equal(maps["lumen_density"], densities["lumen_density"])


def test_written_mat_feature_maps_match_feature_tables(tmp_path):
    labels = np.zeros((8, 16), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    labels[0:2, 4:6] = 1
    labels[4:6, 0:2] = 2
    labels[4, 0] = 5
    labels[4:6, 4:6] = 2
    labels[4, 4:6] = 5
    block_size = (8, 16)
    _, lumen, epithelium = core.extract_features(
        labels, min_area=1, block_size=block_size, n_jobs=1
    )
    path = tmp_path / "features.mat"

    core.write_mat(
        labels,
        lumen,
        epithelium,
        path,
        block_size=block_size,
        n_jobs=1,
    )
    maps = scipy.io.loadmat(path)

    for map_name, frame, feature in (
        ("lumen_roundness", lumen, "roundness"),
        ("epithelium_roundness", epithelium, "roundness"),
        ("epithelium_thickness", epithelium, "average_thickness"),
        ("cell_fraction", epithelium, "cell_fraction"),
    ):
        expected = np.average(frame[feature], weights=frame["area"])
        assert maps[map_name].item() == pytest.approx(expected)
