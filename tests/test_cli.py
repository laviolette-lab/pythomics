"""Tests for the standalone pythomics command-line interface."""

import numpy as np
import pytest
import skimage.io

from pythomics import cli


def test_build_parser_exposes_defaults_and_label_overrides():
    args = cli.build_parser().parse_args(
        ["labels.png", "--block-size", "2", "3", "--epithelium-label", "7"]
    )

    assert args.label_map.name == "labels.png"
    assert args.output_dir.name == "pythomics-output"
    assert args.min_area == 16
    assert args.block_size == [2, 3]
    assert args.epithelium_label == 7
    assert args.save_mat is False


def test_main_writes_feature_outputs_for_input_image(tmp_path):
    image = tmp_path / "labels.png"
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    labels[2:4, 2:4] = 2
    skimage.io.imsave(image, labels, check_contrast=False)
    output = tmp_path / "features"

    assert cli.main(
        [
            str(image),
            "--output-dir",
            str(output),
            "--min-area",
            "1",
            "--block-size",
            "2",
            "2",
            "--save-mat",
        ]
    ) == 0

    assert (output / "labels_lumen_features.csv").is_file()
    assert (output / "labels_epithelium_features.csv").is_file()
    assert (output / "labels_densities.npz").is_file()
    assert (output / "labels_features.mat").is_file()


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--min-area", "-1"], "--min-area must be non-negative"),
        (["--block-size", "0", "2"], "--min-area must be non-negative"),
        (["--block-size", "2", "-1"], "--min-area must be non-negative"),
        (["--lumen-label", "2"], "label values must be distinct"),
    ],
)
def test_main_rejects_invalid_options(tmp_path, extra_args, message):
    image = tmp_path / "labels.png"
    skimage.io.imsave(image, np.zeros((2, 2), dtype=np.uint8), check_contrast=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(image), *extra_args])

    assert exc_info.value.code == 2


def test_main_rejects_missing_input_file(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(tmp_path / "missing.tif")])

    assert exc_info.value.code == 2
