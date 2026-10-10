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
    assert args.min_area == 2048
    assert args.block_size == [2, 3]
    assert args.epithelium_label == 7
    assert args.save_mat is False


def test_jobs_default_is_capped_and_can_be_overridden(monkeypatch):
    monkeypatch.setattr(cli, "_default_job_count", lambda: 12)

    assert cli.build_parser().parse_args(["labels.png"]).jobs == 12
    assert cli.build_parser().parse_args(["labels.png", "--jobs", "24"]).jobs == 24


def test_main_writes_feature_outputs_for_input_image(tmp_path):
    image = tmp_path / "labels.png"
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    labels[2:4, 2:4] = 2
    skimage.io.imsave(image, labels, check_contrast=False)
    output = tmp_path / "features"

    assert (
        cli.main(
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
        )
        == 0
    )

    assert (output / "labels_lumen.parquet").is_file()
    assert (output / "labels_epithelium.parquet").is_file()
    assert (output / "labels_densities.npz").is_file()
    assert (output / "labels_features.mat").is_file()


def test_mat_from_parquet_and_full_resolution(tmp_path):
    import scipy.io

    image = tmp_path / "labels.png"
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[0:2, 0:2] = 1
    skimage.io.imsave(image, labels, check_contrast=False)
    out = tmp_path / "o"
    base = [str(image), "-o", str(out), "--min-area", "1", "--block-size", "2", "2"]
    assert cli.main(base) == 0
    assert cli.main([*base, "--save-mat"]) == 0
    small = scipy.io.loadmat(out / "labels_features.mat")["lumen_area"]
    assert small.shape == (2, 2) and small[0, 0] == 4

    (out / "labels_features.mat").unlink()
    parquet = [
        "--lumen-parquet",
        str(out / "labels_lumen.parquet"),
        "--epithelium-parquet",
        str(out / "labels_epithelium.parquet"),
    ]
    assert cli.main([*base, *parquet, "--save-full-mat"]) == 0
    full = scipy.io.loadmat(out / "labels_features.mat")["lumen_area"]
    assert full.shape == (4, 4)


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
