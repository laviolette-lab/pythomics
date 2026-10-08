"""Tests for TAPS inference utilities."""

import nibabel as nib
import numpy as np
import pytest
import torch

from taps.inference import (
    DEFAULT_PROBABILITY_THRESHOLD,
    DEFAULT_SIGMA,
    _run_qc,
    _warn_border_contact,
    _warn_geometry,
    _warn_holes,
    _warn_slice_continuity,
    gaussian_blur_logits,
    resolve_checkpoint,
    segment,
)


def test_default_postprocessing_matches_optimized_settings():
    """The packaged model defaults should match the optimized blur/threshold settings."""
    logits = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])

    assert DEFAULT_SIGMA == 2.5
    assert DEFAULT_PROBABILITY_THRESHOLD == 0.55
    blurred = gaussian_blur_logits(logits)
    assert blurred.shape == logits.shape
    assert torch.isfinite(blurred).all()


def test_resolve_checkpoint_returns_explicit_path(tmp_path):
    """An explicit checkpoint path is returned unchanged."""
    checkpoint = tmp_path / "custom.pth"

    assert resolve_checkpoint(checkpoint) == checkpoint


def test_resolve_checkpoint_falls_back_to_bundled_model():
    """With no override, the bundled checkpoint is used."""
    bundled = resolve_checkpoint(None)

    assert bundled.name == "model_v1.onnx"
    assert bundled.is_file()


def test_segment_raises_for_missing_image(tmp_path):
    """A missing input image raises FileNotFoundError before any inference."""
    with pytest.raises(FileNotFoundError, match="Input image"):
        segment(tmp_path / "missing.nii.gz", tmp_path / "out.nii.gz", None)


def test_segment_raises_for_missing_checkpoint(tmp_path):
    """A missing checkpoint override raises FileNotFoundError."""
    image = tmp_path / "scan.nii.gz"
    image.touch()

    with pytest.raises(FileNotFoundError, match="Checkpoint"):
        segment(image, tmp_path / "out.nii.gz", tmp_path / "missing.pth")


def test_segment_raises_for_invalid_output_suffix(tmp_path):
    """An output path without a NIfTI suffix raises ValueError."""
    image = tmp_path / "scan.nii.gz"
    image.touch()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="must end in"):
        segment(image, tmp_path / "out.txt", checkpoint)


def test_qc_marks_abnormal_volume_and_keeps_largest_component(tmp_path, caplog):
    """QC warns on volume/components and writes an abnormal output suffix."""
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[:3, :3, :2] = 1
    mask[8:, 8:, :2] = 1
    output = tmp_path / "mask.nii.gz"
    affine = np.diag([1, 1, 1, 1])

    cleaned, abnormal, selected_output = _run_qc(mask, affine, output, exact=False)

    assert abnormal is True
    assert cleaned.sum() == 18
    assert selected_output == tmp_path / "mask_abnormal.nii.gz"
    assert "outside the expected" in caplog.text
    assert "disconnected 3D components" in caplog.text


def test_qc_blanks_multi_component_slices_and_writes_cleaned_mask(tmp_path, caplog):
    """QC writes a cleaned NIfTI when a slice has multiple components."""
    mask = np.zeros((10, 10, 2), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    mask[7:9, 7:9, 0] = 1
    output = tmp_path / "mask.nii.gz"

    _, abnormal, selected_output = _run_qc(
        mask, np.diag([3, 3, 3, 1]), output, exact=False
    )

    cleaned_path = tmp_path / "mask_cleaned.nii.gz"
    assert abnormal is True
    assert selected_output == tmp_path / "mask_abnormal.nii.gz"
    assert cleaned_path.is_file()
    assert np.count_nonzero(nib.load(cleaned_path).get_fdata()) == 0
    assert "multiple 2D components" in caplog.text


def test_qc_exact_does_not_append_abnormal(tmp_path):
    """Exact mode leaves the requested output name unchanged."""
    mask = np.ones((3, 3, 3), dtype=np.uint8)
    output = tmp_path / "mask.nii.gz"

    _, abnormal, selected_output = _run_qc(
        mask, np.diag([1, 1, 1, 1]), output, exact=True
    )

    assert abnormal is True
    assert selected_output == output
    assert not (tmp_path / "mask_abnormal.nii.gz").exists()


def test_qc_warns_for_border_contact():
    """QC detects foreground touching the image boundary."""
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 1:3, 1:3] = 1

    assert _warn_border_contact(mask) is True


def test_qc_warns_for_slice_gaps_and_isolated_runs():
    """QC detects gaps and one-slice runs in the occupied slice indices."""
    mask = np.zeros((4, 4, 6), dtype=np.uint8)
    mask[1:3, 1:3, 0] = 1
    mask[1:3, 1:3, 2:4] = 1

    assert _warn_slice_continuity(mask) is True


def test_qc_warns_for_enclosed_holes():
    """QC detects background enclosed by foreground."""
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    mask[2, 2, 2] = 0

    assert _warn_holes(mask) is True


def test_qc_warns_for_implausible_geometry():
    """QC detects a physically implausible mask extent."""
    mask = np.ones((2, 2, 2), dtype=np.uint8)

    assert _warn_geometry(mask, np.diag([1, 1, 1, 1])) is True
