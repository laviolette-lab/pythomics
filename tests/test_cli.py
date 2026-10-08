"""Tests for the TAPS command-line interface."""

import logging

from taps import cli
from taps.inference import BlankMaskError


def test_segment_command_delegates_to_library(monkeypatch, caplog):
    """The CLI passes parsed segment arguments to the inference API."""
    caplog.set_level(logging.INFO)
    calls = []

    def fake_segment(image, output, checkpoint, device, exact, **kwargs):
        calls.append(
            (
                image,
                output,
                checkpoint,
                device,
                exact,
                kwargs["sigma"],
                kwargs["threshold"],
            )
        )
        return output

    monkeypatch.setattr(cli, "segment", fake_segment)

    exit_code = cli.main(
        [
            "segment",
            "scan.nii.gz",
            "mask.nii.gz",
            "--checkpoint",
            "model.pth",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("scan.nii.gz", "mask.nii.gz", "model.pth", "cpu", False, 2.5, 0.55)
    ]
    assert "Saved segmentation to mask.nii.gz" in caplog.text


def test_segment_command_uses_bundled_checkpoint_by_default(monkeypatch, caplog):
    """The CLI should use the packaged checkpoint when no override is provided."""
    caplog.set_level(logging.INFO)
    calls = []

    def fake_segment(image, output, checkpoint, device, exact, **kwargs):
        calls.append(
            (
                image,
                output,
                checkpoint,
                device,
                exact,
                kwargs["sigma"],
                kwargs["threshold"],
            )
        )
        return output

    monkeypatch.setattr(cli, "segment", fake_segment)

    exit_code = cli.main(["segment", "scan.nii.gz", "mask.nii.gz", "--device", "cpu"])

    assert exit_code == 0
    assert calls == [("scan.nii.gz", "mask.nii.gz", None, "cpu", False, 2.5, 0.55)]
    assert "Saved segmentation to mask.nii.gz" in caplog.text


def test_segment_command_allows_bundled_checkpoint_by_default():
    """The CLI should not require an explicit checkpoint path for the packaged model."""
    parser = cli.build_parser()

    args = parser.parse_args(["segment", "scan.nii.gz", "mask.nii.gz"])

    assert args.image == "scan.nii.gz"
    assert args.output == "mask.nii.gz"
    assert args.checkpoint is None
    assert args.exact is False


def test_segment_command_accepts_exact_flag():
    """The exact flag is exposed on the segment subcommand."""
    args = cli.build_parser().parse_args(
        ["segment", "scan.nii.gz", "mask.nii.gz", "--exact"]
    )

    assert args.exact is True


def test_segment_command_returns_one_for_blank_mask(monkeypatch, caplog):
    """The CLI returns failure when inference produces a blank mask."""

    def fake_segment(image, output, checkpoint, device, exact, **kwargs):
        raise BlankMaskError("Inferred segmentation mask is blank: mask.nii.gz")

    monkeypatch.setattr(cli, "segment", fake_segment)

    exit_code = cli.main(["segment", "scan.nii.gz", "mask.nii.gz"])

    assert exit_code == 1
    assert "Inferred segmentation mask is blank" in caplog.text
