"""Command-line interface for standalone pythomics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .core import (
    DEFAULT_LABELS,
    DEFAULT_MIN_AREA,
    extract_features,
    load_label_map,
    save_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate lumen and epithelium features from an integer label map."
    )
    parser.add_argument(
        "label_map", type=Path, help="Input 2-D integer label map image"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("pythomics-output")
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum connected-region area in pixels",
    )
    parser.add_argument(
        "--block-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(20, 20)
    )
    parser.add_argument(
        "--save-mat",
        action="store_true",
        help="Also write a full-resolution feature-map MAT file",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    for name, default in DEFAULT_LABELS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}-label",
            type=int,
            default=default,
            help=f"Label value for {name}",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s"
    )
    if args.min_area < 0 or min(args.block_size) <= 0:
        build_parser().error(
            "--min-area must be non-negative and --block-size values must be positive"
        )
    if not args.label_map.is_file():
        build_parser().error(f"label map does not exist: {args.label_map}")

    labels = {name: getattr(args, f"{name}_label") for name in DEFAULT_LABELS}
    if len(set(labels.values())) != len(labels):
        build_parser().error("label values must be distinct")

    label_map = load_label_map(args.label_map)
    densities, lumen, epithelium = extract_features(
        label_map,
        labels=labels,
        min_area=args.min_area,
        block_size=tuple(args.block_size),
    )
    save_outputs(
        label_map,
        densities,
        lumen,
        epithelium,
        args.output_dir,
        args.label_map.stem,
        save_mat=args.save_mat,
        labels=labels,
        block_size=tuple(args.block_size),
    )
    logging.info("Wrote outputs to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
