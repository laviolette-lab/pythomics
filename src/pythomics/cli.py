"""Command-line interface for standalone pythomics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .core import (
    DEFAULT_LABELS,
    DEFAULT_MIN_AREA,
    _default_job_count,
    extract_features,
    load_label_map,
    save_outputs,
    write_mat,
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
        help="Also write a feature-map MAT file at density (block) resolution",
    )
    parser.add_argument(
        "--save-full-mat",
        action="store_true",
        help="Write the MAT file at full resolution (implies --save-mat)",
    )
    parser.add_argument(
        "--lumen-parquet",
        type=Path,
        help="Existing lumen parquet; with --epithelium-parquet, only write the MAT file",
    )
    parser.add_argument(
        "--epithelium-parquet",
        type=Path,
        help="Existing epithelium parquet; with --lumen-parquet, only write the MAT file",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_job_count(),
        help="Worker threads for large-image calculations (default: up to 12 CPUs)",
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

    parquets = (args.lumen_parquet, args.epithelium_parquet)
    if any(parquets) and not all(parquets):
        build_parser().error(
            "--lumen-parquet and --epithelium-parquet must be given together"
        )
    if args.jobs < 1:
        build_parser().error("--jobs must be positive")
    block_size = tuple(args.block_size)

    label_map = load_label_map(args.label_map, n_jobs=args.jobs)
    if all(parquets):
        for path in parquets:
            if not path.is_file():
                build_parser().error(f"parquet file does not exist: {path}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_mat(
            label_map,
            pd.read_parquet(args.lumen_parquet),
            pd.read_parquet(args.epithelium_parquet),
            args.output_dir / f"{args.label_map.stem}_features.mat",
            labels=labels,
            block_size=block_size,
            full_resolution=args.save_full_mat,
            n_jobs=args.jobs,
        )
        logging.info("Wrote MAT file to %s", args.output_dir)
        return 0

    densities, lumen, epithelium = extract_features(
        label_map,
        labels=labels,
        min_area=args.min_area,
        block_size=block_size,
        n_jobs=args.jobs,
    )
    save_outputs(
        label_map,
        densities,
        lumen,
        epithelium,
        args.output_dir,
        args.label_map.stem,
        save_mat=args.save_mat or args.save_full_mat,
        labels=labels,
        block_size=block_size,
        full_resolution_mat=args.save_full_mat,
        n_jobs=args.jobs,
    )
    logging.info("Wrote outputs to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
