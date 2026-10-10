# Standalone pythomics feature extraction

Extract per-object lumen and epithelium measurements and block-summed density maps from an integer label-map image. This folder is self-contained and does not import the repository's model/inference code.

## Install

From this directory, install the listed dependencies into your environment:

```bash
python -m pip install -r requirements.txt
```

Image readers are provided by scikit-image. Install `glymur` (`python -m pip install glymur`) to enable multi-threaded JPEG2000 decoding; otherwise JPEG2000 files use the scikit-image reader. Object-level tables are Parquet, density arrays are compressed NPZ, and MAT output is optional.

## CLI

Run from the repository root:

```bash
python -m pythomics_standalone path/to/labels.png --output-dir output --save-mat
```

To regenerate a MAT file using an existing label map and object tables:

```bash
python -m pythomics_standalone path/to/labels.png \
  --lumen-parquet output/labels_lumen.parquet \
  --epithelium-parquet output/labels_epithelium.parquet \
  --output-dir output
```

The default label values are lumen=1, epithelium=2, stroma=4, epithelial-cells=5, stromal-cells=6. Override with `--lumen-label`, `--epithelium-label`, `--stroma-label`, `--epithelial-cells-label`, and `--stromal-cells-label`. `--min-area` sets the minimum connected component size in source pixels (default 2048); `--block-size HEIGHT WIDTH` sets density aggregation blocks (default 20 20).

Epithelium regions include both epithelium and epithelial-cell pixels. Stroma density includes stromal-cell pixels. Connected components are labeled independently in each tissue mask. Tables contain source-pixel area, circularity (`roundness`), and for epithelium average skeleton thickness and epithelial-cell fraction. Density NPZ arrays are block sums, with edge blocks padded with zeros. MAT density maps are block sums; per-object maps at density resolution are pixel-area-weighted means of the feature values for included objects in each block, excluding background and regions filtered out by `--min-area`. `--save-full-mat` writes those per-object maps at source-pixel resolution instead.

Outputs are `<input-stem>_lumen.parquet`, `<input-stem>_epithelium.parquet`, and `<input-stem>_densities.npz`; with `--save-mat`, also `<input-stem>_features.mat` at density (block) resolution, or full resolution with `--save-full-mat`. To build a MAT file from an existing label map and tables, pass `--lumen-parquet` and `--epithelium-parquet`. `-j/--jobs` sets worker threads for JPEG2000 decoding and large-image mask, density, component-labeling, and feature-map calculations (default: up to 12 CPUs); specify a higher value to use more.
