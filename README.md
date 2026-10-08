# Standalone pythomics feature extraction

Extract per-object lumen and epithelium measurements and block-summed density maps from an integer label-map image. This folder is self-contained and does not import the repository's model/inference code.

## Install

From this directory, install the listed dependencies into your environment:

```bash
python -m pip install -r requirements.txt
```

Image readers are provided by scikit-image. Parquet output is deliberately not required; object-level tables are CSV, density arrays are compressed NPZ, and MAT output is optional.

## CLI

Run from the repository root:

```bash
python -m pythomics_standalone path/to/labels.png --output-dir output --save-mat
```

The default label values are lumen=1, epithelium=2, stroma=4, epithelial-cells=5, stromal-cells=6. Override with `--lumen-label`, `--epithelium-label`, `--stroma-label`, `--epithelial-cells-label`, and `--stromal-cells-label`. `--min-area` sets the minimum connected component size in source pixels (default 2048); `--block-size HEIGHT WIDTH` sets density aggregation blocks (default 20 20).

Epithelium regions include both epithelium and epithelial-cell pixels. Stroma density includes stromal-cell pixels. Connected components are labeled independently in each tissue mask. Tables contain source-pixel area, circularity (`roundness`), and for epithelium average skeleton thickness and epithelial-cell fraction. Density NPZ arrays are block sums, with edge blocks padded with zeros. The optional MAT file includes densities plus feature maps projected across every source-resolution pixel.

Outputs are `<input-stem>_lumen_features.csv`, `<input-stem>_epithelium_features.csv`, and `<input-stem>_densities.npz`; with `--save-mat`, also `<input-stem>_features.mat`.
