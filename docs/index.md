# TAPS

TAPS deploys a trained prostate MRI segmentation model as a Python package and command-line application. It accepts a NIfTI image and writes a binary NIfTI mask aligned with the original input image.

## Install

```console
pip install lavlab-taps
```

## CLI

```console
taps segment input_image.nii.gz prostate_mask.nii.gz
```

TAPS checks volume, geometry, border contact, slice continuity, enclosed holes, and connected components. Abnormal outputs receive an `_abnormal` suffix. Pass `--exact` to keep the requested output name. A mask with multiple components in a slice also produces a `_cleaned` NIfTI with that slice blanked.

Blank inferred masks are not written and cause the CLI to exit with status `1`.

## Python API

```python
from taps import segment

segment("input_image.nii.gz", "prostate_mask.nii.gz", checkpoint=None)
```

See the [API Reference](api.md) for full details.
