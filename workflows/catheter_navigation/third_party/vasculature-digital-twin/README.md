# vasculature-digital-twin

Standalone CT ingestion and vessel digital-twin preprocessing package extracted from the i4h catheter workflow.

## What it includes

- DICOM and NIfTI CT loading to HU volumes
- HU to mu preprocessing and cache serialization
- Vessel segmentation (TotalSegmentator with HU-threshold fallback)
- Centerline graph extraction via skeletonization

## Install

```bash
pip install -e .
```

With all optional dependencies:

```bash
pip install -e ".[all]"
```

## CLI

Preprocess CT into cache artifacts:

```bash
vdt-preprocess-ct --nifti /path/to/ct.nii.gz --output-dir /tmp/ct_cache
```

Segment vessels and extract centerline graph:

```bash
vdt-segment-vessels --ct-dir /tmp/ct_cache
```

## Output artifacts

- `mu_volume.npy`
- `metadata.json`
- `hu_volume.npy` (if enabled)
- `vessel_mask.npy`
- `centerline_points_mm.npy`
- `centerline_edges.npy`
- `centerline_radii_mm.npy`
