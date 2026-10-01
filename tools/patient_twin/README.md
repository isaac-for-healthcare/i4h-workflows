# Patient twin preparation

This tool calls `patient_digital_twin.main` and returns the `patient_twin.yaml`
bundle consumed by `endoluminal_navigation`. Model inference, named mesh
extraction, centerlines, CT conversion, and USD output belong to that library.

## Install

`third_party/setup.sh` pins the patient library commit that adds this pipeline.
To test these paired branches before the library commit is published, use a
local checkout as the source:

```bash
I4H_DIGITAL_TWIN_URL=/absolute/path/to/i4h-digital-twin \
  ./third_party/setup.sh patient-twin
uv sync --project tools/patient_twin --extra dev
```

After that commit is available upstream, omit the URL override. Set
`I4H_DIGITAL_TWIN_REF` to test a different commit. The checkout is kept at
`third_party/i4h-digital-twin`; existing simulation dependency checkouts are
independent of it. Install NV-Segment or NV-Generate and their model weights in
a separate compatible GPU Python environment; `--python` selects it.

## Segment a CT

```bash
./tools/patient_twin/run.sh --source nvsegment \
  --input /path/to/s0011/ct.nii.gz --classes aorta \
  --bundle-root /path/to/NV-Segment-CTMR/NV-Segment-CTMR \
  --python /path/to/model-env/bin/python \
  --patient-id s0011 --output ./data/patient_twins/s0011_aorta

./run.sh endoluminal_navigation --mode validate_fluoroscopy --episodes 1 \
  --patient-twin ./data/patient_twins/s0011_aorta/patient_twin.yaml \
  --record verify.hdf5
```

The input is a 3D CT NIfTI in HU; supplied dataset masks are not read.
`--classes` takes space- or comma-separated names. Output must be a new directory.
The default format is `workflow`, with CT, attenuation, vessel mask, centerlines,
anatomy USD, and a manifest that records their LPS-to-world placement. Missing
centerlines are calculated; existing ones are reused. Axis-oblique CT must be
resampled before inference. The attenuation default is `--hu-to-mu interventional`.

For generation, use `--source nvgenerate --source-root /path/to/NV-Generate-CTMR`
and omit `--input`. For standalone USD, use the library's
`python -m patient_digital_twin.main --format usd` entry point.

The isolated `patient_digital_twin.legacy_ct` module implements the temporary CT
artifact compatibility layer. The workflow carries no duplicate CT processing,
segmentation fallback, or anatomy USD writer.

Run the tool's CPU contract tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --project tools/patient_twin --extra dev pytest tools/patient_twin/tests
```
