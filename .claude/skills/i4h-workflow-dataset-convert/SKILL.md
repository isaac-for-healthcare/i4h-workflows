---
name: i4h-workflow-dataset-convert
description: Convert an agentic HDF5 recording into a LeRobot dataset (parquet + meta + videos). Use when the user asks to convert HDF5, prepare for training, or export to LeRobot.
---

# i4h Workflow — Convert Dataset

## Basics

- Use the same `--env` that produced the HDF5.
- Env YAML supplies robot, task, camera, modality, and converter defaults.
- Output goes to `HF_LEROBOT_HOME/<repo-id>`.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ENV_ID=scissor_pick_and_place
HDF5_PATH="${REPO_ROOT}/workflows/agentic/runs/<run>/data/demo.hdf5"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/convert_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"
export HF_LEROBOT_HOME="${RUN_DIR}/lerobot"

"${REPO_ROOT}/workflows/agentic/dataset/run.sh" \
  --env "${ENV_ID}" \
  --hdf5-path "${HDF5_PATH}" \
  --repo-id "local/${ENV_ID}" \
  --video-codec h264 \
  --overwrite \
  2>&1 | tee "${RUN_DIR}/logs/convert.log"
```

## Notes

- `--video-codec h264` is required. The converter's default AV1 codec breaks GR00T's `decord` video reader at finetune time.
- Scissor SO-ARM generates `meta/modality.json` from YAML splits and does not need `dataset.modality_template_path`.
- G1 locomanip and assemble-trocar use `dataset.modality_template_path` from the env YAML.
- All camera streams are resized to the env YAML `policy.image_size` (override with `--image-size H W`), normalizing mixed-resolution cameras (e.g. head cam + overview cam) to the one size the modality config expects.

## Verify

- `${HF_LEROBOT_HOME}/local/${ENV_ID}/meta/info.json` exists.
- Log reports the saved episode count.
- Per-episode video files are present under `${HF_LEROBOT_HOME}/local/${ENV_ID}/`.

## Final Response

Report source HDF5, dataset path, repo id, episode count, skipped or failed episodes.
