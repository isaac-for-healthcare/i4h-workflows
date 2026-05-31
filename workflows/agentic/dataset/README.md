# Agentic Dataset

Convert arena HDF5 recordings (from `--record-to`) into LeRobot datasets.

## Setup

```bash
workflows/agentic/dataset/setup.sh
```

## Convert

The script auto-detects action/state dims and camera keys from the HDF5
schema. `--env` reads dataset defaults from `config/environments/<env>.yaml`, such as
task text, robot type, joint remapping, modality names, and skipped warmup
frames.

```bash
export HF_LEROBOT_HOME=workflows/agentic/runs/scissor_pick_and_place/lerobot

workflows/agentic/dataset/run.sh \
  --env scissor_pick_and_place \
  --hdf5-path recording.hdf5 \
  --repo-id local/scissor_pick_and_place \
  --overwrite
```

`--repo-id` controls the output dataset name or Hub repo. It defaults to
`local/<env>` when omitted. Override `--task-description` or `--joint-space`
only when needed.

To use another workflow's monolithic env YAML, set `ENVIRONMENT_CONFIG=/path/to/envs.yaml`.

Use `--push-to-hub` to upload after conversion. Use `--cameras` to override
the auto-detected camera keys if multiple cameras are in `obs/`.

Stop a running conversion gracefully, or force kill it:

```bash
workflows/agentic/dataset/stop.sh --env scissor_pick_and_place
workflows/agentic/dataset/stop.sh --env scissor_pick_and_place --force
```
