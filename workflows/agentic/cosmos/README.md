# Agentic Cosmos

Expand Agentic HDF5 camera streams with NVIDIA Cosmos Transfer 2.5. Cosmos
changes only videos; actions, joint states, `initial_state`, and metadata are
copied from the source demos.

## Setup

```bash
workflows/agentic/cosmos/setup.sh
```

Docker must support GPUs. Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` after
accepting the NVIDIA Cosmos model license.

## Expand

```bash
workflows/agentic/cosmos/run.sh \
  --env scissor_pick_and_place \
  --input recording.hdf5 \
  --output cosmos_expanded.hdf5 \
  --variants 2 \
  --prompt "A photorealistic hospital robot manipulation scene with varied lighting and background" \
  --workspace cosmos_workspace/recording
```

Add `--run-cosmos` to run Docker immediately. Otherwise the command writes a
manifest and prints the manual next steps.

## Manual Steps

```bash
workflows/agentic/cosmos/scripts/export.sh --env scissor_pick_and_place --input recording.hdf5 --workspace cosmos_workspace/recording --variants 2 --prompt "A photorealistic hospital robot manipulation scene"
workflows/agentic/cosmos/scripts/run-docker.sh --env scissor_pick_and_place --manifest cosmos_workspace/recording/manifest.json
workflows/agentic/cosmos/scripts/import.sh --env scissor_pick_and_place --manifest cosmos_workspace/recording/manifest.json --output cosmos_expanded.hdf5
```

Pass `--camera <key>` one or more times to limit the camera streams; otherwise
cameras are auto-detected from `obs/`.

Stop running Cosmos jobs:

```bash
workflows/agentic/cosmos/stop.sh
workflows/agentic/cosmos/stop.sh --force
```
