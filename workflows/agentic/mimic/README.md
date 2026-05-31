# Agentic Mimic

Generate additional HDF5 demos from seed recordings by copying trajectories and
adding small action/state noise. This is a lightweight data expansion helper; it
does not run IsaacLab Mimic or solve new IK trajectories.

## Setup

```bash
workflows/agentic/mimic/setup.sh
```

## Generate Variants

```bash
workflows/agentic/mimic/run.sh \
  --env scissor_pick_and_place \
  --input recording.hdf5 \
  --output recording_mimic.hdf5 \
  --episodes 6 \
  --noise-std 0.01 \
  --include-source \
  --overwrite
```

`--episodes` is the total number of generated demos. `--include-source` also
copies the input demos into the output. Pass `--env` to clamp jittered joints
using dataset metadata from `config/environments/<env>.yaml` and to resolve relative
HDF5 names under `workflows/agentic/runs/<env>/`; use `--seed` for reproducible
perturbations.

Inspect an output file:

```bash
uv --directory workflows/agentic/mimic run i4h-agentic-mimic-inspect \
  --env scissor_pick_and_place \
  recording_mimic.hdf5
```

Stop a running generation:

```bash
workflows/agentic/mimic/stop.sh --env scissor_pick_and_place
workflows/agentic/mimic/stop.sh --env scissor_pick_and_place --force
```
