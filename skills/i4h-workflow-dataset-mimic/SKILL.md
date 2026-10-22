---
name: i4h-workflow-dataset-mimic
description: Expand HDF5 demonstrations with action jitter, optionally per node segment of older recordings. Use for synthetic variants; do not use to collect data, alter state directly, or generate new images.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - augmentation
    - hdf5
---

# Mimic Workflow Demonstrations

## Purpose

Clone successful source episodes and perturb their recorded actions with `tools/mimic`.

## Instructions

1. Run the checkout resolver and select the source HDF5.
2. Inspect the successful episodes.
3. Run mimic to generate the requested additional action-jitter variants.
4. Compare exact counts and preserved contracts.

## Resolve and inspect

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/arena/i4h_arena/scenes/manifest" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/arena/i4h_arena/scenes/manifest" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT"
find runs -name '*.hdf5' -type f -printf '%T@ %p\n' | sort -nr | head
HDF5_PATH=/absolute/path/to/source.hdf5
uv run --project tools/dataset i4h-dataset inspect "$HDF5_PATH"
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit or current-chain recording; otherwise select the newest plausible HDF5 and state that choice. Require at least one episode marked `ok` (its `success` attribute): the policy runner records every episode with its flag, and `record_demos.py` writes only successful ones.

Node-scoped jitter needs per-node `segments`, which only recordings from the removed `run.sh` launcher carry; Isaac Lab recordings have none. If the user asks for it, check with `i4h-dataset inspect "$HDF5_PATH" --segments` and report that it is unavailable when no segment is listed.

## Expand

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
uv run --project tools/mimic i4h-mimic \
  "$HDF5_PATH" "$RUN_DIR/expanded.hdf5" \
  --episodes <additional-count> \
  --noise 0.01 \
  --seed <seed> \
  --include-source \
  --successful-only
```

`--episodes` is the number of generated variants, not the final total; with `--include-source` the output holds the successful sources first, then the variants, renumbered from `demo_0`. Variants cycle through the sources. `--noise` is the standard deviation in action units. Add `--node <node-id>` only when requested and the recording carries that node's segments. Preserve the default seed `0` unless the user supplies another.

The tool jitters the `actions` dataset only. `processed_actions`, `states`, `initial_state`, observations, and camera frames are copied from the source demo, as are the file's `format_version` and the `env_args` naming its environment, so the output still replays with `replay_demos.py`. Each variant records `source_demo`, `variant_index`, `mimic_noise`, `mimic_node`, and `mimic_frames_jittered`, and inherits its source's `success` flag (`True` when the source has none) without re-simulation.

## Verify and compose

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/expanded.hdf5"
```

Confirm the source count, generated count, final total, action width, and success flags. Reject zero generated episodes. To check that a variant still succeeds, replay it with `--validate_success_rate` through `i4h-workflow-dataset-replay`; its stored frames show the source, so VLM grading reuses the source's verdict.

If the same prompt requests visualization, continue with `i4h-workflow-dataset-convert`, then `i4h-lerobot-viz`; raw HDF5 is not a LeRobot visualizer input.

## Troubleshooting

`has no successful demos to clone` means no source episode carries `success`; drop `--successful-only` only if the user accepts failed sources. `no demo carries a segment for node` means the recording has no segments for that node; report the nodes `--segments` lists, if any.

## Prerequisites

Require `./setup.sh` to have built the `tools/mimic` and `tools/dataset` environments and a readable HDF5 recording with at least one successful episode.

## Limitations

The tool adds no independent state noise, does not rerun physics or re-label success, and does not synthesize camera frames. It is not Isaac Lab Mimic's subtask-based data generation.

## Examples

- `Mimic 3 more episodes and visualize my dataset.` → generate three variants of the current recording with its sources included, convert the expanded HDF5, then launch LeRobot visualization.

## Completion gate

Report input and output paths, source and generated counts, final total, action-noise value, seed, optional node, the inherited-success and frame-copy limitations, and the conversion and visualizer result when requested.
