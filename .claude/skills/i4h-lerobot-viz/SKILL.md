---
name: i4h-lerobot-viz
description: Serve the LeRobot HTML visualizer for a converted dataset in a browser. Use when the user asks to visualize, inspect, or open a LeRobot dataset.
---

# i4h Workflow — LeRobot Viz

## Basics

- Input is a converted LeRobot dataset directory containing `meta/info.json`.
- Use for visual checks after conversion or video augmentation.

## Run

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
DATASET_DIR="${REPO_ROOT}/workflows/agentic/runs/<run>/lerobot/local/<env>"
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/viz_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/viz_state"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/dataset/viz.sh" "${DATASET_DIR}" \
  --state-dir "${RUN_DIR}/viz_state" \
  2>&1 | tee "${RUN_DIR}/logs/viz.log"
```

## Notes

- The dataset path must be absolute. `viz.sh` treats relative paths as Hugging Face repo ids and looks them up under `~/.cache/huggingface/lerobot/<path>`.
- Override `--state-dir` only when the caller provides one.

## Verify

- The visualizer prints a local URL (e.g. `http://127.0.0.1:9090/`).
- Videos and joint timelines load in the browser.

## Final Response

Report dataset path, visualizer URL, stop command, startup failures.
