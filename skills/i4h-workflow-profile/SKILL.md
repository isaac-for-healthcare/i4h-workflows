---
name: i4h-workflow-profile
description: Profile a workflow with Nsight Systems and apply a measured step-time optimization. Use for slow steps or nsys reports; do not use for rollout success or policy accuracy.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  tags:
    - isaac-for-healthcare
    - i4h
    - simulation
    - profiling
    - nsight
---

# Profile and Optimize a Workflow

## Purpose

Capture a Nsight Systems report of a workflow's steady-state steps, attribute per-step time to the phase that owns it, change the owning code, and prove the change with a second capture and an unchanged rollout outcome.

## Instructions

1. Resolve the base checkout, the workflow, and a live run mode.
2. Capture a baseline report of warm steps only.
3. Summarize per-phase cost and classify each heavy phase.
4. Map the dominant phase to its owning code and propose the smallest change that targets it.
5. After the user accepts the change, apply it, re-capture with the identical command, and compare.
6. Confirm the rollout outcome is unchanged with `i4h-workflow-validate`.

## Resolve live support

Use the maintained repository below or a source selected by the user or trusted project configuration. An inherited environment variable alone does not authorize another source. Honor requested revisions and preserve local changes.

```bash
export I4H_WORKFLOWS_REPO_URL="${I4H_WORKFLOWS_REPO_URL:-https://github.com/isaac-for-healthcare/i4h-workflows}"
I4H_REPO_DIR_NAME="${I4H_WORKFLOWS_REPO_URL%/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*/}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME##*:}"
I4H_REPO_DIR_NAME="${I4H_REPO_DIR_NAME%.git}"
[ -n "$I4H_REPO_DIR_NAME" ] || { echo "Cannot derive a checkout name from I4H_WORKFLOWS_REPO_URL" >&2; exit 2; }
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/i4h_workflows" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/$I4H_REPO_DIR_NAME}"
  [ -d "$ROOT/workflows/i4h_workflows" ] || git clone "$I4H_WORKFLOWS_REPO_URL" "$ROOT" || exit 2
fi
[ -d "$ROOT/workflows/i4h_workflows" ] && [ -x "$ROOT/run.sh" ] || { echo "Incomplete workflow checkout: $ROOT" >&2; exit 2; }
export I4H_WORKFLOWS="$ROOT"
cd "$ROOT" || exit 2
git remote get-url origin || exit 2
git status --short || exit 2
./run.sh list
```

Check that the reported origin is the intended source and review local changes before executing repository scripts. Choose only a workflow and mode shown by `./run.sh list`. Require `./run.sh --help` to list `--nsys-range`; if it does not, this checkout predates the capture range, so stop and report that rather than profiling the whole run. Require `nsys` on `PATH` and record `nsys --version`.

## Capture a baseline

Keep the profiled `run.sh` as this agent's foreground tool call, as in `i4h-workflow-validate`. Fix every input that changes per-step cost and reuse it unchanged for the comparison capture: workflow, mode, checkpoint, device, `--headless` or visible, episode count, step cap, and warmup. A visible viewport renders every step, so never compare a visible capture with a headless one.

`--nsys-range` opens the capture with `cudaProfilerStart` after `--nsys-warmup` completed steps, counted across episodes, and closes it when the run ends. Bound the capture with the episode count and, if needed, a lower `--episode-steps`. Choose the run directory first so the report lands beside the launcher's logs:

```bash
RUN_DIR="$PWD/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
nsys profile --output "$RUN_DIR/profile" --force-overwrite=true \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  ./run.sh <workflow> --<mode> --episodes 1 --attempts 1 \
  --run-dir "$RUN_DIR" --nsys-range --nsys-warmup=50
```

The default warmup is 20 steps. Raise it when the first policy requests or the first sensor renders are still slow after that. Use the `--nsys-warmup=N` form so the value stays attached to the flag. Require `$RUN_DIR/profile.nsys-rep` and the `nsys capture range opened` line in `$RUN_DIR/i4h_arena.log`. Add `osrt` to `--trace` only when a CPU-bound phase needs OS-runtime evidence, because it enlarges the report.

## Summarize

Run the exports in [references/nsys-analysis.md](references/nsys-analysis.md) and read them in this order:

1. `nvtx_pushpop_sum`: wall time per phase. `Avg (ns)` is the per-step cost; treat the largest phase as the target.
2. `nvtx_gpu_proj_sum`: GPU time each range launched. Range time far above projected GPU time means that phase is CPU-bound or waiting.
3. `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`: which kernels and copies the GPU time goes to.
4. `cuda_api_sum` and the `nsys analyze` rules: synchronizing calls and GPU idle gaps.
5. The per-process query: how GPU time splits between the simulator and any backend process.

Report a per-step table with phase, mean milliseconds, share of the step, GPU-projected milliseconds, and classification. Quote numbers from the CSVs; do not estimate them from the timeline.

## Map phases to owners

| NVTX range | Covers | Owning code |
|---|---|---|
| `tick` | `engine.tick`: every active Task, including policy inference and any remote policy round trip | the Task under `tasks/<project>/i4h_tasks/<project>/` named by `./run.sh show` |
| `env.step` | physics sub-steps, sensor updates such as fluoroscopy, and the renders between them | the Scene under `arena/i4h_arena/scenes/`, its manifest, and sensors under `arena/i4h_arena/sensors/` |
| `record` | HDF5 capture, including reading camera frames back to the host | the recorder under `arena/i4h_arena/recording/` |

Time between ranges inside a step belongs to reset handling, sensor windows, and publishing in `arena/i4h_arena/runner.py`. Find it as step-to-step time not covered by the three ranges.

`nsys` wraps the whole launcher, so a remote policy backend started by `run.sh` is inside the capture as its own process. Its GPU work appears in the per-process query, not under the simulator's ranges, and the simulator's wait for it appears as CPU time inside `tick`. Attribute that to the backend and its `backend-*.log`.

Ranges from Kit or other libraries can appear beside these. Name them, but optimize only code this repository owns.

## Optimize

Propose one change per iteration, aimed at the dominant phase, with the expected effect stated in milliseconds per step. Get the user's acceptance before editing. Common findings and their usual fixes are listed in the reference.

These changes alter what the workflow means, not only how fast it runs. Treat them as out of scope unless the user explicitly asks, and never apply them silently:

- Raising the Scene manifest's step cap, or changing a success rule.
- Changing the physics time step, decimation, solver iterations, or a camera resolution, rate, or observation that a trained policy consumes.
- Dropping recorded fields or cameras from the HDF5 contract in `common/i4h_common/episode.py`.

Follow the repository change discipline: a Task reads `ctx.scene` and writes `ctx.act` and never calls `env.step`, and policy packages stay out of arena. Add or update CPU tests for any changed Task or graph behavior. When a phase needs finer attribution, add a `phase(...)` range from `i4h_arena.profiling` around the suspect call with `enabled=self.capture.enabled`, so it stays inert outside a capture.

## Verify

Re-capture with the identical command, regenerate the reports into a separate directory, and compare per-phase means before and after. Report a change as an improvement only when the target phase drops by more than its run-to-run spread. If the result is unclear, take one more capture of each version.

Then run the same command without `nsys` and `--nsys-range` through `i4h-workflow-validate`, and require the same episode success as before the change. For a Scene or sensor change, also perform the visible simulator check that the repository requires.

## Troubleshooting

- **Empty report:** the run ended before warmup did. Lower `--nsys-warmup` or raise `--episodes`.
- **`--nsys-range needs a CUDA device`:** the simulator has no CUDA device; select one with `--device`.
- **The whole run is captured:** `--capture-range=cudaProfilerApi` is missing from the `nsys` command.
- **A report too large to open:** lower the episode count or `--episode-steps` instead of dropping traces.

## Prerequisites

Require Nsight Systems 2024.5 or newer with `nsys` on `PATH`, a checkout whose `run.sh` lists `--nsys-range`, synced simulator assets, a CUDA device, and any backend or checkpoint the selected run mode declares.

## Limitations

The capture covers processes `run.sh` starts. A backend reached through `--policy-endpoint` runs elsewhere and appears only as wait time inside `tick`. Data conversion and training are out of scope.

## Examples

- `Profile surgical_reach_psm in rule-based mode and tell me where the step time goes.` → capture warm steps, then report the per-phase table and its classification.
- `Endoluminal navigation is slow; optimize it.` → capture, find the dominant phase, propose one owned change, re-capture, compare, and confirm the rollout with `i4h-workflow-validate`.

## Completion gate

Report the workflow, mode, device, headless or visible, `nsys` version, warmup, captured step count, and report paths for both captures. Include the before-and-after per-phase table, the change made with its owning file, the measured improvement in milliseconds per step, the validation outcome, and any phase left unexplained.
