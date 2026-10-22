---
name: i4h-workflow-dataset-annotate
description: Grade or filter Isaac Lab HDF5 episodes of i4h workflows with an OpenAI-compatible vision model. Use for visual success labels; do not use for replay, policy evaluation, or recordings without frames.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - dataset
    - annotation
    - vlm
---

# Annotate Workflow Recordings

## Purpose

Grade sampled camera frames of recorded episodes against a natural-language success criterion with `tools/annotator`, keeping VLM labels separate from simulator success.

## Instructions

1. Run the checkout resolver and select one HDF5 and one criterion.
2. Test camera sampling and the vision endpoint.
3. Run grading and optional filtering on every selected episode.
4. Compare verdict and output counts.

## Resolve input and criterion

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
```

Treat the resolver above as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

Use the explicit or current-chain HDF5. "All recorded episodes" means every episode in that selected file, not every historical run. Inspect it with `uv run --project tools/dataset i4h-dataset inspect <file>`, which lists each demo's cameras (`obs/<camera>` frames). Identify its scene from the conversation or the run directory; `record_demos.py` recordings also name it as `env_name` in the `data` group's `env_args` attribute.

Use the user's explicit success criterion when supplied. Otherwise read the scene manifest `arena/i4h_arena/scenes/manifest/<scene>.yaml`: its `annotate:` block, when present, gives the `criterion`, `cameras`, `judge`, and `frames` for every recording of that scene, and `scripts/e2e/run.sh` passes exactly those. Without one, use the `prompt` (or `summary`) of the workflow's policy task manifest, `tasks/<stack>/i4h_tasks/<stack>/manifest/<task>.yaml` (usually named after the workflow), as `scripts/e2e/run.sh` does, or the visible success condition on the workflow's `docs/workflows/<specialty>.md` page. Phrase placement success as the object reaching and remaining at its target, not as the robot continuing to hold it.

Choose `--judge terminal` (the default) when success is a state the final frames show, such as an object in a tray. Choose `--judge trajectory` when success is a motion, such as a probe sweep, whose end looks much like its start. Repeat `--camera`, or pass a comma-separated list, when no single camera shows the task; without it the annotator grades only the first camera by name.

## Resolve the endpoint

Use a caller-provided OpenAI-compatible vision endpoint/model first. Local Agent exposes that configuration as `I4H_AGENT_VL_BASE_URL`, `I4H_AGENT_VL_MODEL`, and either `I4H_AGENT_VL_API_KEY` or `I4H_AGENT_API_KEY`. Map those generic agent variables to the annotator without printing the credential:

```bash
VLM_ARGS=()
if [ -n "${I4H_AGENT_VL_BASE_URL:-}" ] && [ -n "${I4H_AGENT_VL_MODEL:-}" ]; then
  I4H_VLM_URL="${I4H_AGENT_VL_BASE_URL%/}"
  case "$I4H_VLM_URL" in */v1) ;; *) I4H_VLM_URL="$I4H_VLM_URL/v1" ;; esac
  export I4H_VLM_URL
  export OPENAI_API_KEY="${I4H_AGENT_VL_API_KEY:-${I4H_AGENT_API_KEY:-EMPTY}}"
  VLM_ARGS=(--model "$I4H_AGENT_VL_MODEL")
fi
```

If no caller-provided endpoint/model is available, start the repository's local service:

```bash
tools/annotator/scripts/vllm.sh ensure
```

Record whether this invocation started it. Do not hard-code a model name in the skill; use the CLI/service defaults unless the user supplies one.

## Dry-run sampling when needed

```bash
uv run --project tools/annotator i4h-annotator \
  --task "<success criterion>" \
  --dry-run \
  offline /absolute/path/to/recording.hdf5
```

Use this to verify cameras and sampled frames without transmitting images.

## Grade and filter

```bash
RUN_DIR="$(pwd)/runs/<workflow>/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
uv run --project tools/annotator i4h-annotator \
  --task "<success criterion>" \
  "${VLM_ARGS[@]}" \
  offline /absolute/path/to/recording.hdf5 \
  --write
```

Global options go before `offline`: add `--base-url`, `--model`, `--camera`, `--judge`, or `--frames` only when selected. `offline` takes `--write`, `--filter`, and `--node`. `--write` stores `vlm_success` and `vlm_success_reason` on each demo. Add `--filter "$RUN_DIR/filtered.hdf5"` only when filtering was requested; a summarize-only prompt must grade all episodes without requiring at least one success. `--node` grades one node's frames and works only for recordings that carry per-node `segments`, which only the removed `run.sh` launcher wrote; Isaac Lab recordings have none. Keep credentials in environment variables; never print them.

Stop the local VLM only if this invocation started it:

```bash
tools/annotator/scripts/vllm.sh stop
```

## Verify

Inspect the annotator summary. If filtering was requested, also inspect the filtered file:

```bash
uv run --project tools/dataset i4h-dataset inspect "$RUN_DIR/filtered.hdf5"
```

Require a verdict for every selected episode and reconcile pass/fail counts plus filtered counts when applicable. Episodes with identical sampled frames, such as `tools/mimic` variants of one source demo, receive one verdict; the summary line counts graded and reused verdicts separately, and `--write` records the episode it came from in `vlm_success_from` (`--filter` rewrites it to the renumbered name). The filtered file keeps the source's `format_version` and `env_args`, so it still replays and converts. Treat endpoint errors, absent cameras, partial writes, and unexplained zero-episode output as failure. An all-failure verdict set is a valid completed grading run for summarize-only prompts; it is not a valid filtered dataset.

The annotator warns about demos that predate `recording_version` 1 only in recordings from the removed `run.sh` launcher, whose final frame may show the next episode; re-record those rather than grade their final frames. Isaac Lab's recorder writes an episode before resetting it.

## Troubleshooting

Check camera sampling before endpoint/authentication errors. Never accept partial writes or a filtered file with an unexplained zero count. When simulator-successful episodes all fail, dump the sampled frames from each camera before changing the criterion: a tiny or occluded subject calls for another `--camera`, and a motion task graded on its end state calls for `--judge trajectory`. A camera scene recorded without `--enable_cameras` has no frames to grade; endoluminal recordings keep `obs/fluoroscopy` either way.

## Prerequisites

Require a readable HDF5 recording with camera frames and, unless dry-running, a reachable OpenAI-compatible vision endpoint.

## Limitations

Visual grading cannot recover missing frames or prove simulator state that is not visible. The annotator grades recorded files only; it has no live mode.

## Examples

- `Run annotation on all recorded episodes and summarize.` → select the current HDF5, take the criterion from its scene's `annotate:` block or task prompt, grade every episode with `--write`, and report pass/fail counts.

## Completion gate

Report source HDF5, selected criterion and its origin, camera, judge, model, endpoint origin, graded and reused pass/fail counts, filtered path/count when requested, dry-run result if used, and local-service cleanup.
