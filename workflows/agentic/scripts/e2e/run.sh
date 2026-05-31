#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# scripts/e2e/run.sh -> e2e -> scripts -> agentic -> workflows -> repo_root
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_DIR="$REPO_ROOT/workflows/agentic/scripts"
# shellcheck source=config.sh
source "$SCRIPT_DIR/e2e/config.sh"
# shellcheck source=services.sh
source "$SCRIPT_DIR/e2e/services.sh"
# shellcheck source=hdf5.sh
source "$SCRIPT_DIR/e2e/hdf5.sh"
# shellcheck source=stages.sh
source "$SCRIPT_DIR/e2e/stages.sh"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--env <env_id>] [--dry-run] [--skip-*|--no-skip-*] [--run-dir <path>] [--from-stage <stage>] [env_id]

Runs the full agentic e2e smoke pipeline for one env:
setup -> record -> mimic -> annotate/filter -> replay -> convert -> viz ->
finetune -> validate -> summary.

Any env id registered with workflows/agentic/policy/run.sh is supported:
  workflows/agentic/policy/run.sh --list-envs

Default env: scissor_pick_and_place

Options:
  --env <env_id>       Environment id to run.
  --dry-run            Print the resolved plan/configuration without running stages.
  --skip-mimic         Use the recorded HDF5 directly instead of mimic expansion.
  --no-skip-mimic      Run mimic expansion even if SKIP_MIMIC=1 by default/env.
  --skip-annotate      Skip VLM annotation/filtering; use unfiltered demos.
  --no-skip-annotate   Run annotation/filtering even if SKIP_ANNOTATE=1 by default/env.
  --skip-verify-annotate
                      Skip VLM annotation of validation rollouts.
  --no-skip-verify-annotate
                      Run VLM annotation of validation rollouts.
  --skip-replay        Skip replaying a filtered episode in Isaac Sim.
  --no-skip-replay     Run replay even if SKIP_REPLAY=1 by default/env.
  --skip-viz           Skip starting the LeRobot visualizer.
  --no-skip-viz        Start the visualizer even if SKIP_VIZ=1 by default/env.
  --keep-vllm          Leave the annotator vLLM container running after e2e exits.
  --run-dir <path>     Use a specific run directory instead of creating one.
  --from-stage <stage> Resume an existing run from a stage. Requires --run-dir
                       unless stage is setup. Stages: ${E2E_STAGE_NAMES[*]}
  -h, --help           Show this help.
EOF
}

parse_args() {
  e2e_set_defaults

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --env)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --env" >&2
          usage >&2
          exit 2
        fi
        ENV="$2"
        shift 2
        ;;
      --env=*)
        ENV="${1#*=}"
        shift
        ;;
      --dry-run|--dry-run-plan)
        DRY_RUN_PLAN=1
        shift
        ;;
      --skip-viz)
        SKIP_VIZ=1
        shift
        ;;
      --no-skip-viz)
        SKIP_VIZ=0
        shift
        ;;
      --keep-vllm)
        E2E_KEEP_VLLM=1
        shift
        ;;
      --skip-mimic)
        SKIP_MIMIC=1
        shift
        ;;
      --no-skip-mimic)
        SKIP_MIMIC=0
        shift
        ;;
      --skip-annotate)
        SKIP_ANNOTATE=1
        shift
        ;;
      --no-skip-annotate)
        SKIP_ANNOTATE=0
        shift
        ;;
      --skip-verify-annotate)
        SKIP_VERIFY_ANNOTATE=1
        shift
        ;;
      --no-skip-verify-annotate)
        SKIP_VERIFY_ANNOTATE=0
        shift
        ;;
      --skip-replay)
        SKIP_REPLAY=1
        shift
        ;;
      --no-skip-replay)
        SKIP_REPLAY=0
        shift
        ;;
      --run-dir)
        if [[ $# -lt 2 ]]; then
          echo "missing value for --run-dir" >&2
          usage >&2
          exit 2
        fi
        RUN_DIR_OVERRIDE="$2"
        shift 2
        ;;
      --run-dir=*)
        RUN_DIR_OVERRIDE="${1#*=}"
        shift
        ;;
      --from-stage)
        if [[ $# -lt 2 || "$2" == -* ]]; then
          e2e_print_stage_choices >&2
          exit 2
        fi
        FROM_STAGE="$2"
        shift 2
        ;;
      --from-stage=*)
        FROM_STAGE="${1#*=}"
        if [[ -z "$FROM_STAGE" ]]; then
          e2e_print_stage_choices >&2
          exit 2
        fi
        shift
        ;;
      --success-source|--success-source=*)
        echo "[e2e] --success-source has been removed; filtering is automatic (VLM, then simulator labels)." >&2
        usage >&2
        exit 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        echo "unknown flag: $1" >&2
        usage >&2
        exit 2
        ;;
      *)
        ENV="$1"
        shift
        ;;
    esac
  done
}

init_paths() {
  RUNS_ROOT="$REPO_ROOT/workflows/agentic/runs"
  if [[ -n "$RUN_DIR_OVERRIDE" ]]; then
    if [[ -d "$RUN_DIR_OVERRIDE" ]]; then
      RUN_DIR="$(cd "$RUN_DIR_OVERRIDE" && pwd -P)"
    else
      RUN_DIR="$RUN_DIR_OVERRIDE"
    fi
  else
    RUN_DIR="$RUNS_ROOT/e2e_${ENV}_$(date +%Y%m%d_%H%M%S)"
  fi

  RECORD_HDF5="$RUN_DIR/data/recording.hdf5"
  EXPANDED_HDF5="$RUN_DIR/data/expanded.hdf5"
  FILTERED_HDF5="$RUN_DIR/data/filtered.hdf5"
  ANNOTATIONS="$RUN_DIR/annotations/annotations.jsonl"
  REPO_ID="local/${ENV}_e2e"
  DATASET_DIR="$RUN_DIR/lerobot/$REPO_ID"
  CKPT_DIR="$RUN_DIR/checkpoint"
  VERIFY_HDF5="$RUN_DIR/data/verify.hdf5"
  VERIFY_ANNOT="$RUN_DIR/annotations/verify_annotations.jsonl"
  LOGS="$RUN_DIR/logs"
  WORKFLOW_LOG="$LOGS/workflow.log"
  VIZ_STATE_DIR="$RUN_DIR/viz-state"
  FILTER_SOURCE=""
  CKPT=""
}

prepare_run_dir() {
  mkdir -p "$LOGS" "$(dirname "$RECORD_HDF5")" "$(dirname "$ANNOTATIONS")" "$(dirname "$DATASET_DIR")"
  ln -sfn "$RUN_DIR" "$RUNS_ROOT/.latest"
  touch "$WORKFLOW_LOG"
  exec > >(tee -a "$WORKFLOW_LOG") 2>&1

  export HF_LEROBOT_HOME="$RUN_DIR/lerobot"
  export LEROBOT_HOME="$RUN_DIR/lerobot"

  trap e2e_cleanup EXIT
}

main() {
  parse_args "$@"
  e2e_load_env_config
  init_paths
  e2e_validate_stage "$FROM_STAGE"
  if [[ "$FROM_STAGE" != "setup" && -z "$RUN_DIR_OVERRIDE" ]]; then
    echo "[e2e] --from-stage requires --run-dir pointing at the existing run directory" >&2
    exit 2
  fi

  if [[ "$FROM_STAGE" != "setup" ]]; then
    e2e_check_prereqs "$FROM_STAGE"
  fi

  if [[ "$DRY_RUN_PLAN" != "1" ]]; then
    prepare_run_dir
  fi

  echo "[e2e] RUN_DIR=$RUN_DIR"
  echo "[e2e] env=$ENV  stack=$POLICY_STACK  train=$([[ "$TRAIN_SUPPORTED" == "1" ]] && echo "$TRAIN_BIN" || echo "inference-only")"
  echo "[e2e] combined log=$WORKFLOW_LOG"
  echo "[e2e] tail with: tail -f \"$WORKFLOW_LOG\""
  e2e_print_plan

  if [[ "$DRY_RUN_PLAN" == "1" ]]; then
    echo "[e2e] dry run: no stages executed"
    exit 0
  fi

  e2e_run_stages "$FROM_STAGE"
}

main "$@"
