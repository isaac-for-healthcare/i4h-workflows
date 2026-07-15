#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

e2e_set_defaults() {
  ENV="${ENV:-scissor_pick_and_place}"
  DRY_RUN_PLAN="${DRY_RUN_PLAN:-0}"
  SKIP_VIZ="${SKIP_VIZ:-1}"
  SKIP_MIMIC="${SKIP_MIMIC:-0}"
  SKIP_ANNOTATE="${SKIP_ANNOTATE:-1}"
  SKIP_VERIFY_ANNOTATE="${SKIP_VERIFY_ANNOTATE:-1}"
  SKIP_REPLAY="${SKIP_REPLAY:-1}"
  RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-}"
  FROM_STAGE="${FROM_STAGE:-setup}"

  RECORD_EPISODES="${RECORD_EPISODES:-3}"
  RECORD_MAX_ATTEMPTS="${RECORD_MAX_ATTEMPTS:-3}"
  MIMIC_EPISODES="${MIMIC_EPISODES:-3}"
  MIMIC_NOISE_STD="${MIMIC_NOISE_STD:-0.01}"
  ANNOTATION_SAMPLE_FRAMES="${ANNOTATION_SAMPLE_FRAMES:-5}"
  VERIFY_EPISODES="${VERIFY_EPISODES:-1}"
  VERIFY_MAX_ATTEMPTS="${VERIFY_MAX_ATTEMPTS:-1}"
  FINETUNE_MAX_STEPS="${FINETUNE_MAX_STEPS:-500}"
  FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-500}"
  FINETUNE_GPUS="${FINETUNE_GPUS:-1}"
  VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-600}"
  VIDEO_CODEC="${VIDEO_CODEC:-h264}"
  ARENA_HEADLESS="${ARENA_HEADLESS:-0}"
  ARENA_RENDERING_MODE="${ARENA_RENDERING_MODE:-performance}"
}

e2e_env_policy_model_repo() {
  local env="$1"
  local config="$REPO_ROOT/workflows/agentic/config/environments/${env}.yaml"
  [[ -f "$config" ]] || return 0

  awk '
    /^[^[:space:]].*:/ {
      in_policy = ($1 == "policy:")
    }
    in_policy && /^[[:space:]]+model_repo:[[:space:]]*/ {
      sub(/^[[:space:]]+model_repo:[[:space:]]*/, "")
      sub(/[[:space:]]+#.*$/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      gsub(/^["\047]|["\047]$/, "")
      if ($0 != "null") print
      exit
    }
  ' "$config"
}

e2e_env_policy_train_module() {
  local env="$1"
  local config="$REPO_ROOT/workflows/agentic/config/environments/${env}.yaml"
  [[ -f "$config" ]] || return 0

  python3 -c "
import yaml
with open('$config') as f:
    policy = (yaml.safe_load(f) or {}).get('policy') or {}
value = policy.get('train_module', '__default__')
if value is None:
    print('')
else:
    print(value)
"
}

e2e_load_env_config() {
  POLICY_STACK="$(workflows/agentic/policy/run.sh --list-envs 2>/dev/null \
    | awk -v e="$ENV" '$1 == e { print $2; exit }')"
  if [[ -z "$POLICY_STACK" ]]; then
    echo "[e2e] env '$ENV' is not registered with workflows/agentic/policy/run.sh" >&2
    echo "[e2e] available envs:" >&2
    workflows/agentic/policy/run.sh --list-envs >&2 || true
    exit 2
  fi

  POLICY_TRAIN_MODULE="$(e2e_env_policy_train_module "$ENV")"
  TRAIN_SUPPORTED=1
  if [[ -z "$POLICY_TRAIN_MODULE" ]]; then
    TRAIN_SUPPORTED=0
  fi
  TRAIN_BIN="i4h-agentic-${POLICY_STACK//_/-}-train"
  POLICY_MODEL_REPO="$(e2e_env_policy_model_repo "$ENV")"
  STACK_TRAIN_EXTRA=()
  ENV_TRAIN_EXTRA=()

  case "$POLICY_STACK" in
    gr00t_n15)
      POLICY_LOAD_WAIT_SECONDS="${POLICY_LOAD_WAIT_SECONDS:-60}"
      FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-32}"
      ;;
    gr00t_n16|gr00t_n17)
      POLICY_LOAD_WAIT_SECONDS="${POLICY_LOAD_WAIT_SECONDS:-120}"
      FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-8}"
      ;;
    openpi_pi0)
      POLICY_LOAD_WAIT_SECONDS="${POLICY_LOAD_WAIT_SECONDS:-120}"
      FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-1}"
      ;;
    *)
      echo "[e2e] WARN: unknown stack '$POLICY_STACK'; using conservative smoke defaults"
      POLICY_LOAD_WAIT_SECONDS="${POLICY_LOAD_WAIT_SECONDS:-120}"
      FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-1}"
      ;;
  esac

  if [[ "$TRAIN_SUPPORTED" == "1" && "$POLICY_STACK" == gr00t_* && -n "$POLICY_MODEL_REPO" ]]; then
    STACK_TRAIN_EXTRA+=(--base-model-path "$POLICY_MODEL_REPO")
  fi

  if [[ -z "${POLICY_HEALTH_PORT:-}" ]]; then
    POLICY_HEALTH_PORT="$(python3 -c "
import sys, yaml
with open('${REPO_ROOT}/workflows/agentic/config/environments/${ENV}.yaml') as f:
    cfg = yaml.safe_load(f) or {}
port = (cfg.get('policy') or {}).get('health_port')
if port is None:
    sys.exit(1)
print(port)
")"
  fi
  if [[ -z "${POLICY_HEALTH_PORT}" ]]; then
    echo "[e2e] ERROR: policy.health_port is not set in config/environments/${ENV}.yaml" >&2
    exit 2
  fi
}
