#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

E2E_STAGE_NAMES=(setup record mimic annotate replay convert viz finetune validate summary)
E2E_STAGE_FUNCS=(
  stage_setup
  stage_record
  stage_mimic
  stage_annotate
  stage_replay
  stage_convert
  stage_viz
  stage_finetune
  stage_validate
  stage_summary
)

e2e_stage() {
  echo
  echo "==================== [$1] $2 ===================="
}

e2e_stage_index() {
  local stage="$1"
  local i
  for i in "${!E2E_STAGE_NAMES[@]}"; do
    if [[ "${E2E_STAGE_NAMES[$i]}" == "$stage" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

e2e_validate_stage() {
  local stage="$1"
  if ! e2e_stage_index "$stage" >/dev/null; then
    echo "[e2e] unknown stage '$stage'" >&2
    echo "[e2e] valid stages: ${E2E_STAGE_NAMES[*]}" >&2
    exit 2
  fi
}

e2e_print_stage_choices() {
  echo "[e2e] --from-stage needs one of:"
  local stage
  for stage in "${E2E_STAGE_NAMES[@]}"; do
    echo "  $stage"
  done
}

e2e_require_file() {
  local path="$1"
  local stage="$2"
  if [[ ! -f "$path" ]]; then
    echo "[e2e] cannot resume from '$FROM_STAGE': stage '$stage' needs missing file: $path" >&2
    exit 2
  fi
}

e2e_require_dir() {
  local path="$1"
  local stage="$2"
  if [[ ! -d "$path" ]]; then
    echo "[e2e] cannot resume from '$FROM_STAGE': stage '$stage' needs missing directory: $path" >&2
    exit 2
  fi
}

e2e_has_checkpoint() {
  local candidates=()
  shopt -s nullglob
  candidates=("$CKPT_DIR"/checkpoint-* "$CKPT_DIR"/*/*/[0-9]*)
  shopt -u nullglob
  (( ${#candidates[@]} > 0 ))
}

e2e_check_prereqs() {
  local stage="$1"
  if [[ "$stage" != "setup" && ! -d "$RUN_DIR" ]]; then
    echo "[e2e] cannot resume from '$stage': run directory does not exist: $RUN_DIR" >&2
    exit 2
  fi

  case "$stage" in
    setup|record)
      ;;
    mimic)
      e2e_require_file "$RECORD_HDF5" "$stage"
      ;;
    annotate)
      if [[ "$SKIP_MIMIC" == "1" ]]; then
        e2e_require_file "$RECORD_HDF5" "$stage"
      else
        e2e_require_file "$EXPANDED_HDF5" "$stage"
      fi
      ;;
    replay|convert)
      e2e_require_file "$FILTERED_HDF5" "$stage"
      ;;
    viz|finetune)
      e2e_require_dir "$DATASET_DIR/meta" "$stage"
      ;;
    validate)
      if [[ "$TRAIN_SUPPORTED" == "0" ]]; then
        :
      elif ! e2e_has_checkpoint; then
        echo "[e2e] cannot resume from 'validate': no checkpoint found under $CKPT_DIR" >&2
        exit 2
      fi
      ;;
    summary)
      e2e_require_dir "$LOGS" "$stage"
      ;;
  esac
}

e2e_stage_description() {
  local stage="$1"
  case "$stage" in
    setup) echo "Setup component venvs/checkouts if needed." ;;
    record) echo "Start the base policy daemon and record ${RECORD_EPISODES} successful policy episodes." ;;
    mimic)
      if [[ "$SKIP_MIMIC" == "1" ]]; then
        echo "Skip mimic expansion and use the recording directly."
      else
        echo "Mimic-expand the recording with ${MIMIC_EPISODES} variants per source, plus source demos."
      fi
      ;;
    annotate)
      if [[ "$SKIP_ANNOTATE" == "1" ]]; then
        echo "Skip annotation/filtering and use unfiltered demos."
      else
        echo "Annotate expanded episodes and filter to successful samples."
      fi
      ;;
    replay)
      if [[ "$SKIP_REPLAY" == "1" ]]; then
        echo "Skip replaying a filtered episode."
      else
        echo "Replay one random filtered episode in Isaac Sim."
      fi
      ;;
    convert) echo "Convert the filtered HDF5 to LeRobot using ${VIDEO_CODEC}." ;;
    viz)
      if [[ "$SKIP_VIZ" == "1" ]]; then
        echo "Skip the LeRobot visualizer."
      else
        echo "Start the LeRobot visualizer."
      fi
      ;;
    finetune)
      if [[ "$TRAIN_SUPPORTED" == "0" ]]; then
        echo "Skip fine-tuning; ${ENV} is inference-only."
      else
        echo "Fine-tune a smoke checkpoint for ${FINETUNE_MAX_STEPS} steps via $TRAIN_BIN."
      fi
      ;;
    validate)
      if [[ "$TRAIN_SUPPORTED" == "0" ]]; then
        echo "Skip checkpoint validation; ${ENV} is inference-only."
      elif [[ "$SKIP_VERIFY_ANNOTATE" == "1" ]]; then
        echo "Validate the new checkpoint with ${VERIFY_EPISODES} rollout episode(s), without VLM verify annotation."
      else
        echo "Validate the new checkpoint with ${VERIFY_EPISODES} rollout episode(s), then VLM-annotate verification."
      fi
      ;;
    summary) echo "Write a summary report." ;;
  esac
}

e2e_print_stage_plan() {
  local from_stage="$1"
  local start_index
  start_index="$(e2e_stage_index "$from_stage")"

  local i number=1 stage
  for ((i = start_index; i < ${#E2E_STAGE_NAMES[@]}; i++)); do
    stage="${E2E_STAGE_NAMES[$i]}"
    printf '  %d. [%s] %s\n' "$number" "$stage" "$(e2e_stage_description "$stage")"
    number=$((number + 1))
  done
}

e2e_print_plan() {
  cat <<EOF

[e2e] Plan
EOF
  e2e_print_stage_plan "$FROM_STAGE"
  cat <<EOF

[e2e] Configuration
  env / stack:                 $ENV / $POLICY_STACK
  policy model repo:           ${POLICY_MODEL_REPO:-"(not configured)"}
  run dir:                     $RUN_DIR
  record episodes/attempts:    ${RECORD_EPISODES}/${RECORD_MAX_ATTEMPTS}
  mimic episodes/noise:        ${MIMIC_EPISODES}/${MIMIC_NOISE_STD}
  annotation sample frames:    $ANNOTATION_SAMPLE_FRAMES
  filter source:               $(e2e_filter_plan_description)
  skipped stages:              mimic=$SKIP_MIMIC annotate=$SKIP_ANNOTATE verify_annotate=$SKIP_VERIFY_ANNOTATE replay=$SKIP_REPLAY viz=$SKIP_VIZ
  train support:                $([[ "$TRAIN_SUPPORTED" == "1" ]] && echo "$TRAIN_BIN" || echo "inference-only")
  finetune steps/save/bs/gpus: ${FINETUNE_MAX_STEPS}/${FINETUNE_SAVE_STEPS}/${FINETUNE_BATCH_SIZE}/${FINETUNE_GPUS}
  verify episodes/attempts:    ${VERIFY_EPISODES}/${VERIFY_MAX_ATTEMPTS}
  arena render:                 $([[ "$ARENA_HEADLESS" == "1" ]] && echo "headless cameras (${ARENA_RENDERING_MODE})" || echo "visible")
  policy ready timeout:        ${POLICY_LOAD_WAIT_SECONDS}s
  vLLM ready timeout:          ${VLLM_READY_TIMEOUT_SECONDS}s
  viz:                         $([[ "$SKIP_VIZ" == "1" ]] && echo "skip" || echo "start")
  from stage:                  $FROM_STAGE

[e2e] To monitor
  tail -f "$WORKFLOW_LOG"

EOF
}

e2e_filter_plan_description() {
  if [[ "$SKIP_ANNOTATE" == "1" ]]; then
    echo "unfiltered (annotation skipped)"
  else
    echo "automatic (VLM, then simulator success labels)"
  fi
}

e2e_arena_rollout_args() {
  if [[ "$ARENA_HEADLESS" == "1" ]]; then
    printf '%s\0' --headless --enable_cameras --rendering_mode "$ARENA_RENDERING_MODE"
  fi
}

stage_setup() {
  e2e_stage 1 "setup"
  workflows/agentic/setup.sh 2>&1 | tee "$LOGS/01_setup.log"
}

stage_record() {
  e2e_stage 2 "record ${RECORD_EPISODES} successful base-policy episodes"
  local arena_args=()
  mapfile -d '' -t arena_args < <(e2e_arena_rollout_args)
  e2e_start_policy "$LOGS/02a_policy.log"
  workflows/agentic/arena/run.sh --env "$ENV" --episodes "$RECORD_EPISODES" --max-attempts "$RECORD_MAX_ATTEMPTS" \
    --record-to "$RECORD_HDF5" "${arena_args[@]}" 2>&1 | tee "$LOGS/02b_arena.log"
  e2e_stop_policy
}

stage_mimic() {
  e2e_stage 3 "mimic-expand (${MIMIC_EPISODES} variants per source, +source)"
  if [[ "$SKIP_MIMIC" == "1" ]]; then
    echo "[e2e] skipping mimic; using recording as expanded input"
    e2e_copy_hdf5 "$RECORD_HDF5" "$EXPANDED_HDF5" 2>&1 | tee "$LOGS/03_mimic.log"
    return 0
  fi

  workflows/agentic/mimic/run.sh --env "$ENV" \
    --input "$RECORD_HDF5" --output "$EXPANDED_HDF5" \
    --episodes "$MIMIC_EPISODES" --noise-std "$MIMIC_NOISE_STD" --include-source --overwrite \
    2>&1 | tee "$LOGS/03_mimic.log"
}

stage_annotate() {
  e2e_stage 4 "annotate and filter successful samples"
  local annotate_input="$EXPANDED_HDF5"
  if [[ "$SKIP_MIMIC" == "1" && ! -f "$annotate_input" ]]; then
    annotate_input="$RECORD_HDF5"
  fi

  if [[ "$SKIP_ANNOTATE" == "1" ]]; then
    echo "[e2e] skipping annotate/filter; using unfiltered demos"
    e2e_copy_hdf5 "$annotate_input" "$FILTERED_HDF5" 2>&1 | tee "$LOGS/04b_annotate.log"
    e2e_set_filter_source "$FILTERED_HDF5" "unfiltered"
    FILTER_SOURCE="unfiltered"
    return 0
  fi

  e2e_ensure_vllm "$LOGS/04a_vllm_start.log" 1

  set +e
  workflows/agentic/annotator/run.sh \
    --env "$ENV" \
    --output "$ANNOTATIONS" \
    offline \
    --hdf5-path "$annotate_input" \
    --sample-frames "$ANNOTATION_SAMPLE_FRAMES" \
    --filter "$FILTERED_HDF5" \
    2>&1 | tee "$LOGS/04b_annotate.log"
  local annotate_rc=${PIPESTATUS[0]}
  set -e

  local filtered_count=""
  if [[ -f "$FILTERED_HDF5" ]]; then
    filtered_count="$(e2e_count_demos "$FILTERED_HDF5" | tr -d '[:space:]')"
  fi

  if [[ -n "$filtered_count" && "$filtered_count" =~ ^[0-9]+$ && "$filtered_count" -gt 0 ]]; then
    e2e_set_filter_source "$FILTERED_HDF5" "vlm"
    FILTER_SOURCE="vlm"
    echo "[e2e] VLM filter kept ${filtered_count} demo(s)"
  else
    echo "[e2e] VLM filter did not produce usable demos (rc=$annotate_rc); trying simulator success labels"
    e2e_write_sim_filter "$annotate_input" "$FILTERED_HDF5"
    FILTER_SOURCE="sim_success_attr"
  fi

  echo "[e2e] filter source: $FILTER_SOURCE"
  echo "[e2e] stopping e2e-started vLLM, if any, to free GPU memory before finetune"
  e2e_stop_vllm_if_started
}

stage_replay() {
  e2e_stage 5 "replay one random filtered episode"
  if [[ "$SKIP_REPLAY" == "1" ]]; then
    echo "[e2e] skipping replay"
    return 0
  fi

  local n_demos
  n_demos="$(e2e_count_demos "$FILTERED_HDF5" | tr -d '[:space:]')"
  if [[ -z "$n_demos" || ! "$n_demos" =~ ^[0-9]+$ || "$n_demos" -lt 1 ]]; then
    echo "[e2e] ERROR: filtered HDF5 has no demos: $FILTERED_HDF5" >&2
    exit 1
  fi

  local index=$((RANDOM % n_demos))
  echo "[e2e] replaying episode $index of $n_demos"
  workflows/agentic/arena/run.sh --env "$ENV" \
    --replay "$FILTERED_HDF5" --episode-index "$index" \
    2>&1 | tee "$LOGS/05_replay.log"
}

stage_convert() {
  e2e_stage 6 "convert filtered HDF5 -> LeRobot (${VIDEO_CODEC})"
  ( unset LEROBOT_HOME; workflows/agentic/dataset/run.sh --env "$ENV" \
      --hdf5-path "$FILTERED_HDF5" --repo-id "$REPO_ID" --overwrite \
      --video-codec "$VIDEO_CODEC" ) 2>&1 | tee "$LOGS/06_dataset.log"
}

stage_viz() {
  e2e_stage 7 "start LeRobot viz"
  if [[ "$SKIP_VIZ" == "1" ]]; then
    VIZ_STATUS="skipped"
    echo "[e2e] skipping LeRobot viz"
    return 0
  fi

  set +e
  ( unset LEROBOT_HOME; workflows/agentic/dataset/viz.sh "$DATASET_DIR" \
    --state-dir "$VIZ_STATE_DIR" ) \
    2>&1 | tee "$LOGS/07_viz.log"
  local viz_rc=${PIPESTATUS[0]}
  set -e

  if compgen -G "$VIZ_STATE_DIR/lerobot-viz.*.pid" >/dev/null; then
    VIZ_STARTED_BY_E2E=1
    VIZ_STATUS="ok"
  else
    VIZ_STATUS="failed (rc=$viz_rc; see logs/07_viz.log)"
    echo "[e2e] WARN: LeRobot viz failed to start; pipeline continues" >&2
  fi
}

stage_finetune() {
  e2e_stage 8 "finetune (smoke: ${FINETUNE_MAX_STEPS} steps, bs=${FINETUNE_BATCH_SIZE} on ${POLICY_STACK})"
  if [[ "$TRAIN_SUPPORTED" == "0" ]]; then
    echo "[e2e] skipping finetune; ${ENV} is inference-only" | tee "$LOGS/08_finetune.log"
    return 0
  fi

  local train_args=(
    --env "$ENV"
    --dataset-path "$DATASET_DIR"
    --output-dir "$CKPT_DIR"
    --max-steps "$FINETUNE_MAX_STEPS"
    --save-steps "$FINETUNE_SAVE_STEPS"
    --batch-size "$FINETUNE_BATCH_SIZE"
    --num-gpus "$FINETUNE_GPUS"
    "${STACK_TRAIN_EXTRA[@]}"
    "${ENV_TRAIN_EXTRA[@]}"
  )

  uv --directory "workflows/agentic/policy/$POLICY_STACK" run "$TRAIN_BIN" \
    "${train_args[@]}" \
    2>&1 | tee "$LOGS/08_finetune.log"
}

e2e_discover_checkpoint() {
  CKPT=""
  EXTRA_POLICY_ARGS=()

  local candidates=()
  local sorted=()
  shopt -s nullglob
  candidates=("$CKPT_DIR"/checkpoint-*)
  if (( ${#candidates[@]} > 0 )); then
    mapfile -t sorted < <(printf '%s\n' "${candidates[@]}" | sort -V)
    CKPT="${sorted[$((${#sorted[@]} - 1))]}"
  else
    candidates=("$CKPT_DIR"/*/*/[0-9]*)
    if (( ${#candidates[@]} > 0 )); then
      mapfile -t sorted < <(printf '%s\n' "${candidates[@]}" | sort -V)
      CKPT="${sorted[$((${#sorted[@]} - 1))]}"
      EXTRA_POLICY_ARGS+=(--repo-id "$REPO_ID")
    fi
  fi
  shopt -u nullglob

  echo "[e2e] checkpoint=$CKPT"
  if [[ -z "${CKPT:-}" || ! -d "$CKPT" ]]; then
    echo "[e2e] ERROR: no checkpoint found under $CKPT_DIR; aborting verify" >&2
    exit 1
  fi
}

stage_validate() {
  e2e_stage 9 "validate: roll out new checkpoint for ${VERIFY_EPISODES} episode(s)"
  if [[ "$TRAIN_SUPPORTED" == "0" ]]; then
    echo "[e2e] skipping checkpoint validation; ${ENV} is inference-only" | tee "$LOGS/09_verify.log"
    return 0
  fi

  e2e_discover_checkpoint

  local arena_args=()
  mapfile -d '' -t arena_args < <(e2e_arena_rollout_args)
  e2e_start_policy "$LOGS/09a_verify_policy.log" --model-path "$CKPT" "${EXTRA_POLICY_ARGS[@]}"
  workflows/agentic/arena/run.sh --env "$ENV" --episodes "$VERIFY_EPISODES" --max-attempts "$VERIFY_MAX_ATTEMPTS" \
    --record-to "$VERIFY_HDF5" "${arena_args[@]}" 2>&1 | tee "$LOGS/09b_verify_arena.log" || true
  e2e_stop_policy

  if [[ "$SKIP_VERIFY_ANNOTATE" == "1" ]]; then
    echo "[e2e] skipping verify annotation because --skip-verify-annotate is set" | tee "$LOGS/10_verify_annot.log"
    return 0
  fi

  if e2e_ensure_vllm "$LOGS/10a_vllm_restart.log" 0; then
    workflows/agentic/annotator/run.sh \
      --env "$ENV" \
      --output "$VERIFY_ANNOT" \
      offline \
      --hdf5-path "$VERIFY_HDF5" \
      --sample-frames "$ANNOTATION_SAMPLE_FRAMES" \
      2>&1 | tee "$LOGS/10_verify_annot.log" || true
  else
    echo "[e2e] skipping verify annotation because vLLM is not ready" | tee "$LOGS/10_verify_annot.log"
  fi
}

stage_summary() {
  e2e_stage 10 "summary report"
  FILTER_SOURCE="${FILTER_SOURCE:-$(e2e_filter_source "$FILTERED_HDF5")}"
  FILTER_SOURCE="${FILTER_SOURCE:-unknown}"

  local annotated=0
  local vlm_pass=0
  if [[ -f "$VERIFY_ANNOT" ]]; then
    annotated="$(wc -l < "$VERIFY_ANNOT" 2>/dev/null || echo 0)"
    if command -v jq >/dev/null 2>&1; then
      vlm_pass="$(jq -r 'select(.annotation.success == true) | .episode_id' "$VERIFY_ANNOT" 2>/dev/null | wc -l)"
    fi
  fi

  local arena_verify_line
  arena_verify_line="$(awk '/run complete: [0-9]+\/[0-9]+ episodes succeeded/ { line=$0 } END { if (line) print line }' "$LOGS/09b_verify_arena.log" 2>/dev/null || true)"
  arena_verify_line="${arena_verify_line:-"(no arena summary line found in 09b_verify_arena.log)"}"

  {
    echo "env / stack: $ENV / $POLICY_STACK"
    echo "run dir: $RUN_DIR"
    echo "record: $RECORD_HDF5"
    echo "expanded: $EXPANDED_HDF5"
    echo "filtered: $FILTERED_HDF5  (source: $FILTER_SOURCE)"
    echo "annotations: $ANNOTATIONS"
    echo "dataset: $DATASET_DIR"
    echo "viz: $VIZ_STATUS"
    echo "checkpoint: ${CKPT:-}"
    echo "verify hdf5: $VERIFY_HDF5"
    echo "verify annotations: $VERIFY_ANNOT"
    echo "verify (arena sim labels):   $arena_verify_line"
    echo "verify (vlm annotated/pass): $annotated annotated, $vlm_pass vlm-passed"
  } | tee "$LOGS/SUMMARY.txt"

  echo "[e2e] DONE"
}

e2e_run_stages() {
  local from_stage="${1:-setup}"
  local start_index
  start_index="$(e2e_stage_index "$from_stage")"

  local stage_funcs=("${E2E_STAGE_FUNCS[@]:start_index}")
  local stage_func
  for stage_func in "${stage_funcs[@]}"; do
    "$stage_func"
  done
}
