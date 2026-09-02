#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Policy training entry point. Supervised fine-tuning remains owned by each
# remote Task; vectorized online RL is a separate lifecycle under rl/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
usage: ./train.sh rl list
       ./train.sh rl show <workflow>
       ./train.sh rl <workflow> [--model-path PATH] [options]
       ./train.sh rl export <workflow> --checkpoint PATH --output-dir PATH

RL options:
  --dry-run             resolve and validate without starting Isaac Sim
  --eval                evaluate instead of updating the policy
  --num-envs N          vectorized environments (profile default otherwise)
  --epochs N            maximum PPO epochs (profile default otherwise)
  --episodes N          RSL-RL evaluation episodes (default: 20)
  --run-dir PATH        logs and checkpoints directory
  --resume-dir PATH     resume a supported trainer run
  --checkpoint PATH     native trainer checkpoint or run bundle for evaluation/export
  --rl-model-path PATH  backward-compatible alias for --checkpoint
  --output-dir PATH     exported policy/checkpoint destination
  --train-config PATH   resolved RLinf config.yaml for checkpoint export
  --runtime-python PATH trainer Python for RLinf, simulator/trainer Python for RSL-RL
  --sim-runtime-python PATH  Isaac Sim Python for an RLinf environment worker
  --video               record supported trainer output
  --set KEY=VALUE       additional trainer override; repeat as needed

Supervised LeRobot fine-tuning is still run through the owning Task project;
see skills/i4h-workflow-finetune.
EOF
}

[ $# -gt 0 ] || { usage >&2; exit 2; }
case "$1" in
  -h|--help) usage; exit 0 ;;
  rl) shift ;;
  *) echo "train.sh: unknown training kind '$1' (supported: rl)" >&2; usage >&2; exit 2 ;;
esac

export I4H_WORKFLOWS="$ROOT"
exec env -u VIRTUAL_ENV uv run --project rl i4h-train-rl "$@"
