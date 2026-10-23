---
name: i4h-workflow-setup
description: Preflight and set up the i4h component environments and third-party checkouts. Use for installation, missing component environments, or third-party failures; do not use for rollout validation.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  version: "0.8.0"
  verification-request: "2026-09-15"
  tags:
    - isaac-for-healthcare
    - i4h
    - setup
    - dependencies
---

# i4h Workflow Setup

## Purpose

Prepare every component environment, then prove that the scenes and their IsaacLab-Arena environments are discoverable without launching Isaac Sim.

## Instructions

1. Resolve the base checkout.
2. Run every host preflight check.
3. Run default full setup.
4. Verify the component environments, scene discovery, and the CPU contract checks.

## Resolve the checkout

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
```

Treat this resolver as part of the skill contract: a hosted copy may run outside the base repository, so never assume the current checkout contains the scene manifests in `arena/i4h_arena/scenes/manifest`. `I4H_WORKFLOWS_REPO_URL` selects the clone source. When `I4H_WORKFLOWS` is unset, derive the fallback directory from that URL; set `I4H_WORKFLOWS` only to reuse or choose a specific destination. Never replace an existing checkout.

## Preflight

```bash
uname -srm
command -v uv
command -v git
nvidia-smi
df -h .
./setup.sh --help
```

Require supported Linux on `x86_64` or `aarch64`, `uv`, Git, network access for missing packages/assets, sufficient disk and model-cache space, and an NVIDIA driver compatible with the repository's Isaac Sim version. Read the exact current requirements from `./README.md`; do not rely on remembered version numbers. Require Docker only when the requested local VLM or Cosmos service needs it.

## Sync

Run the default full setup:

```bash
./setup.sh
```

The explicit equivalent is:

```bash
./setup.sh all
```

Setup syncs every uv project, each with its own lock and `.venv`: the light projects `common` and `rl`; the policy stacks `tasks/gr00t_n15`, `tasks/gr00t_n16`, `tasks/gr00t_n17`, and `tasks/openpi_pi0`; the tools `tools/mimic`, `tools/dataset`, `tools/cosmos`, `tools/annotator`, and `tools/patient_twin`; and `arena`, the IsaacLab-Arena simulator environment. Set `I4H_SETUP_PROJECTS` to a space-separated subset only when the user asks for specific components or a specialized container needs them; ordinary host setup leaves it unset.

Use `./setup.sh links` only to restore project `.venv` links when `I4H_VENV_ROOT` keeps the environments outside the checkout. Use `./setup.sh clean` only when the user explicitly requests a full rebuild or cleanup because it deletes generated caches, component environments, and third-party checkouts. Setup is idempotent; do not clean as a generic repair. The setup script restores the pinned third-party sources (Isaac Lab, IsaacLab-Arena, GR00T, openpi, and the others) and the upstream Isaac Sim skill catalog; do not copy that catalog into `skills/`.

## Verify

```bash
ls arena/i4h_arena/scenes/manifest
for project in common rl tasks/gr00t_n15 tasks/gr00t_n16 tasks/gr00t_n17 tasks/openpi_pi0 \
  tools/mimic tools/dataset tools/cosmos tools/annotator tools/patient_twin arena; do
  [ -x "$project/.venv/bin/python" ] && echo "ok $project" || echo "missing $project"
done
test -f ./third_party/IsaacSim-74a9356/skills/SKILLS.md
python3 scripts/run_tests.py --suite light
arena/.venv/bin/python -m pytest arena/tests/test_scene.py -q
```

The light suite checks contracts, manifests, and layering; `arena/tests/test_scene.py` checks that every scene is an IsaacLab-Arena environment named after it. Neither launches Isaac Sim. When the user named a workflow, find its environment and commands in its `docs/workflows/<specialty>.md` page. Use `i4h-workflow-validate` for an actual rollout.

## Troubleshooting

On failure, report the first host, network, checkout, or uv-sync error and rerun full setup only after correcting that cause. Setup prints `failed: <projects>` for the projects whose sync failed; report each one.

## Prerequisites

Require supported Linux, network access, `uv`, Git, disk space, and a compatible NVIDIA GPU/driver for simulator use.

## Limitations

Setup does not download every optional checkpoint or prove any rollout succeeds. `clean` is destructive to generated environments and third-party checkouts.

## Examples

- `Set up the i4h workflow on this machine and tell me if any host requirements are missing.` → preflight, run default full setup, then verify the component environments, scene discovery, the light suite, and the Arena scene test.

## Completion gate

Report every host check, synced project, first failing component if any, the scene list and CPU check results, and the smallest next smoke command, such as the `HoldPolicy` `psm_reach` run in `AGENTS.md`. Do not claim simulator or policy success from setup alone.
