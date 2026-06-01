# AGENTS.md — Isaac for Healthcare Agent Entry Point

AI agent skills for the **Isaac for Healthcare (i4h)** agentic workflow — a sim-to-policy pipeline under `workflows/agentic/`. Skills live in **`skills/`** (repo root; `.claude/skills` and `.codex/skills` symlink to it). Each pipeline stage is its own skill: compose them, or use `skills/i4h-workflow-e2e/` to run the whole pipeline. Other workflows under `workflows/` (telesurgery, robotic surgery/ultrasound, …) are not covered by these skills.

## Conventions

**Base code.** Every skill drives the i4h-workflows base code (the `workflows/agentic/` tree). Inside this repo it is already present. Running a skill standalone (e.g. from a central skills repo)? Set `I4H_WORKFLOWS` to an existing checkout to reuse it; otherwise it clones to `~/i4h-workflows` without prompting. Run from the resolved root:

```bash
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

Other conventions:

- **Env YAMLs are the source of truth** — `workflows/agentic/config/environments/<env>.yaml` defines each env's robot, task, scene, policy, cameras, and randomization.
- **Set up first** — if `.venv` or third-party checkouts are missing, run `skills/i4h-workflow-setup/` before any hands-on stage.
- **Run from the repo root** so the per-skill `workflows/agentic/...` paths resolve.

## Skills directory

### Start here

- `skills/i4h-workflow/` — overview/router: supported envs, subproject layout, and which stage skill to use next.
- `skills/i4h-workflow-setup/` — install/sync dependencies and third-party checkouts; verify the toolchain.

### Author environments

- `skills/i4h-workflow-create/` — create a new env from an existing one (robot + task + scene + policy).
- `skills/i4h-workflow-scene-edit/` — edit an existing env's scene, cameras, or task in place (the `--bridge` session).

### Build datasets

- `skills/i4h-workflow-dataset-teleop/` — record human demos to HDF5 (keyboard / SO-ARM leader / VR).
- `skills/i4h-workflow-dataset-replay/` — replay an HDF5 episode in Isaac Sim to verify it.
- `skills/i4h-workflow-dataset-mimic/` — expand HDF5 demos by replicating trajectories with action/state noise.
- `skills/i4h-workflow-dataset-annotate/` — VLM success labels and episode filtering.
- `skills/i4h-workflow-dataset-convert/` — convert an HDF5 recording to a LeRobot dataset.
- `skills/i4h-workflow-dataset-transfer/` — Cosmos Transfer video augmentation (requires Docker + GPU).

### Train, evaluate, run

- `skills/i4h-workflow-finetune/` — fine-tune a GR00T or openpi PI0 policy on a LeRobot dataset.
- `skills/i4h-workflow-validate/` — roll out a policy against an env and record verification episodes.
- `skills/i4h-workflow-e2e/` — run the full pipeline end-to-end (record → mimic → annotate → replay → convert → finetune → validate).
- `skills/i4h-lerobot-viz/` — open the LeRobot HTML dataset viewer.

## Supported envs

| Env | Robot | Policy |
|---|---|---|
| `scissor_pick_and_place` | SO-ARM 101 | GR00T N1.5 (N1.7 alternative) |
| `locomanip_tray_pick_and_place` | Unitree G1 | GR00T N1.6 |
| `locomanip_push_cart` | Unitree G1 | GR00T N1.6 |
| `assemble_trocar` | Unitree G1 + Dex hands | GR00T N1.5 (inference-only) |
| `ultrasound_liver_scan` | Franka-style arm | openpi PI0 |

## Validation

Skills are validated locally with NV-BASE — see `TESTING.md` for the command, the report-inspection queries, and the current findings summary.

## Resources

- Repository: <https://github.com/isaac-for-healthcare/i4h-workflows>
- Asset catalog: <https://github.com/isaac-for-healthcare/i4h-asset-catalog>
