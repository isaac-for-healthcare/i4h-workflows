# AGENTS.md — Isaac for Healthcare Agent Entry Point

AI agent skills for **Isaac for Healthcare (i4h)** workflows. Skills live in **`skills/`** (repo root; `.claude/skills` and `.codex/skills` symlink to it).

- **Agentic workflow** (`workflows/agentic/`) — sim-to-policy pipeline; each stage is its own `i4h-workflow-*` skill. Compose them, or use `skills/i4h-workflow-e2e/` for the full pipeline.
- **Catheter navigation** (`workflows/catheter_navigation/`) — fluorosim DRR + XPBD physics + vasculature digital twin; start at `skills/i4h-catheter-navigation/`.

Other workflows under `workflows/` (telesurgery, robotic surgery, …) do not yet have dedicated skills.

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
- **Stop/cleanup prompts are direct commands** — for prompts such as `Stop all`, run `workflows/agentic/stop.sh all` from the repo root and report the stopped components. Do not route this through a stage skill or invent `run.sh stop`.
- **Do not hard-wrap skill Markdown** — keep prose and bullets in `skills/**/*.md` as single logical lines unless a code block, table, or command needs explicit line breaks.

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

### Train, evaluate, run

- `skills/i4h-workflow-finetune/` — fine-tune a GR00T or openpi PI0 policy on a LeRobot dataset.
- `skills/i4h-workflow-validate/` — roll out a policy against an env and record verification episodes.
- `skills/i4h-workflow-e2e/` — run the full pipeline end-to-end (record → mimic → annotate → replay → convert → finetune → validate).
- `skills/i4h-lerobot-viz/` — open the LeRobot HTML dataset viewer.

### Catheter navigation (`workflows/catheter_navigation/`)

- `skills/i4h-catheter-navigation/` — overview/router: CLI modes, fluorosim layout, which stage skill to use next.
- `skills/i4h-catheter-navigation-setup/` — host/GPU preflight, PYTHONPATH, smoke-test verify.
- `skills/i4h-catheter-navigation-digital-twin/` — CT preprocess + vessel segmentation (digital twin).
- `skills/i4h-catheter-navigation-render-drr/` — single DRR frame (cache or synthetic phantom).
- `skills/i4h-catheter-navigation-viewport/` — interactive fluoroscopy + catheter physics.
- `skills/i4h-catheter-navigation-smoke/` — CPU unittest smoke (CI-friendly).
- `skills/i4h-catheter-navigation-e2e/` — chained smoke (setup → digital twin → DRR → tests).

## Supported envs

| Env | Robot | Policy |
|---|---|---|
| `scissor_pick_and_place` | SO-ARM 101 | GR00T N1.5 (N1.7 alternative) |
| `locomanip_tray_pick_and_place` | Unitree G1 | GR00T N1.6 |
| `locomanip_push_cart` | Unitree G1 | GR00T N1.6 |
| `assemble_trocar` | Unitree G1 + Dex hands | GR00T N1.5 (inference-only) |
| `ultrasound_liver_scan` | Franka-style arm | openpi PI0 |
| `surgical_reach_psm` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_reach_dual_psm` | dVRK dual PSM | GR00T N1.5 or scripted state machine |
| `surgical_reach_star` | STAR | GR00T N1.5 or scripted state machine |
| `surgical_lift_block` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_lift_needle` | dVRK PSM | GR00T N1.5 or scripted state machine |
| `surgical_lift_needle_organs` | dVRK PSM | GR00T N1.5 or scripted state machine |

## Validation

Skills are validated locally with NV-BASE. After any change under `skills/`, run the Recommended Local Validation and Eval Dataset Schema Check in `TESTING.md` before reporting the skill work complete. Treat that file as the source of truth for validation standards; do not substitute lighter checks such as `quick_validate.py` for the NV-BASE report.

## Resources

- Repository: <https://github.com/isaac-for-healthcare/i4h-workflows>
- Asset catalog: <https://github.com/isaac-for-healthcare/i4h-asset-catalog>
