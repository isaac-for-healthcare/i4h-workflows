---
name: i4h-workflow-create
version: "0.6.0"
description: Create a new agentic environment from an existing one (robot + task + scene + policy). Use when asked to add a new env or task. For in-place scene edits, see [[i4h-workflow-scene-edit]].
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - environment
    - scaffolding
---

# i4h Workflow — Create Env

## Purpose

Create a new agentic environment by forking the closest existing one (robot + task + scene + policy). Use when the user wants to add a new env or task; for in-place edits to an existing env see [[i4h-workflow-scene-edit]].

## Base Code

These steps drive the i4h-workflows base code (the `workflows/agentic/` tree). To reuse an existing checkout, set `I4H_WORKFLOWS` to its path (no clone happens). Otherwise this resolves the current repo, or clones to `~/i4h-workflows` — pick that default without prompting. Run every command below from the resolved root:

```bash
# Resolve the i4h-workflows base code (provides workflows/agentic/).
ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"
if [ ! -d "$ROOT/workflows/agentic" ]; then
  ROOT="${I4H_WORKFLOWS:-$HOME/i4h-workflows}"
  [ -d "$ROOT/workflows/agentic" ] || git clone https://github.com/isaac-for-healthcare/i4h-workflows "$ROOT"
fi
export I4H_WORKFLOWS="$ROOT"; cd "$ROOT"
```

## Choose Components

The env YAML at `workflows/agentic/config/environments/<env>.yaml` is the source
of truth. Ask the user to pick one option from each row below before editing —
asset/scene, robot, and policy are independent, so a choice in one row does not
imply a choice in another.

| Choice | Examples |
|---|---|
| Assets / scene | `scissor_pick_and_place`, `locomanip_tray_pick_and_place`, healthcare catalog USDs |
| Robot | SO-ARM, Unitree G1, Franka-style arm |
| Policy stack | `gr00t_n15`, `gr00t_n16`, `gr00t_n17`, `openpi_pi0` |
| Foundation / base model | `nvidia/GR00T-N1.6-3B`, `nvidia/SO_ARM_Starter_Gr00t`, `nvidia/GR00T-N1.6-Rheo-PickNPlaceTray`, custom HF/local path |

Stack rules:

- `gr00t_n15` — scissor envs, assemble-trocar inference.
- `gr00t_n16` — shared G1 locomanip code (`policy.locomanip.*`).
- `gr00t_n17` — TRT alternative for scissor envs.
- `openpi_pi0` — ultrasound.
- `assemble_trocar` is inference-only (no train module).

## Reference Recipes

If the request matches a recipe below, use it to pre-fill the Plan (component
choices already resolved — skip re-asking, go straight to forking) and still run
the full static + bridge validation. For anything not listed, fall back to
Choose Components.

### `g1_surgical_tool_sort` — "surgical tool sorting using G1 based on scissor_pick_and_place"

A recurring eval prompt. Resolved choices:

- **Env id**: `g1_surgical_tool_sort`
- **Scene source**: `scissor_pick_and_place` — keep its inline `InteractiveSceneCfg` + `ConfigAsset` + `make_*_scene_assets()` shape.
- **Robot**: Unitree G1 via `HumanoidEnvironmentBase` + registry embodiment (WBC + head cam).
- **Policy stack**: `gr00t_n16` locomanip (`policy.locomanip.infer`/`train`), base model `nvidia/GR00T-N1.6-3B`.
- **Hybrid wiring**: drop the SO-ARM `wrist`/`room` cameras (G1 head cam is the POV); keep `ground` paired with the WBC base-height (see G1 vertical setup for the values).
- **Robot stance + table offset** (horizontal): offset the forked scissor table forward (tools/trays moved with it) **or** stand the G1 back and let the policy walk up — never leave the centered table under the G1 (see Footprint clearance for the offsets).
- **Objects → destinations**: `SCISSORS_USD` scissors → `tray_a`, `SURGICAL_TWEEZERS_USD` tweezers → `tray_b`. Both trays are `SCISSOR_TRAY_USD`, kinematic, distinct colors.
- **Success**: each tool inside its own tray, per (tool, tray) pair so a swap never passes — object-position only, no robot-joint checks (works for the 43-dof G1).
- **Heights**: tabletop ~1 ft below the waist — see G1 vertical setup for the `SCISSOR_TABLE_USD` scale/pos.
- **YAML**: fork `locomanip_tray_pick_and_place.yaml`; `health_port: 8771`; sort `language_instruction`.

## Plan

Write the plan before editing:

```text
Env id:
Assets / scene source:
Robot:
Policy stack:
Foundation model / checkpoint:
Objects and destinations:
Robot stand + support-surface offset (footprints must not overlap):
Success rule:
Files to create:
YAML routing:
Validation steps:
```

For sorting tasks, require ≥2 object types, ≥2 destinations, and a success rule that fails swapped placements.

## Files to Create

All paths are relative to the **repo root** (where the agent's edit/write tools resolve); keep the `workflows/agentic/` prefix on every one — a bare `arena/...` or `config/...` creates a stray dir at the repo root.

| Path | Source pattern |
|---|---|
| `workflows/agentic/arena/arena/environments/<env>_environment.py` | Fork from the env that owns the chosen robot. |
| `workflows/agentic/arena/arena/tasks/<env>.py` | Fork from the chosen task. |
| `workflows/agentic/arena/arena/assets/<env>.py` | Fork the chosen scene source verbatim. |
| `workflows/agentic/arena/arena/runtimes/<env>.py` | Re-export the policy stack's runtime when one exists. |
| `workflows/agentic/config/environments/<env>.yaml` | Fork the YAML of the env nearest in stack/robot. |

Fork the scene source's asset-pattern shape exactly; do not switch it when the robot changes (see Hybrid Envs for the inline-vs-registry patterns).

For G1 locomanip envs:

```yaml
policy:
  stack: gr00t_n16
  infer_module: policy.locomanip.infer.infer
  train_module: policy.locomanip.train.train
```

For inference-only envs, set `train_module: null` (or omit).

## Hybrid Envs (Scene-of-A + Robot-of-B)

When the chosen scene source and robot come from different envs, this is a
**hybrid**. Do not pick the construction approach yourself — present these
options and let the user choose before editing:

- **Robot integration** (ask): (a) the robot-owner's Arena embodiment
  (e.g. `HumanoidEnvironmentBase` + the registry embodiment for G1) — brings the
  WBC action space + head camera the robot's policy stack drives, **required to
  run that stack's policy**; or (b) a raw IsaacLab `ArticulationCfg` in the scene
  cfg — no WBC / head camera, so it can't run a WBC policy.
- **Scene asset pattern** (ask only if scene source and robot-owner differ):
  keep the scene source's pattern — inline `InteractiveSceneCfg` + `ConfigAsset`
  - `make_<env>_scene_assets()` (scissor) or the `@register_asset` registry
  (locomanip). Do not mix the two.

Once the user has chosen, wire it up per the Files to Create table (env extends
the robot-owner's base — `HumanoidEnvironmentBase` for G1), plus these
hybrid-specific rules:

- For G1, the embodiment provides its own head camera via `G1EmbodimentBase.get_scene_cfg()`. Do not add cameras to the forked `InteractiveSceneCfg`.
- Keep the scene source's `ground = AssetBaseCfg(GroundPlaneCfg, ...)` field. Without a ground plane the G1 falls into the void. Pair ground z with the WBC base-height command (`apply_wbc_default_base_height`) per G1 vertical setup below.
- Static destination assets (trays, fixtures) that use `SCISSOR_TRAY_USD` must spawn `kinematic_enabled=True, disable_gravity=True`. Dynamic spawning settles the visual rim into the tabletop.

## Robot Reach

The env class's `embodiment.set_initial_pose(...)` sets where the robot stands; the assets file's `init_state.pos` sets where props start. Position the work zone within reach:

| Robot | Standing world x | Work-zone x |
|---|---|---|
| SO-ARM 101 | `(0.0, 0.0, 0.0)` (tabletop mount) | `0.0` … `0.30` |
| Unitree G1 (locomanip) | `(-0.6 … -0.3, 0.0, 0.0)` | `-0.2` … `0.2` |

When props default outside reach, move the table (with its props/destinations), not just the props.

**Footprint clearance — a free-standing robot must not stand inside the support
surface (a horizontal decision, separate from the vertical height setup).** The
G1 standing band above suits *locomanip room* scenes (open floor); a forked
**scissor table is centered at `x≈0`** (`SCISSOR_TABLE_USD` spans `x ∈ [-0.40,
+0.40]`), so that band puts the G1's torso/thighs *inside* the table and **the WBC
topples on reset — a body–table collision, not a height problem.** The G1 occupies
roughly `x ∈ [root-0.08, root+0.42]` (arms forward), so keep the table's near edge
in front of that. Two options (footprints must not overlap):

- **Offset the forked table forward (`+x`)** — e.g. center `x≈+0.45` (near edge `x≈+0.05`), G1 at `x≈-0.40`, props/destinations moved with it.
- **Stand the G1 well back (`x≈-1.0` or further)** on open floor and let the loco-manipulation policy walk up — how `locomanip_tray_pick_and_place` is laid out, and why a too-close G1 becomes stable once moved away from the table.

Verify clearance from the live poses (`GET /object?name=robot`/`table`), not the bbox — see the Phase 1 probe for the method.

### G1 vertical setup

Pair the ground z with the WBC base-height command: a ground at `z=-X` pairs with
`apply_wbc_default_base_height(embodiment, base_height_m=X)`, called in `get_env`
(WBC default 0.75 m). **`X` must be ≥ 0.792**: a standing G1 (base height `0.80`)
has its waist at `z≈0.0` and **feet at `z≈-0.792`**, so a shallower ground like
`-0.75` penetrates the feet ~4 cm and topples the WBC on reset — use `z=-0.80` /
`base_height_m=0.80`. Put the tabletop **~1 ft below the waist (`z≈-0.30`)** —
SO-ARM-derived tables default to chest height (`z≈0.238`), too high. Pick
`scale_z` + `pos.z` so the tabletop hits the target while the legs rest on the
ground: `pos.z ± half_height = tabletop_z` / `-0.80`. For `SCISSOR_TABLE_USD` at
`z≈-0.30` → `spawn.scale=(0.7,0.7,0.547)`, `init_state.pos.z=-0.55` (0.50 m tall,
top -0.30 / legs -0.80 — legs clipping the floor is cosmetic); props a few mm above.

Resizing is **source-only** (+ relaunch): you can't rescale a support surface
live — its cooked collision mesh keeps the old size and props fall through. To
preview a height in edit mode, *translate* the kinematic body down instead (see
[[i4h-workflow-scene-edit]] — "live-move embedded rigid body").

## Adding New USD Assets

When the task needs a prop the workflow doesn't already use:

- Prefer the healthcare catalog: <https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/main/catalog.md>; fall back to generic Isaac Sim / Isaac Lab assets only when no healthcare USD fits.
- Discover the exact USD path by listing the public S3 bucket:

  ```bash
  curl -s 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&prefix=Assets/Isaac/Healthcare/0.5.0/132c82d/Props/'
  ```

- Verify the USD's authored scale via the bridge `/object` bbox before picking a `scale=` tuple. Catalog USDs ship at varying unit lengths (e.g. `SCISSORS_USD` needs `(0.006, ...)` while `SURGICAL_TWEEZERS_USD` needs `(1.0, 1.0, 1.0)`).
- For static destination assets, also follow the kinematic + gravity-off rule above, pin `init_state.pos.z` so the bbox bottom is 3–5 mm above the tabletop, and reuse `_asset_world_position` for success checks (IsaacLab classifies `AssetBaseCfg` prims as XformPrim regardless of `rigid_props`; `asset.data.root_pos_w` is absent).

## YAML Checklist

Required keys:

- `robot.type`
- `zenoh.camera_names`
- `policy.stack`
- `policy.health_port`
- `policy.model_repo` / `model_revision`
- `policy.infer_module`
- `policy.train_module` (or `null` for inference-only)
- `policy.task_description` (scissor) or `policy.language_instruction` (locomanip)
- `arena.description`
- `arena.max_timesteps`
- `dataset.*` per the converter's requirements

G1 locomanip cameras use the list form only:

```yaml
policy:
  pov_cam_names_sim:
    - {obs_key: robot_head_cam_rgb, video_key: ego_view}
```

Do not use the singular `pov_cam_name_sim` key.

## Validation

Static checks (always):

```bash
workflows/agentic/policy/run.sh --list-envs
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
python -m py_compile <changed-python-files>
```

Then run the bridge scene-validation flow below — **probe → live-fix → bake → exit**; don't skip phases and don't bake before the scene passes.

**Bridge scene-validation is a required step, not a user choice.** A forked
env's geometry is only verified in the bridge, so once the files exist run it
**automatically** — don't ask or offer a "stop at code / static-only" option (the
user can interrupt). A missing `.venv` isn't a reason to ask: run
[[i4h-workflow-setup]] first, then continue. The **only** acceptable skip is a
host that can't launch Isaac Sim (no GPU / launch fails) — then still run every
static check and **report the skip explicitly** as a blocker, never as an option
the user picked.

**Minimize bridge cold starts** (each is a ~30 s Isaac Sim launch): **batch**
all source edits (use the G1 vertical-setup numbers up front so the first build
is already at the target height) and relaunch **once**, doing every live fix in
that one session and only relaunching for source edits that change
spawn/scale/collision. A confirming relaunch just to *look* after a bake is
optional — `--dry-run` + `py_compile` are the post-bake check (at most once).

### Phase 1 — Probe

```bash
REPO_ROOT="${I4H_WORKFLOWS:-$(git rev-parse --show-toplevel 2>/dev/null)}"; [ -d "$REPO_ROOT/workflows/agentic" ] || REPO_ROOT="$HOME/i4h-workflows"
ENV_ID=<env>
RUNS_ROOT="${REPO_ROOT}/workflows/agentic/runs"
RUN_DIR="${RUNS_ROOT}/scene_edit_${ENV_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${RUN_DIR}/captures"
ln -sfn "${RUN_DIR}" "${RUNS_ROOT}/.latest"

"${REPO_ROOT}/workflows/agentic/arena/run.sh" --env "${ENV_ID}" --bridge \
  2>&1 | tee "${RUN_DIR}/logs/bridge.log"
```

After `[agentic-arena] scene-edit bridge ready`:

- `GET http://127.0.0.1:8765/objects` — confirm every expected entity is `valid: true`.
- `GET /object?name=<key>` for table, robot, props, destinations, ground. Read `xform_ops` and `bbox`.
- **Robot upright + clear of the support surface (do not skip — this is how a
  toppling robot is caught).** A floating-base G1 can topple at spawn even when a
  viewport frame + stale `bbox` look fine. Confirm from the **live** pose, not the
  bbox: read `GET /object?name=robot` → `live.root_pose_w` **three times a second
  or two apart**.
  Upright = pelvis `x,y` steady and `z` constant; a topple = `x,y` drifting and/or
  `z` sinking turn after turn. Also confirm the robot's `x`-extent doesn't overlap
  the table's (see Footprint clearance). **Edit-mode caveat:** the keep-open idle
  holds `base_height≈0.65`, squatting the G1 to pelvis `z≈feet+0.65≈-0.14` and
  *holding* there — that steady value is the cosmetic idle squat, **not** a fall;
  only continued sinking / drift is a fall.
- `POST /capture` the **viewport** plus every task-relevant camera into
  `${RUN_DIR}/captures`, and read the JPEGs. Judge overall scene layout (heights,
  reach, placement) from the perspective **viewport** — the authoritative
  whole-scene view. Robot / POV cameras only check what the policy will see
  (manipulation-zone framing), not global layout.
- Score the scene against the checklist in Phase 2.

### Phase 2 — Live-fix

Apply fixes through the bridge ([[i4h-workflow-scene-edit]] for endpoint
patterns); write `/script` payloads under `${RUN_DIR}/scripts/`. Fix the scene
in dependency order — each asset rests on the one before it, so lock the
underlying asset before adjusting what sits on it:

1. **Support surface (table/shelf) first — set it in source, not live** (see G1
   vertical setup). Its height/scale determine where every other asset sits. Set
   `init_state.pos` / `spawn.scale` in `workflows/agentic/arena/arena/assets/<env>.py`,
   relaunch, and confirm via bbox that it rests on the ground (`z_min` ≈ ground z)
   with the tabletop at the robot's working height — before adjusting anything on it.
2. **Robot stance + reach** (see Robot Reach) — pin the reachable work-zone band
   next. For a free-standing robot (G1), first confirm it is **clear of the table
   footprint and stays upright** (re-read `live.root_pose_w` over a few steps per
   Phase 1); if it topples, fix the footprint overlap *before* tuning anything on
   the table.
3. **Props** rest on or just above the tabletop world z, within the reach band; nothing clips through.
4. **Static destinations** have a 3–5 mm visible gap above the tabletop.
5. **Prop USD scales** visually match real-world dimensions (use the bbox).
6. **Cameras** see the manipulation zone with the robot in frame.
7. **Per-reset randomization** keeps props on the table and away from each other.

Steps 2–7 are live bridge edits; do not edit source during them. If a live edit
returns an error, report the request payload and error to the user. Do not
restart the bridge.

### Phase 3 — Bake

```text
POST /bake names=[<adjusted entities>]
```

Apply the returned snippets:

| Bridge result | Source |
|---|---|
| Asset xform | `workflows/agentic/arena/arena/assets/<env>.py` (`init_state.pos`, `init_state.rot`, `spawn=...scale`) |
| Robot stand | `workflows/agentic/arena/arena/environments/<env>_environment.py` (`embodiment.set_initial_pose(...)`) |
| Reset randomization range | `workflows/agentic/arena/arena/tasks/<env>.py` events cfg |
| Camera / language / dataset fields | env YAML |

Re-run static validation:

```bash
python -m py_compile <changed-python-files>
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

### Phase 4 — Exit

Stop the bridge before reporting completion or proceeding to teleop/mimic/convert/finetune.

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (`.venv` and third-party checkouts present); the `run.sh` and `--bridge` flows depend on it.
- The component choices resolved (see Choose Components), plus the closest existing env to fork from.
- Bridge validation needs a GPU host able to launch Isaac Sim (each cold start is ~30 s).

## Limitations

- Fork-only: build by forking the nearest existing env's assets/task/runtime/YAML; do not assemble from scratch or mix scene-asset patterns.
- Support-surface scale/size is source-only — a live rescale won't hold; relaunch after such edits.
- `assemble_trocar` is inference-only (no train module); set `train_module: null`.
- Single env per invocation; the scene-validation flow runs one `--bridge` session at a time.

## Troubleshooting

- **Error:** `.venv` / module import fails or `run.sh` missing - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** new env not listed by `--list-envs` or `--dry-run` fails - Cause: missing/misnamed `workflows/agentic/config/environments/<env>.yaml`, an unforked file, or files written without the `workflows/agentic/` prefix (a stray `arena/` at the repo root). Fix: ensure all rows in Files to Create exist at their full paths and the YAML id matches `<env>`.
- **Error:** props fall through a support surface after resizing live - Cause: collision mesh keeps the old size. Fix: set `spawn.scale` / `init_state.pos` in source and relaunch — resizing is source-only (see G1 vertical setup).
- **Error:** G1 topples on reset - Cause (vertical): ground z too shallow so the feet penetrate it. Fix: use `z=-0.80` / `base_height_m=0.80` (see G1 vertical setup).
- **Error:** G1 topples on reset / "keeps falling" even with the correct ground z - Cause (horizontal): the robot is standing **inside the table footprint** (a placement bug, not a height bug). Fix: offset the table forward / stand the robot back so the footprints don't overlap, verifying from `live.root_pose_w` over several steps (see Footprint clearance in Robot Reach).

## Final Response

Report env id, scene/robot/policy/foundation choices, files created, static + bridge validation results, blockers.
