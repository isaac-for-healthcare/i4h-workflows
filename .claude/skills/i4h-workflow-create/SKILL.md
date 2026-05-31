---
name: i4h-workflow-create
description: Create a new agentic environment from an existing one (robot + task + scene + policy). Use when the user wants to add a new env or task. For in-place scene edits to an existing env see [[i4h-workflow-scene-edit]].
---

# i4h Workflow — Create Env

## Basics

- Env YAML at `workflows/agentic/config/environments/<env>.yaml` is the source of truth.
- Build the new env by forking the closest existing env's assets, task, runtime, and YAML. Do not assemble from scratch.
- Treat asset/scene, robot, and policy as independent choices. Ask the user for each before editing if any is ambiguous.

## Choose Components

Ask the user to pick one option from each row before editing. A choice in one row does not imply a choice in another.

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

If the request matches a recipe below, use it to pre-fill the Plan — the
component choices are already resolved, so skip re-asking them and go straight
to forking. Still run the full static + bridge validation. For anything not
listed, fall back to Choose Components.

### `g1_surgical_tool_sort` — "surgical tool sorting using G1 based on scissor_pick_and_place"

A recurring eval prompt. Resolved choices:

- **Env id**: `g1_surgical_tool_sort`
- **Scene source**: `scissor_pick_and_place` — keep its inline `InteractiveSceneCfg` + `ConfigAsset` + `make_*_scene_assets()` shape.
- **Robot**: Unitree G1 via `HumanoidEnvironmentBase` + registry embodiment (WBC + head cam).
- **Policy stack**: `gr00t_n16` locomanip (`policy.locomanip.infer`/`train`), base model `nvidia/GR00T-N1.6-3B`.
- **Hybrid wiring**: drop the SO-ARM `wrist`/`room` cameras (G1 head cam is the POV); keep `ground`; ground `z=-0.80` paired with `apply_wbc_default_base_height(embodiment, 0.80)`. (Do **not** use `-0.75`: the standing G1's feet reach `z≈-0.792`, so a `-0.75` ground penetrates them ~4 cm at spawn and the WBC topples on reset — see G1 vertical setup.)
- **Objects → destinations**: `SCISSORS_USD` scissors → `tray_a`, `SURGICAL_TWEEZERS_USD` tweezers → `tray_b`. Both trays are `SCISSOR_TRAY_USD`, kinematic, distinct colors.
- **Success**: each tool settled inside its own tray, checked per (tool, tray) pair so a swapped placement never passes. Object-position only — no robot-joint checks (works for the 43-dof G1).
- **Heights**: tabletop ~1 ft below the waist — see G1 vertical setup for the `SCISSOR_TABLE_USD` scale/pos.
- **YAML**: fork `locomanip_tray_pick_and_place.yaml`; `health_port: 8771`; sort `language_instruction`.
- **Scene-validation pass** — the bridge "done" check; don't exit until both hold, fixing live if off (each has its own tool):
  1. **Table ~1 ft below the G1 waist.** `GET /object?name=table` → tabletop `z_max≈-0.30`, legs `z_min≈-0.80` (resting on the `-0.80` ground). If off, lower it live with `helpers.move("table", dpos=(0,0,Δz))` (legs clipping the floor is cosmetic).
  2. **Tools + trays reachable and clearly visible.** bbox: `scissors`, `tweezers`, `tray_a`, `tray_b` rest on the tabletop near the robot-side edge (within the G1 reach band) and read clearly in the **perspective viewport** (authoritative). The head camera is a secondary check — it only needs to cover the manipulation zone, not fit all four. If off, slide them in — trays via `helpers.move("tray_a", pos=…)`, tools via `POST /object/teleport`.

## Plan

Write the plan before editing:

```text
Env id:
Assets / scene source:
Robot:
Policy stack:
Foundation model / checkpoint:
Objects and destinations:
Success rule:
Files to create:
YAML routing:
Validation steps:
```

For sorting tasks, require ≥2 object types, ≥2 destinations, and a success rule that fails swapped placements.

## Files to Create

| Path | Source pattern |
|---|---|
| `arena/arena/environments/<env>_environment.py` | Fork from the env that owns the chosen robot. |
| `arena/arena/tasks/<env>.py` | Fork from the chosen task. |
| `arena/arena/assets/<env>.py` | Fork the chosen scene source verbatim. |
| `arena/arena/runtimes/<env>.py` | Re-export the policy stack's runtime when one exists. |
| `config/environments/<env>.yaml` | Fork the YAML of the env nearest in stack/robot. |

Fork the scene source's `InteractiveSceneCfg` + `ConfigAsset` + `make_<env>_scene_assets()` shape exactly. Do not switch the asset pattern when the robot changes.

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
  cfg — no WBC / head camera, so a WBC policy cannot drive it.
- **Scene asset pattern** (ask only if scene source and robot-owner differ):
  keep the scene source's pattern — inline `InteractiveSceneCfg` + `ConfigAsset`
  - `make_<env>_scene_assets()` (scissor) or the `@register_asset` registry
  (locomanip). Do not mix the two.

Once the user has chosen, wire it up:

- `arena/assets/<env>.py` forks from the scene source.
- `arena/environments/<env>_environment.py` extends the base that owns the robot (`HumanoidEnvironmentBase` for G1).
- `arena/runtimes/<env>.py` re-exports the policy stack's runtime.
- For G1, the embodiment provides its own head camera via `G1EmbodimentBase.get_scene_cfg()`. Do not add cameras to the forked `InteractiveSceneCfg`.
- Keep the scene source's `ground = AssetBaseCfg(GroundPlaneCfg, ...)` field. Without a ground plane the G1 falls into the void.
- Pair ground z with the WBC's base-height command: a ground at `z=-X` pairs with `apply_wbc_default_base_height(embodiment, base_height_m=X)`, called in `get_env`. The WBC default is 0.75 m. **`X` must be ≥ 0.792**: the standing G1's feet reach `z≈-0.792`, so a shallower ground (e.g. the old `-0.75`) penetrates the feet at spawn and the WBC topples on reset. Use `z=-0.80` / `base_height_m=0.80`.
- Static destination assets (trays, fixtures) that use `SCISSOR_TRAY_USD` must spawn `kinematic_enabled=True, disable_gravity=True`. Dynamic spawning settles the visual rim into the tabletop.

## Robot Reach

The env class's `embodiment.set_initial_pose(...)` sets where the robot stands; the assets file's `init_state.pos` sets where props start. Position the work zone within reach:

| Robot | Standing world x | Work-zone x |
|---|---|---|
| SO-ARM 101 | `(0.0, 0.0, 0.0)` (tabletop mount) | `0.0` … `0.30` |
| Unitree G1 (locomanip) | `(-0.6 … -0.3, 0.0, 0.0)` | `-0.2` … `0.2` |

When the scene's props default outside reach, move the table (and the props/destinations attached to it), not just the props.

### G1 vertical setup

A standing G1 (WBC base height `0.80`, ground `z=-0.80`) has its waist at `z≈0.0`
and its **feet at `z≈-0.792`** — so the ground must sit at `z≈-0.80` (not `-0.75`,
which the feet penetrate, toppling the WBC on reset; see the ground-pairing note
above). Put the tabletop **~1 ft below the waist (`z≈-0.30`)** — SO-ARM-derived
tables default to chest height (`z≈0.238`), too high. Pick `scale_z` + `pos.z` so
the tabletop hits the target while the legs rest on the ground: `pos.z ±
half_height = tabletop_z` / `-0.80`. For `SCISSOR_TABLE_USD` at `z≈-0.30` →
`spawn.scale=(0.7,0.7,0.547)`, `init_state.pos.z=-0.55` (0.50 m tall, top -0.30 /
legs -0.80); props a few mm above.

Resizing is **source-only** (+ relaunch): you can't rescale a support surface
live — its cooked collision mesh keeps the old size and props fall through. To
preview a height in edit mode, *translate* the kinematic body down instead (see
[[i4h-workflow-scene-edit]] — "live-move embedded rigid body").

## Adding New USD Assets

When the task needs a prop the workflow doesn't already use:

- Prefer the healthcare catalog first: <https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/main/catalog.md>. Fall back to generic Isaac Sim / Isaac Lab assets only when no healthcare USD fits.
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

Then autorun the scene-validation flow through the bridge: **probe → live-fix → bake → exit**. Do not skip phases, do not bake before the scene passes, do not leave the bridge running after.

**Minimize bridge cold starts** (each is a ~30 s Isaac Sim launch):

- **Set the support surface once.** Use the G1 vertical-setup numbers up front so the first build already has the tabletop at the target height — don't relaunch to fix "floating", then relaunch again to fix the height. **Batch** all source edits, then relaunch **once**.
- **Static dry-runs are the validation after a bake.** `--dry-run` + `py_compile` confirm the env builds; a confirming bridge relaunch just to *look* is optional — do it at most once at the very end, not after each change.
- Do every **live** fix in the one running bridge; only relaunch for source edits that change spawn / scale / collision.

### Phase 1 — Probe

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
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
- `POST /capture` the **viewport** plus every task-relevant camera into
  `${RUN_DIR}/captures`, and read the JPEGs. Judge overall scene layout (heights,
  reach, placement) from the perspective **viewport** — it is the authoritative
  whole-scene view. Robot / POV cameras only check what the policy will see
  (manipulation-zone framing), not global layout.
- Score the scene against the checklist in Phase 2.

### Phase 2 — Live-fix

Apply fixes through the bridge ([[i4h-workflow-scene-edit]] for endpoint
patterns); write `/script` payloads under `${RUN_DIR}/scripts/`. Fix the scene
in dependency order — each asset rests on the one before it, so do not adjust a
dependent asset before the thing it sits on is locked:

1. **Support surface (table/shelf) first — set it in source, not live.** Its
   height/scale determine where every other asset sits. Moving an `AssetBaseCfg`
   surface live (`xformOp:translate` / `scale`) moves only the visual, not the
   collision mesh, so props placed on it fall through. Set `init_state.pos` /
   `spawn.scale` in `arena/assets/<env>.py`, relaunch the bridge, and confirm via
   bbox that it rests on the ground (`z_min` ≈ ground z) with the tabletop at the
   robot's working height — before adjusting anything that sits on it.
2. **Robot stance + reach** (see Robot Reach) — pin the reachable work-zone band next.
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
| Asset xform | `arena/assets/<env>.py` (`init_state.pos`, `init_state.rot`, `spawn=...scale`) |
| Robot stand | `arena/environments/<env>_environment.py` (`embodiment.set_initial_pose(...)`) |
| Reset randomization range | `arena/tasks/<env>.py` events cfg |
| Camera / language / dataset fields | env YAML |

Re-run static validation:

```bash
python -m py_compile <changed-python-files>
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

### Phase 4 — Exit

Stop the bridge before reporting completion or proceeding to teleop/mimic/convert/finetune.

## Final Response

Report env id, scene/robot/policy/foundation choices, files created, static + bridge validation results, blockers.
