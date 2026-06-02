---
name: i4h-workflow-scene-edit
version: "0.6.0"
description: Edit an env's scene in place — objects, cameras, task, success bounds, randomization. Use when asked to edit a scene or launch/run/open an env in edit mode (`--bridge`), incl. a just-created env.
license: Apache-2.0
metadata:
  author: "Isaac for Healthcare Team <isaac-for-healthcare-support@nvidia.com>"
  tags:
    - isaac-for-healthcare
    - i4h
    - agentic-workflow
    - scene-edit
    - environment
---

# i4h Workflow — Scene Edit

## Purpose

Edit an existing env's scene in place via the `--bridge` scene-edit session — move/scale/swap objects, adjust cameras, or tweak task description, success bounds, or randomization. Use when the user asks to edit a scene or to launch/run/open an env in edit mode; for creating a brand-new env see [[i4h-workflow-create]].

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

## Basics

- Edits run live through the scene-edit bridge first, then persist to source on explicit user request.
- Preserve env ids and scene keys.
- **Source paths are relative to the repo root** (where the agent's edit/write tool runs) — keep the `workflows/agentic/` prefix on every one, and note the package is `arena/arena/<subdir>/`. A bare `arena/...` resolves to the wrong place.
- Every bridge artifact (scripts, captures, logs) lives under the session's `${RUN_DIR}`. Never use `/tmp`.

## Edit Lifecycle

1. **Live.** Apply each edit through the bridge HTTP API. Capture the viewport after every change.
2. **Bake.** Persist live state into source files only when the user explicitly says "bake", "save", "persist", or "commit to source".
3. **Exit.** Stop the bridge before moving to downstream steps.

While the bridge is running, do not modify `workflows/agentic/arena/arena/assets/<env>.py`, `workflows/agentic/arena/arena/tasks/<env>.py`, the env class, runtime, or env YAML.

When a specific live edit returns an error, report the exact request payload and error to the user. Do not restart the bridge as a fallback.

## Launch

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

Bridge readiness is signaled by `[agentic-arena] scene-edit bridge ready` in the log. After it appears, call `GET http://127.0.0.1:8765/objects` to enumerate scene entities.

## Bridge Endpoints (port 8765)

Base URL `http://127.0.0.1:8765`. JSON responses are either `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`.

| Method + Path | Purpose | Body / Query |
|---|---|---|
| `GET /health` | Server readiness + endpoint discovery. | — |
| `GET /context` | Exec globals, helper names, endpoint inventory. | — |
| `GET /objects` | List scene entities with `kind` (`articulation` / `rigid` / `camera` / `xform`) and prim path. | — |
| `GET /object?name=<key>` | Full state for one entity: `xform_ops`, `bbox`, `live` (authoritative PhysX pose), children. | `name=<key>` or `path=<prim_path>` |
| `GET /cameras` | List live RGB camera outputs. | — |
| `POST /capture` | Save camera frames and viewport as JPEG. | `{"output_dir": "<abs>", "viewport": true, "cameras": ["<name>", ...]}` |
| `POST /object/teleport` | Live-set pose for rigid bodies and articulations. | `{"name": "<key>", "translation": [x,y,z], "rotation_wxyz": [w,x,y,z], "zero_velocity": true, "env_index": 0}` |
| `POST /script` | Run a trusted absolute Python file on Isaac's main loop. Globals: `ctx`, `env`, `app`, `args`, `helpers`, `stage`, `get_stage`. | `{"path": "/abs/path/to/script.py"}` |
| `POST /bake` | Return Python snippets reflecting the current live xform of named entities. | `{"names": ["<key>", ...]}` |

After a teleport, read the `live` field from `GET /object?name=<key>` to verify. The `bbox` field is USD-derived and may lag a physics step.

## Edit Matrix

| Edit | Live (bridge) | Bake target |
|---|---|---|
| Move/rotate rigid object | `POST /object/teleport` | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos`/`rot` |
| Move/rotate truly-static XformPrim (no physics body anywhere in the USD — lights, decals) | `POST /script` → `xformOp:translate` / `xformOp:orient` | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos` |
| Move/rotate `AssetBaseCfg` whose USD embeds a rigid body (e.g. `SCISSOR_TRAY_USD` trays/fixtures — kinematic **child mesh**) | `POST /script` → `helpers.move("<key>", pos=/dpos=)` — drives the child PhysX body (raw USD writes snap back; see recipe) | `workflows/agentic/arena/arena/assets/<env>.py` `init_state.pos` |
| Rescale a prim | `POST /script` → `xformOp:scale` | `workflows/agentic/arena/arena/assets/<env>.py` `spawn=...scale` |
| Move robot stand | `POST /object/teleport name=robot` | `workflows/agentic/arena/arena/environments/<env>_environment.py` `embodiment.set_initial_pose(...)` |
| Add a new prim | `POST /script` → spawn USD prim (e.g. `sim_utils.CuboidCfg(...).func(...)`); a live-added body isn't GPU-simulated — don't tensor-query it (see recipe), relaunch to simulate | `workflows/agentic/arena/arena/assets/<env>.py` + `make_*_scene_assets()` |
| Toggle gravity | `POST /script` → set `physxRigidBody:disableGravity`; zero `root_lin_vel_w` / `root_ang_vel_w` | `workflows/agentic/arena/arena/assets/<env>.py` `rigid_props.disable_gravity` |
| Toggle kinematic | `POST /script` → flip `physics:kinematicEnabled` | `workflows/agentic/arena/arena/assets/<env>.py` `rigid_props.kinematic_enabled` |
| Change mass / collider props | `POST /script` → write `physxRigidBody:*` / `physxCollision:*` | `workflows/agentic/arena/arena/assets/<env>.py` `mass_props` / `collision_props` |
| Swap a USD reference | `POST /script` → `prim.GetReferences().SetReferences(...)` | `workflows/agentic/arena/arena/assets/<env>.py` `spawn.usd_path` |
| Add/remove a camera | `POST /script` → spawn `UsdGeom.Camera` + register `TiledCamera` on `env.scene.sensors` | See "Adding a Camera" |
| Change task wording | preview only | env YAML `policy.language_instruction` / `task_description` |
| Change success rule | `POST /script` → swap term on `env.unwrapped.termination_manager` | `workflows/agentic/arena/arena/tasks/<env>.py` |
| Change reset randomization range | `POST /script` → mutate `EventTerm.pose_range`; `env.reset()` | `workflows/agentic/arena/arena/tasks/<env>.py` events cfg |

## Live-Edit Recipes

Rigid / articulation teleport:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"name":"<key>","translation":[x,y,z],"rotation_wxyz":[1,0,0,0],"zero_velocity":true}' \
  http://127.0.0.1:8765/object/teleport
```

XformPrim translate (write the script under `${RUN_DIR}/scripts/`):

```bash
SCRIPT="${RUN_DIR}/scripts/move_$(date +%H%M%S)_<PrimName>.py"
cat > "${SCRIPT}" <<'PY'
from pxr import Gf
stage = get_stage()
prim = stage.GetPrimAtPath("/World/envs/env_0/<PrimName>")
prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(<x>, <y>, <z>))
PY
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"path\":\"${SCRIPT}\"}" http://127.0.0.1:8765/script
```

Live-move an embedded rigid body (kinematic `AssetBaseCfg` props / support
surfaces) — **symptom: moves for one frame then snaps back.** The prim carries a
rigid body you must drive at the PhysX layer. Catalog USDs (`SCISSOR_TRAY_USD`,
`SCISSOR_TABLE_USD`) ship the body as a **kinematic child mesh** deep in the USD,
not on the root, so `/object/teleport` refuses them (`kind` is `xform`) and
`xformOp:translate` / `set_world_poses` only touch USD (PhysX overwrites it each
step). Use the bridge's guarded **`helpers.move()`** — it finds the body (root or
child), drives it through the tensor view so the pose holds, and falls back to USD
for anything not registered at init:

```bash
SCRIPT="${RUN_DIR}/scripts/move_$(date +%H%M%S).py"
cat > "${SCRIPT}" <<'PY'
print(helpers.move("tray_a", pos=(-0.196, None, None)))             # scene key; None = keep that axis
print(helpers.move("/World/envs/env_0/Table", dpos=(0, 0, -0.30)))  # prim path; delta (lower a surface)
PY
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"path\":\"${SCRIPT}\"}" http://127.0.0.1:8765/script
```

`helpers.move` returns `{"mode": "physx", ...}` when it drove the body (pose
holds) or `{"mode": "usd", "warning": ...}` for a live-added / non-physics prim
(repositioned but not simulated). It also takes `rot_wxyz=(w,x,y,z)`.

Notes: **never** call `create_rigid_body_view(path).get_transforms()` yourself on
a prim that wasn't in the scene at bridge launch — a live-added body isn't in the
GPU pipeline and querying it is a **fatal, unrecoverable CUDA fault that kills the
bridge**. `helpers.move` / `helpers.rigid_body_view` guard against exactly this
(they refuse bodies not registered at init). Verify a move held by re-reading the
pose after a few steps (trust the live pose, not `bbox` — USD-derived, stale). A
translate carries collision too (props still rest) but only repositions — you
can't **rescale** this way, so a support-surface resize stays a source
`spawn.scale` change. The durable equivalent of any move is `init_state.pos` in
source (the kinematic target seeds from the spawn pose).

Capture viewport:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"${RUN_DIR}/captures\",\"viewport\":true}" \
  http://127.0.0.1:8765/capture
```

## Adding a Camera

### Live

Spawn a `UsdGeom.Camera` prim and register it as a `TiledCamera` sensor through a single `/script` payload. Pose source is the user-specified world transform or the current `/OmniverseKit_Persp` viewport.

**Default the new camera's `width`/`height` to match the scene's existing cameras** (the head cam / env YAML `policy.image_size`) rather than the viewport resolution — when posing from `/OmniverseKit_Persp`, take its pos/rot/focal but keep the existing pixel size (the viewport gives the *framing*, not the resolution).

```python
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
img_h, img_w = 480, 640  # match existing cameras = env YAML policy.image_size [H, W]
cfg = TiledCameraCfg(
    prim_path="/World/envs/env_0/<CameraPrim>",
    offset=TiledCameraCfg.OffsetCfg(pos=(x, y, z), rot=(w, qx, qy, qz), convention="opengl"),
    data_types=["rgb"],
    spawn=PinholeCameraCfg(focal_length=focal, horizontal_aperture=h_aperture, clipping_range=(0.1, 1.0e5)),
    width=img_w, height=img_h, update_period=1/30.0,
)
env.scene.sensors["<scene_key>"] = TiledCamera(cfg)
env.scene.sensors["<scene_key>"]._initialize_impl()
```

Verify with `GET /cameras` and a `POST /capture` against the new label.

### Bake

Apply all touchpoints in one pass:

1. **Scene** — `workflows/agentic/arena/arena/assets/<env>.py`: add a `TiledCameraCfg` field with `prim_path="{ENV_REGEX_NS}/<CameraPrim>"`, include the field name in `make_*_scene_assets()`'s `asset_names` tuple.
2. **Observation (this is what gets RECORDED)** — `workflows/agentic/arena/arena/tasks/<env>.py.modify_env_cfg`: append `env_cfg.observations.policy.<obs_key> = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("<scene_key>"), "data_type": "rgb", "normalize": False})`. The recorder serializes the **`policy`** group only — a camera in `camera_config`/`camera_obs` renders but is NEVER recorded. The `<obs_key>` here becomes the HDF5 obs key, so it MUST equal the `dataset.camera_mappings` key in step 5 (e.g. `robot_room_cam`) — do NOT add a `_rgb` suffix (that's the `camera_obs` term name, not the recorded key). Verify with a zero-action smoke: the key must appear in the **policy** obs-group table.
3. **Zenoh** — env YAML `zenoh.camera_names`: append the camera label.
4. **Policy input** — env YAML `policy.pov_cam_names_sim`: append `{obs_key: robot_<scene_key>_cam_rgb, video_key: <video_key>}`. Arena publishes camera observations under `robot_<scene_key>_cam_rgb`; the `robot_` prefix is required.
5. **Dataset columns** — env YAML `dataset.camera_mappings`: add `<sim_cam_name>: observation.images.<video_key>`.
6. **Dataset modality** — env YAML `dataset.modality_template_path`: point at a modality JSON whose `video.<video_key>` entry maps to `observation.images.<video_key>`.
7. **Train modality** — env YAML `policy.train.modality_config_path`: point at a GR00T config module whose `ModalityConfig(modality_keys=[...])` matches the dataset template's video keys. The path is resolved relative to `workflows/agentic`; use `policy/<stack>/policy/<task>/config*.py`.
8. **Re-record** — any prior single-camera HDF5 is unusable. Re-teleop before convert/mimic/finetune.

### Camera YAML Rules

- Use the list form of `pov_cam_names_sim` only:

  ```yaml
  policy:
    pov_cam_names_sim:
      - {obs_key: robot_head_cam_rgb, video_key: ego_view}
      - {obs_key: robot_room_cam_rgb, video_key: room_view}
  ```

- Do not use the singular `pov_cam_name_sim` key.
- Single-camera and dual-camera checkpoints are not interchangeable.

## Durable Touchpoints (bake targets)

- `workflows/agentic/arena/arena/environments/<env>_environment.py`: env wiring, robot stand pose.
- `workflows/agentic/arena/arena/assets/<env>.py`: static scene assets.
- `workflows/agentic/arena/arena/tasks/<env>.py`: reset randomization, success, task text.
- `workflows/agentic/arena/arena/runtimes/<env>.py`: runtime-specific camera/state/action logic.
- `workflows/agentic/config/environments/<env>.yaml`: cameras, policy language, dataset mappings.

## Notes

- `assemble_trocar` is inference-only. Do not add train hooks during a scene edit.
- If adding/removing cameras, update `policy.data_config`, `dataset.camera_mappings`, and the train modality config together.
- Scissor SO-ARM generates `meta/modality.json` from YAML splits and does not need `dataset.modality_template_path`. G1 locomanip and assemble-trocar do.

## Verify (after bake)

```bash
python -m py_compile <changed-python-files>
python - <<'PY'
import yaml, pathlib
for p in pathlib.Path('workflows/agentic/config/environments').glob('*.yaml'):
    yaml.safe_load(p.read_text())
PY
workflows/agentic/arena/run.sh --env <env> --dry-run
workflows/agentic/policy/run.sh --env <env> --dry-run
```

## Prerequisites

- Workflow set up via [[i4h-workflow-setup]] (`.venv` present); the `arena/run.sh --bridge` launch depends on it.
- An existing env id with its scene keys (the bridge edits an env in place; preserve its ids).
- A GPU host able to launch Isaac Sim for the bridge session.

## Limitations

- Live edits are not persisted until an explicit bake; only bake on user request ("bake"/"save"/"persist"/"commit to source").
- Support-surface rescale is source-only (`spawn.scale`) — moving an `AssetBaseCfg` surface live moves only the visual, not the collision mesh, so props fall through; relaunch to apply.
- A live-added body is not GPU-simulated and must not be tensor-queried (a manual `create_rigid_body_view(...).get_transforms()` is a fatal CUDA fault); relaunch to simulate it.
- While the bridge runs, do not edit `workflows/agentic/arena/arena/assets/<env>.py`, `workflows/agentic/arena/arena/tasks/<env>.py`, the env class, runtime, or env YAML.

## Troubleshooting

- **Error:** `.venv` / import fails or bridge won't launch - Cause: workflow not set up. Fix: run [[i4h-workflow-setup]] first.
- **Error:** `GET /objects` / `127.0.0.1:8765` unreachable - Cause: bridge not ready yet. Fix: wait for `[agentic-arena] scene-edit bridge ready` in `bridge.log` before calling endpoints.
- **Error:** object moves for one frame then snaps back - Cause: it's a kinematic embedded rigid body (`SCISSOR_TRAY_USD`/`SCISSOR_TABLE_USD`), so `/object/teleport` and raw `xformOp:translate` don't hold. Fix: use `helpers.move("<key>", ...)` to drive the PhysX body.
- **Error:** a live edit returns `{"ok": false, "error": ...}` - Cause: invalid request for that entity. Fix: report the exact payload and error to the user; do not restart the bridge as a fallback.

## Final Response

Live session: report each bridge action, verified live pose, capture path.

After bake: report files touched, validation results, and confirm bridge state matches source.
