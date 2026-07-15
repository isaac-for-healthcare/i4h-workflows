# Agentic Arena

IsaacLab-Arena uv project. Environment classes are discovered from
`arena/environments/*_environment.py` and matched to `config/environments/<env>.yaml`.

```bash
workflows/agentic/arena/setup.sh                                               # clone third-party + uv sync
workflows/agentic/arena/run.sh --list-envs                                     # show registered environments
workflows/agentic/arena/run.sh --env <env_id>                                  # zero-action rollout
workflows/agentic/arena/run.sh --env <env_id> --bridge                         # open edit mode + scene-edit bridge
workflows/agentic/arena/run.sh --env <env_id> --num-steps 40 --dump-scene      # final zero-action scene smoke
workflows/agentic/arena/run.sh --env <env_id> --episodes 1                     # one policy-driven episode
workflows/agentic/arena/run.sh --env <env_id> --state-machine --episodes 1     # one scripted state-machine episode
workflows/agentic/arena/run.sh --env <env_id> --episodes N --max-timesteps T   # cap each episode at T steps
```

`--max-timesteps` defaults to the env's `arena.max_timesteps` in
`config/environments/<env>.yaml`; pass it on the CLI to override.
The same YAML block must also provide `arena.description` for `--list-envs`.

## Scene-Edit Bridge

For a newly created or changed scene, launch edit mode first. Use the bridge to
inspect live prims, capture task-camera frames, capture the current viewport
perspective, make live scripted fixes, then bake accepted changes back into the
env assets before teleop.

```bash
workflows/agentic/arena/run.sh --env scissor_pick_and_place --bridge
```

The local scene-edit bridge uses the env's `arena.bridge_port` from `config/environments/<env>.yaml`. Resolve the URL before calling endpoints:

```bash
BRIDGE_URL="$(workflows/agentic/arena/run.sh bridge-url --env scissor_pick_and_place)"
curl -sS "${BRIDGE_URL}/objects"
curl -sS "${BRIDGE_URL}/cameras"
curl -sS "${BRIDGE_URL}/capture" \
  -H 'Content-Type: application/json' \
  -d '{"viewport": true}'
```

`POST /capture` writes JPEGs under
`workflows/agentic/runs/arena_bridge/captures/` by default. Pass an absolute
`output_dir` in the JSON body to keep captures with a specific create or edit
run.

## Zero-Action Examples

Zero-action runs step the environment with no policy or teleop input. For new or
modified scenes, use them after edit-mode fixes have been baked to double-check
scene layout, camera visibility, object placement, and reset state before
recording or policy rollout.

```bash
# Basic zero-action rollout.
workflows/agentic/arena/run.sh --env scissor_pick_and_place --num-steps 40

# Final smoke: dump camera frames plus scene poses after edit-mode fixes.
workflows/agentic/arena/run.sh --env scissor_pick_and_place --num-steps 40 --dump-scene

# Dump frames and poses at selected zero-action steps.
workflows/agentic/arena/run.sh --env scissor_pick_and_place --num-steps 40 --dump-scene --dump-scene-steps 0,1,10,20,30
```

When `--dump-scene` is passed without a directory, outputs are written to
`workflows/agentic/runs/<env>/scene_dumps/`. The directory
contains `manifest.json`, camera JPEGs, and `scene_poses.json`; each pose record
references the images dumped for the same step.

## State-Machine Examples

Scripted state-machine runs drive an env to task success without a policy daemon, and record the successful episodes to HDF5 when `--record-to` is given. Available for the scissor and ultrasound envs and the surgical reach/lift envs; each uses `arena.max_timesteps` from its YAML unless `--max-timesteps` is provided. See `arena/statemachine/README.md`.

```bash
workflows/agentic/arena/run.sh --env scissor_pick_and_place --state-machine --episodes 1
workflows/agentic/arena/run.sh --env ultrasound_liver_scan --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_reach_psm --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_reach_dual_psm --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_reach_star --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_lift_block --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_lift_needle --state-machine --episodes 1
workflows/agentic/arena/run.sh --env surgical_lift_needle_organs --state-machine --episodes 1
```

## Policy & Teleop Examples

Start the matching policy daemon first, then run one Arena episode:

```bash
workflows/agentic/arena/run.sh --env scissor_pick_and_place --episodes 1
workflows/agentic/arena/run.sh --env scissor_pick_and_place --episodes 1 --teleop --teleop-device so101_leader  # defaults to /dev/ttyACM1

workflows/agentic/arena/run.sh --env locomanip_tray_pick_and_place --episodes 1
workflows/agentic/arena/run.sh --env locomanip_push_cart --episodes 1
workflows/agentic/arena/run.sh --env assemble_trocar --episodes 1
workflows/agentic/arena/run.sh --env ultrasound_liver_scan --episodes 1
```

Record successful episodes to HDF5, then replay one:

```bash
workflows/agentic/arena/run.sh --env scissor_pick_and_place --episodes 3 --record-to recording.hdf5
workflows/agentic/arena/run.sh --env scissor_pick_and_place --replay recording.hdf5 --episode-index 0
```
