# G1 Reach and Contact Authoring

Use this reference when a workflow asks Unitree G1 to face, walk toward, or stop safely near a named object.

## Write the behavior as one Arena policy

Implement the behavior as one rule-based IsaacLab-Arena `PolicyBase` under `arena/i4h_arena/policies/`, registered with `@register_policy` and loaded by class path. Follow the per-env phase state machine that `arena/i4h_arena/policies/__init__.py` describes and `lift.py` implements: advance a phase once per step from privileged scene state and reset it per env in `reset(env_ids)`. Facing and walking are phases of that policy; do not add a workflow module, a task graph, or a second copy of the heading math, WBC encoding, upright check, or collision history elsewhere.

Use `i4h_arena.assets.authoring_catalog.authoring_asset("g1").embodiment` for the registry name, action/joint widths, camera aliases, and verified collision-bearing body names. Use `spawn_pose_facing_rectangle()` from the same module for an initial edge-relative position and heading. Do not paste the G1 body list or hand-compute a one-off spawn quaternion into a scene.

The G1 WBC action is 50-D:

- `[0:43]`: measured 43-joint posture, held while locomoting
- `[43:46]`: forward velocity, lateral velocity, yaw rate
- `[46]`: base-height command
- `[47:50]`: torso roll, pitch, yaw

`arena/i4h_arena/policies/hold.py` shows how to read the standing posture and default base height from the WBC action term. Run at the scene manifest's `control_hz`. The facing phase commands yaw only. The walking phase commands forward velocity only after alignment, tapers its approach, and sends a zero navigation command inside the success band. Expose tunable thresholds as `Cfg` fields, which Arena turns into CLI flags.

## Keep geometry in the scene

The nearest-edge calculation consumes the object's footprint half extents. Measure the object's task-relevant collision footprint from confirmed metric world bounds and keep it in one scene-owned helper beside the scene's assets or task. Both the policy and the success termination read it from there; do not duplicate those extents in policy parameters or termination code.

Use an oriented-footprint implementation before applying an axis-aligned footprint to a rotated obstacle whose world-aligned bounds would materially change the stopping distance.

## Configure robot-object collision sensing

Read the upstream `physics-simulation` and `isaac-sim-sensor` skills. Then inspect the spawned live stage, because the asset wrapper path is not necessarily its rigid-body path. The sensing and filtered paths must each resolve to exactly one supported rigid-contact entry. A path prefix that also matches nested collision or visual prims is not exact enough even when the wrapper itself carries `RigidBodyAPI`.

Isaac Lab's PhysX filtered contact sensor supports one sensing body to many filter bodies, not many-to-many filtering. To detect contact between an articulated G1 and one rigid table body:

1. Enable contact reporting on both asset spawners with `activate_contact_sensors=True`.
2. Identify the table's actual rigid body in the live stage and confirm its filter expression resolves exactly once in the PhysX tensor view.
3. If the imported table hierarchy makes an exact GPU filter impossible, author one invisible kinematic collision proxy with the confirmed table collision footprint. Use its unique rigid-body prim as the filter and keep its dimensions aligned with the scene-owned footprint; avoid leaving a second overlapping collision representation active when that would change task physics.
4. Build the sensors with `i4h_arena.assets.contact.filtered_contact_sensor_family()`, passing `family_name="contact_robot_table"`, the catalog's G1 `contact_body_names`, the robot prim root, and the unique table body or proxy. Do not include IMU, camera, logo, visual-only, or other fixed decoration prims merely because they appear below the robot wrapper.
5. Add the returned sensors, named `contact_robot_table__<body>`, to the scene configuration with its assets. The success termination aggregates the family by reading every sensor with that prefix.

Do not attach one filtered sensor to `Robot/.*`; that is unsupported many-to-many filtering. Do not use a collision-mesh child as a filtered rigid body merely because it makes a prefix match narrower: PhysX can reject that collider for GPU contact filtering. Treat `filter_count` errors, “provided patterns ... did not match,” and errors for the configured sensor/filter prims as launch failures. Warnings for descendant visual prims are noisy but are not equivalent to the configured rigid-body prim failing; verify the sensor itself initializes and produces a filtered force.

## Define safe reach success

Define success as the scene task's success termination in `arena/i4h_arena/envcfg/`, never inside the policy. Require all of the following for a stable interval:

- outside the object's footprint and no farther than the requested edge distance
- upright within the requested tilt
- root linear speed below its threshold
- root angular speed below its threshold
- no robot-object contact at any earlier tick in the episode

Reset collision history on episode reset with a reset event. The success termination and the rule-based policy must use the same reusable helper for the edge distance and upright checks.

## Dynamic validation

Static tests prove only contract structure. A safety-qualified reach requires these policy-runner rollouts, each read from its `episode_results_rank0.jsonl`:

1. A normal visible rollout (`--viz kit`) that reaches the band, stops, remains upright, and records `success: true`.
2. A forced-contact negative rollout, for example with a policy `Cfg` flag that skips the stop, that produces a non-zero filtered force, latches collision history, commands stop, and cannot succeed afterward.
3. A fresh reset proving collision history clears and the next episode can succeed again.

Record sensor names, force-matrix shape, maximum filtered force, final linear and angular speeds, tilt, edge distance, and success/failure. A `HoldPolicy` view alone is insufficient.
