# Workflow Ownership Map

Use this map after reading `./AGENTS.md` and `./DESIGN.md`. Inspect the concrete implementation before editing; filenames are not API contracts.

## Authoring and runtime ownership

| Concern | Owner | Source |
|---|---|---|
| Workflow id → environment, policies, devices, commands | Specialty page | `./docs/workflows/<specialty>.md` |
| Every scene as an IsaacLab-Arena external environment, shared environment flags | Environments | `./arena/i4h_arena/environments.py` |
| World assets, cameras, randomization, embodiment choice, scene-specific flags, teleop device declaration | Scene | `./arena/i4h_arena/scenes/<scene>.py`, `./arena/i4h_arena/assets/` |
| Observations, events, rewards, success and failure terminations | Scene task | `./arena/i4h_arena/envcfg/` |
| Scene capabilities, step budget, control rate | Scene manifest | `./arena/i4h_arena/scenes/manifest/<scene>.yaml` |
| Robot articulation, actions, observations, cameras | Embodiment | `./arena/i4h_arena/embodiments/` |
| Cross-process robot labels, calibration, home pose, teleop device names | Embodiment manifest | `./arena/i4h_arena/embodiments/manifest/<robot>.yaml` |
| Rule-based controllers and remote-policy clients | Arena policies (`PolicyBase`) | `./arena/i4h_arena/policies/` |
| Keyboard, leader-arm, and catheter teleoperation | Isaac Lab devices (`DeviceBase`) | `./arena/i4h_arena/devices/` |
| Live sensor panels for visible runs | Operator UI | `./arena/i4h_arena/ui/` |
| Policy server, backend/model/observation/training contract | Policy stack and task manifest | `./tasks/<stack>/i4h_tasks/<stack>/`, `./tasks/<stack>/i4h_tasks/<stack>/manifest/<task>.yaml` |
| HDF5 contract and manifest loaders | Common | `./common/i4h_common/` |
| Mimic, annotation, conversion, Cosmos, patient twin | Offline tools | `./tools/` |
| Vectorized online RL stepping, trainer mapping, and hyperparameters | RL training lifecycle | `./rl/`, `./train.sh` |
| Live scene editing | Bridge and authoring utilities | `./local-agent/bridge.sh`, `./scripts/live_scene_edit.py`, `./scripts/authoring_info.py` |
| Robot drive (PD) tuning | Arena scripts | `./arena/scripts/robot_pd_tune.py`, `./arena/scripts/robot_pd_tuning.md` |
| Blank scene generation | Generator | `./scripts/create_blank_environment.py` |
| Maintained data-to-policy loop | End-to-end driver | `./scripts/e2e/run.sh` |

`i4h_arena.environments` creates one `<CamelCase>` environment class per scene manifest, so a new manifest is runnable as `--external_environment_class_path i4h_arena.environments:<CamelCase> <scene>` without registration. Environment flags follow the scene name: the shared `--episode_steps`, `--recorder_dataset_export_dir_path`, and `--recorder_dataset_filename`, plus the scene's own `add_cli_args`, such as `--patient_twin`, `--leader_port`, or `--rl_observations`. Launcher and policy flags precede it: Arena turns each policy `Cfg` field into a flag, such as `--embodiment` or `--remote_port`.

## Dependency rules

`tests/test_layering.py` enforces these:

- Keep `common` independent of every other i4h layer, Isaac Sim, and Isaac Lab.
- Keep `arena` free of policy stacks: it never imports `i4h_tasks.*`, GR00T models or configs, or openpi. Only the policy clients Arena itself uses cross over.
- Keep each policy stack a leaf that depends only on `common` among i4h layers and never imports Arena, Isaac Sim, or Isaac Lab (`gr00t_n16` may reuse Arena's G1 joint-remapping helpers).
- Keep offline tools independent of Arena, RL, Isaac Sim, and Isaac Lab.
- Let only an isolated RL trainer own vectorized stepping; `rl` imports the simulator and model stacks lazily, after its preflight.

## Existing pattern families

- SO-ARM scissors: `scissor_pick_and_place` as `SoarmScissors` (`soarm_scissors`); the rule-based `i4h_arena.policies.scissors.ScissorPickPlacePolicy`; GR00T N1.5 served by `i4h_tasks.gr00t_n15.serve` with `Gr00tN15RemotePolicy --embodiment so101_scissors`; GR00T N1.7 served by `i4h_tasks.gr00t_n17.serve` with `Gr00tN17RemotePolicy`; `record_demos.py` with the `keyboard` or `so101_leader` device; `replay_demos.py`; and the e2e loop.
- G1 loco-manipulation: `locomanip_push_cart` and `locomanip_tray_pick_and_place` as `G1Cart` and `G1Tray`; GR00T N1.6 on GR00T's own `run_gr00t_server.py` with Arena's `Gr00tRemoteClosedloopPolicy` and `tasks/gr00t_n16/config/g1_locomanipulation.yaml`; `record_demos.py --embodiment g1_locomanip_wbc_pink --teleop_device keyboard_23d`; and `replay_demos.py`.
- G1 trocar: `assemble_trocar` as `G1Trocar` (`g1_trocar`); GR00T N1.5 with `Gr00tN15RemotePolicy`; separate RLinf PPO post-training through `./train.sh rl assemble_trocar`.
- Ultrasound reach: `ultrasound_probe_reach` as `UltrasoundProbeReach`; RSL-RL PPO training through `./train.sh rl`, evaluated with Arena's `rsl_rl` policy and the scene's `--rl_observations` flag.
- Ultrasound liver scan: `ultrasound_liver_scan` as `PandaPhantom` (`panda_phantom`); PI0 served by `i4h_tasks.openpi_pi0.serve` with `i4h_arena.policies.pi0.LiverScanPi0Policy`; the rule-based `i4h_arena.policies.scan.ProbeSweepPolicy`; `record_demos.py` keyboard teleop; `replay_demos.py`; and the e2e loop in `scripts/e2e/run.sh`.
- Surgical: the PSM, dual-PSM, and STAR reach and lift workflows as `PsmReach`, `DualPsmReach`, `StarReach`, `PsmBlock`, `PsmNeedle`, and `PsmNeedleOrgans` with `i4h_arena.policies.reach.CartesianReachPolicy` or `i4h_arena.policies.lift.CartesianLiftPolicy`.
- Endoluminal: `endoluminal_navigation` and `endoluminal_navigation_arm` as `EndoluminalNavigation` and `EndoluminalNavigationArm` on `--presets newton`; `i4h_arena.policies.catheter.CatheterSweepPolicy`, `FluoroscopyCArmSweepPolicy`, the `catheter_keyboard` device, the scene's `--patient_twin` flag, and GR00T N1.7 `catheter_navigation`.
- Tablecloth: `spread_tablecloth_g1` and `spread_tablecloth_h2` as `G1Tablecloth` and `H2Tablecloth` on `--presets newton`; XR teleop through Isaac Teleop, replay, and `HoldPolicy` inspection.

Confirm these families in `docs/workflows/` and `arena/i4h_arena/scenes/manifest/`; manifests and pages may evolve.

## Validation order

```bash
cd "$(git rev-parse --show-toplevel)"
ls arena/i4h_arena/scenes/manifest
python3 scripts/run_tests.py --suite light
arena/.venv/bin/python -m pytest arena/tests/test_scene.py -q
```

The light suite covers contracts, manifests, and layering without Isaac Sim; `arena/tests/test_scene.py` checks that every scene is an Arena environment named after it. Run focused tests in each owning component environment. For scene, asset, or task changes, finish with a visible policy-runner run (`--viz kit`) of every affected policy or device; `HoldPolicy` viewing proves only a stationary layout.
