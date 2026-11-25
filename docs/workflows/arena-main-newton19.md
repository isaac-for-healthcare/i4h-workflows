# Arena main / Newton 1.9 qualification

This report records the pre-migration campaign at `1208154b`. Its launcher commands, legacy HDF5 replay checks and test counts are historical. Current workflow commands use native Arena tools; see the [liver guide](liver_reach_retract.md) and [endoluminal guide](endoluminal-robotics.md). The standalone aorta capture/calibration utility now uses native construction, but its historical replay-to-recording comparison has not been repeated.

Worktree: `i4h/arena-main-newton19`, based on Workflows
`f0dee7b33a9323cc0cbbc4c64fe3df6940501620`, with the RLinf channel failure
fix from `2ab93f52` and the single-mesh cloth spawn fix ported from `fbd95505`.
The bounded qualification passed on 2026-11-19 on Linux x86-64 with an RTX PRO
6000 Blackwell GPU and driver 615.71.09. The shared-stack and legacy gates passed
before the new aorta Scene was added. Full catalog, headset XR, policy checkpoints
and physical force calibration remain follow-up work.

The pins below are the current stack after the
[latest-upstream bump](#latest-upstream-bump-november-19). The original campaign
ran on Arena `813be56`, Isaac Lab GA `e3f7d9f` and Newton `7f0c13a`. Setup,
Patches and Reproduce describe the current stack. The static-aorta results, test
counts and evidence, retained impulse deviations and follow-up catalog sections
are that original campaign's snapshot.

The separate `endoluminal_aorta_fluoroscopy` Workflow exposes `idle`, `demo` and
`replay`. It uses `aorta_static`, one environment, 60 Hz physics, decimation two,
CUDA graph capture, and the existing 600-step cap. Its three normalized channels
are insertion, axial rotation and tip bend. The legacy navigation Workflows keep
their four physical action channels and original controller/recording contracts.

| Component | Exact pin |
| --- | --- |
| Arena (main) | `196680c28aa53e173d97d9e2dbded9d8d4f981a0` |
| Isaac Lab (`release/3.0.0`: v3.0.0 GA plus backports) | `39394453ceaac256349ad822e675a93485373667` |
| Isaac Sim | `6.2.0.0` |
| Newton (the commit Isaac Lab `main` pins) | `d31603ff5e8389d1d6b2a5daef6ea2cb074c26ff` |
| Numerical Sim (public PR #17 + catheter runtime capabilities) | `cfc05e0d20dda2a841fdf935b666f47b27163554` |
| Catheter solver (same checkout) | `5af32b137680af4500a4f3e5b104a2e77c44d36b` |
| MuJoCo / MuJoCo-Warp | `3.17.0` / `3.17.0` |
| Torch / Torchvision | `2.11.0+cu130` / `0.26.0+cu130` |
| Warp | `1.17.0`, operator-supplied patched Linux x86-64 wheel |

The qualification lock supports Linux x86-64 only. Numerical Sim and the legacy
solver use separate source directories. Do not point one source at the other.
The Lab checkout uses the exact `release/3.0.0` commit with tracked, idempotent patches.

## Latest-upstream bump (November 19)

This bump moves each pin to the newest upstream commit that the rest of the stack
can run:

| Component | From | To | Upstream delta |
| --- | --- | --- | --- |
| Arena | `813be56` | `196680c` (main) | 7 commits: experiment runner, benchmarks, docs. Removes the legacy `EnvironmentRegistry.get_environment_cfg_type`, which i4h does not use. Arena still pins the GA Lab submodule. |
| Isaac Lab | `e3f7d9f` (v3.0.0) | `3939445` (`release/3.0.0`) | Backports: sensor timing drift, MJWarp sparse-state resets, scalar joint friction writes, PhysX Fabric camera poses. Isaac Sim, Warp, MuJoCo and Newton pins unchanged. No tag newer than v3.0.0 exists. |
| Newton | `7f0c13a` | `d31603f` | 141 commits, still 1.9.0.dev0 with MuJoCo 3.17 and Warp >=1.17. Includes VBD/rod cleanup (#5259) and rod mass unification (#5328). |

Relocking changed only the Lab editable packages and Newton. Isaac Sim, Torch,
Warp, MuJoCo, MuJoCo-Warp and the Warp wheel/native hashes are unchanged. All
five Lab patches and the Arena WBC patch apply unchanged. The VBD patch is still
required, because Newton `d31603f` raises on `rigid_compliant_alm=None`.
`ClampingBase` and `SensorTiledCamera` remain available. Numerical Sim does not
use the removed start-origin rods.

Not taken:

- **Newton main** (`b6a2289`) removes `SensorTiledCamera`, which Lab's
  `newton_warp_renderer.py` and Numerical Sim's `i4h_isaaclab/camera/tiled.py`
  use. It also moves to `mujoco-core`/`mujoco-warp` 4.0.1 and locks a Warp
  1.20 dev build, which conflicts with the 3.17 pins and the patched 1.17 wheel.
- **Isaac Lab main** (1068 commits past GA) is not what Arena main targets.
- **Numerical Sim public main** (`86e2fd8`) is older than `d1fc95c` and pins
  Newton 1.6.1. `d1fc95c` came from branch `i4h-isaaclab/manager-env`, which
  no longer exists on the remote. A fresh clone therefore depends on GitHub
  still serving the unreferenced SHA. Push that commit to a retained branch or
  tag before relying on clean setups.

Gates on the bumped stack (evidence
`runs/endoluminal_aorta_fluoroscopy/20261119_184500/`, serial visible runs,
driver `visible_gates.sh`, exit codes in `gates.tsv`):

| Check | Result |
| --- | --- |
| Setup (`--locked`), `verify-asset`, `verify-runtime` | passed. Newton `d31603f`, host API `lab30`, Warp native hash unchanged |
| Light / Arena / full RL suites | 815 passed, 2 skipped / 1096 passed, 1 skipped / 210 passed, 1 skipped (the whole `rl` suite, not the original 39-test selection) |
| Exact two Numerical Sim runtime modules | 26 passed, 3 skipped. No new warning categories |
| Catalog lint | all 17 Workflows |
| Gear preview | 1000 steps, exit 0 |
| Coexistence, both orders, shared initially empty Warp cache | passed, finite state |
| Surgical reach | succeeded at step 76, 150-step cap |
| G1 / H2 cloth | each: one surface, 120 physics steps, finite particles, reset passed |
| Static-aorta demos x2, calibration | same values as the original table. `S_reset = S_tip = 0`, worst marker 1.348 px |
| Static-aorta launcher replay + replay-check | 115/115 samples and frames, identical actions, 0 m tip error |
| Legacy baseline replay (`2ab93f52` recording) | 192 samples, four channels, success, finite, actions identical to baseline |
| Legacy direct / arm demos | each: 192 samples, four channels, success, finite, 256 by 256 fluoroscopy |
| Legacy direct demo repeat | identical to the first bumped-stack demo in every dataset |

The static-aorta demo and replay HDF5 files are byte-identical to the original
campaign's, so the bump does not reach the numerical `aorta_static` path. In the
legacy recordings, actions, states and every non-fluoroscopy dataset are
identical to the original campaign's. Only fluoroscopy RGB pixels differ:

- Direct demo and baseline replay: 16 of 192 frames (127-160), at most 64 of
  65536 pixels per frame, max delta 56/255.
- Arm demo: 41 of 192 frames (121-188), at most 45 pixels per frame, max delta
  57/255.

The change is deterministic: a same-stack repeat is identical. It is
rendering-only, and no Lab or Newton commit has been isolated as its cause.
`legacy-demos.json` and `legacy-recording-replay.json` in the evidence folder
record paths, hashes and these comparisons. The first replay-check attempt failed only because the driver passed a
relative `--record` path, which `run.sh` nested under the run directory. The
check passed on the actual recording, and the command above now uses an
absolute path.

Follow-up catalog on the bumped stack: serial visible launches, seed 0, one
requested episode, up to three attempts, original caps, default PhysX scenes.
Learned policies use the loopback `ZENOH_CONFIG` below. Exit codes and run
directories are in `catalog.tsv`, `catalog-policy.tsv` and
`catalog-policy-repeat.tsv`. Recording checks (sample counts, success flag,
finite floats, RGB cameras) are in the adjacent `*-inspection.json` files.

| Workflow | Mode | Attempts | Recorded samples (original) | Note |
| --- | --- | --- | ---: | --- |
| surgical_lift_block | rule-based | 1 | 201 (201) | |
| surgical_lift_needle | rule-based | 1 | 201 (201) | |
| surgical_lift_needle_organs | rule-based | 1 | 201 (201) | runtime only; room camera occlusion unchanged |
| surgical_reach_dual_psm | rule-based | 1 | 75 (75) | |
| surgical_reach_star | rule-based | 1 | 47 (47) | |
| scissor_pick_and_place | rule-based | 1 | 533 (533) | |
| ultrasound_liver_scan | rule-based | 1 | 244 (244) | tiled renderer; no startup repair needed |
| assemble_trocar | GR00T N1.5 | 1 | 169 (164) | three cameras |
| locomanip_push_cart | GR00T N1.6 | 1 | 1498 (1519, on attempt 2) | |
| locomanip_tray_pick_and_place | GR00T N1.6 | 1 | 1205 (1231) | |
| scissor_pick_and_place | GR00T N1.5 | 1 | 600 (595) | accepted `timeout_success` at the 600-step cap |
| scissor_pick_and_place repeat | GR00T N1.5 | 1 | 586 (595) | graph completed at step 588 |

All runs exited 0 with success, finite data and their declared RGB cameras. The
learned-policy runs differ in length from the original campaign. These PhysX
scenes run with `deterministic=False`, and the two scissor runs on the same stack
and seed differ by more than that. The first scissor run did not finish its graph
before the cap; it was accepted because the success predicate held at timeout.
These are smoke successes, not robustness estimates. The scissor policy's margin
to the 600-step cap remains small. Probe reach, alternative N1.7/OpenPI modes,
XR/teleoperation and the `aorta_shell` impulse deviations were not re-run.

## Setup

Current setup installs stock `warp-lang==1.17.0` from the package index. No local wheel or patched-native hash is required:

```bash
I4H_SETUP_PROJECTS=arena I4H_THIRD_PARTY_TARGET=arena ./setup.sh
export I4H_DATA="$PWD/third_party/i4h-physics-simulation/numerical_sim/data"
python3 scripts/arena_stack_artifacts.py verify-asset
OMNI_KIT_ACCEPT_EULA=YES arena/.venv/bin/python scripts/arena_stack_artifacts.py verify-runtime
```

Setup uses `uv sync --locked` for Arena and checks that Warp's native library imports. Native hashes are reported as provenance. The captured-HashGrid defect is a [known upstream issue](https://github.com/NVIDIA/warp/issues/2426), with opt-in diagnostics described in [Troubleshooting](../../TROUBLESHOOTING.md#warp-hashgrid-queries-return-incorrect-results-after-cuda-graph-replay).

The historical qualification results above used the following local artifacts. These hashes identify those runs; they are not current setup requirements:

| Artifact | SHA-256 |
| --- | --- |
| Warp wheel | `dff334c1589510f4374807eb3fbdc85cfa5e20ab8a0e05548dc54e0b64990546` |
| Loaded `warp.so` | `eb4bda4537f7c7a35ef4640d89fbfba96cc9b2b786fdf1591535ea420ec8b6eb` |
| `AortaWithVesselsStatic.usdc` | `b9eeaa91038c261ad10364e522837be3225c9432ad3740b45941f0676abde91c` |

## Patches

- `isaaclab_newton19_metadata.patch`: port `e53cd288` and the pinned numerical
  cloner patch to GA. Read surface/volume metadata and recognize rod joints.
- `isaaclab_newton19_actuators.patch`: Newton renamed its actuator clamping base
  class to `ClampingBase`. Use the new class in GA's native-actuator adapter.
  The unpatched gear preview fails while importing the articulation backend.
- `isaaclab_newton19_deformables.patch`: remove GA's private cloth/soft group
  recording calls; Newton 1.9 records surface/volume groups while adding meshes.
  Keeping both registrations would duplicate the groups.
- `isaaclab_newton19_vbd.patch`: omit an unspecified `rigid_compliant_alm`;
  Newton 1.9 requires this boolean argument to be omitted to select its default.
- Retain the Pink local mesh-directory, legacy solver ModelBuilder lookup,
  LeIsaac HDF5/CUDA and ActionTermCfg export, and Arena WBC height patches.
- Arena main already contains the former slice-reset, per-modality image term,
  and legacy Franka fixes; setup no longer applies those three obsolete patches.

`Scene.gym_spec()` already uses the typed builder and registered factory's
three-value return. The RSL-RL registration path now uses `build_env_cfg()` in
place of the removed `compose_manager_cfg()`, retaining runtime kwargs and
clearing the inherited recorder before registration. Custom embodiments now declare
Arena's required `registry_name`, and ConfigAsset implements `get_event_cfgs()`.
The isolated GR00T N1.6 service retains its original Arena checkout and unchanged
joint configuration files; setup stages that checkout with policy sources.

## Reproduce the gates

Use `arena/.venv/bin/python` throughout. Keep logs, commands and exit codes in a
fresh `runs/<workflow>/<YYYYMMDD_HHMMSS>/` directory. Cold kernel and shader
compilation can take several minutes. Run visible simulator processes sequentially.

```bash
arena/.venv/bin/python -m pytest \
  third_party/i4h-physics-simulation/numerical_sim/tests/runtime/test_lab_adapters.py \
  third_party/i4h-physics-simulation/numerical_sim/tests/runtime/test_component.py -q -ra

OMNI_KIT_ACCEPT_EULA=YES arena/.venv/bin/python \
  third_party/IsaacLab-Arena-196680c/isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 1000 --num_envs 1 --device cuda:0 \
  --viz kit --enable_cameras gear_insertion

# Choose an initially nonexistent cache and reuse it in the second process.
export WARP_CACHE_PATH="$PWD/runs/endoluminal_aorta_fluoroscopy/<timestamp>/coexistence-cache"
arena/.venv/bin/python scripts/validate_catheter_coexistence.py legacy-first
arena/.venv/bin/python scripts/validate_catheter_coexistence.py numerical-first

./run.sh surgical_reach_psm --rule-based --seed 0 --episodes 1 --attempts 1 \
  --run-dir runs/surgical_reach_psm/<timestamp>
OMNI_KIT_ACCEPT_EULA=YES arena/.venv/bin/python scripts/validate_tablecloth_stack.py spread_tablecloth_g1
OMNI_KIT_ACCEPT_EULA=YES arena/.venv/bin/python scripts/validate_tablecloth_stack.py spread_tablecloth_h2
```

Cloth smokes load each normal Scene and AppContext with XR disabled. They drive
the Scene's neutral actuation, require one nonempty Newton surface per world,
check finite particles, and reset after at least 120 physics steps.

The legacy recording baseline lives in the separate Workflows `2ab93f52`
worktree on Arena `0a1b8c2` and Sim `6.0.1.0`. Run its seed-0 demo there; replay
that recording on the upgraded stack before adding the new Scene. Keep both
recordings and decode their fluoroscopy frames. Legacy demos still use four
physical action channels, 30 Hz control and the original 600-step caps.

The shared Replay Task now rejects a recording whose width differs from the
destination actuation before emitting any command. Its error gives expected
and received widths.

The `aorta_shell` impulse deviations from the numerical campaign remain separate
from this milestone's `aorta_static` position/reset/replay qualification. This
upgrade does not waive them or qualify XR, policies, or the full Scene catalog.

## Static-aorta acceptance results

Two independent fresh-process captured demos calibrated the thresholds before a
third, independently launched replay. Each demo retained the pristine reset,
first zero-action step, all 115 commands, and reset plus zero-action step after
the demonstration. The shared SimulationRunner owns the demo steps and recording;
the qualification harness samples its output and performs the post-demo reset.

| Measurement | Observed result | Acceptance |
| --- | ---: | ---: |
| Insertion target advance | 0.49999985098838806 m | 0.50 +/- 0.001 m |
| Tip displacement from settled position | 0.5168598294258118 m | at least 0.05 m |
| `S_reset` | 0 m | calibration spread |
| `T_reset = max(1e-5, 3*S_reset)` | 0.00001 m | at most 0.0001 m |
| Post-demo reset / first-step error | 0 m / 0 m | within `T_reset` |
| `S_tip` | 0 m | calibration spread |
| `T_replay = max(1e-5, 3*S_tip)` | 0.00001 m | at most 0.001 m |
| Worst live-tip marker alignment | 1.3479979178714308 px | at most 2 px, all samples inside detector |
| Third-process replay tip error | 0 m | within frozen `T_replay` |
| Replay samples / decoded RGB frames | 115 / 115 | 115, 256 by 256 |
| Replayed actions including final action | identical | exact array equality |

All state and diagnostics were finite. Reset command targets returned to the
authored zero baseline. The detector projection is fixed:
`u = 255*(x+0.5)/11.5`, `v = 255*(2-y)/4`, dropping Z. Pixel alignment and metric
motion/replay errors are evaluated separately. The sensor uses the synchronized
component polyline with the demo-catheter fallback disabled. `fluoroscopy` records
the raw image, and the visible sensor panel retains the guidance output.

Reproduce in order, using a fresh timestamp directory for every process:

```bash
arena/.venv/bin/python scripts/qualify_aorta_fluoroscopy.py demo --run-dir runs/endoluminal_aorta_fluoroscopy/<first>
arena/.venv/bin/python scripts/qualify_aorta_fluoroscopy.py demo --run-dir runs/endoluminal_aorta_fluoroscopy/<second>
arena/.venv/bin/python scripts/qualify_aorta_fluoroscopy.py calibrate \
  runs/endoluminal_aorta_fluoroscopy/<first> runs/endoluminal_aorta_fluoroscopy/<second> \
  --output runs/endoluminal_aorta_fluoroscopy/<evidence>/calibration.json
./run.sh endoluminal_aorta_fluoroscopy \
  --replay runs/endoluminal_aorta_fluoroscopy/<first>/demo.hdf5 --seed 0 \
  --record "$PWD/runs/endoluminal_aorta_fluoroscopy/<replay>/replay.hdf5" \
  --run-dir runs/endoluminal_aorta_fluoroscopy/<replay>
arena/.venv/bin/python scripts/qualify_aorta_fluoroscopy.py replay-check \
  --calibration runs/endoluminal_aorta_fluoroscopy/<evidence>/calibration.json \
  --source runs/endoluminal_aorta_fluoroscopy/<first>/demo.hdf5 \
  --replay runs/endoluminal_aorta_fluoroscopy/<replay>/replay.hdf5 \
  --output runs/endoluminal_aorta_fluoroscopy/<evidence>/replay-check.json
```

`run.sh` resolves a relative `--record` path against `--run-dir`, so pass an
absolute recording path as above.

The demo harness supports `--eager` for separate diagnostic runs; eager data is
rejected by captured-stack calibration. The qualified runs used graph capture.
The normal visible launcher replay also passed. Start, insertion-end and reset
PNG frames are retained with each independent demo.

## Test counts and evidence

| Check | Result |
| --- | --- |
| Catalog discovery and lint | all Workflows/modes resolve, including all three new modes |
| Light contract suite | 815 passed, 2 skipped (Torch absent in the lightweight environment) |
| Arena suite | 1096 passed, 1 skipped (requested bend exceeds the mapping's faithful range) |
| RSL-RL wiring and carried RLinf regressions | 39 passed |
| Exact two Numerical Sim runtime modules | 26 passed, 3 skipped (later `newton_solver` host API absent on GA) |
| Documented gear preview | 1000 steps, exit 0 |
| Coexistence, both import/host orders | passed, shared initially empty Warp cache, finite positions and velocities |
| Surgical reach | 75 steps, success within 150-step cap |
| G1 / H2 cloth | each: one 2523-particle surface, 120 physics steps, finite particles, successful reset |
| Legacy baseline recording / upgraded replay | each: 192 samples, four channels, success, finite diagnostics, decoded 256 by 256 fluoroscopy |
| Legacy direct / arm demo on upgraded stack | each: 192 samples, four channels, success, finite diagnostics, decoded fluoroscopy |

Local evidence is under `runs/endoluminal_aorta_fluoroscopy/20261119_125300/`.
The accepted independent demos are `20261119_133500` and `20261119_133600`; the
normal launcher replay is `20261119_133700`. The evidence folder contains a full
report, command records, patch/artifact hashes, runtime paths, calibration,
comparisons and first-failure reproducers. Earlier failures and interrupted
parallel attempts are retained separately and excluded from qualification.

The baseline Workflows worktree is `legacy-baseline-2ab93f52`, with its recording
under `runs/endoluminal_navigation/20261119_130500/`. Its runtime uses Isaac Sim
6.0.1.0, Newton 1.2.1, Warp 1.13.0, Torch 2.10.0+cu128, MuJoCo 3.8.0 and
MuJoCo-Warp 3.8.0.3. Setup refreshed baseline lock metadata and added four
JSON-schema dependencies without changing any locked package versions; the
retained lock and package manifest document the exact baseline environment.

## Retained impulse deviations

The pinned numerical `docs/stage-c21-closure.md` retains earlier `aorta_shell`
impulse discrepancies separately from this static-aorta position campaign:

| Historical comparison | Absolute impulse difference | Unchanged tolerance |
| --- | ---: | ---: |
| Standalone eager | 6.792339445382822e-5 | 4.6263936383184046e-5 |
| Main eager | 7.092467785696499e-5 | 4.6263936383184046e-5 |
| Main captured, stepped and masked reset | 8.587960292061325e-5 | 4.6263936383184046e-5 |

Those observations remain open in their original evidence. Their original
worktree and artifacts were preserved. This milestone establishes the bounded
position, detector-alignment, reset and replay results above.

## Follow-up catalog qualification (November 19)

The follow-up campaign uses visible, serial launcher runs in the same worktree,
with seed 0, one requested episode and up to three attempts. The additional
runs use the scenes' default PhysX backend on the upgraded stack; they do not
establish Newton-preset equivalence for every scene.

The block and needle lifts, dual-PSM and STAR reaches, scissor controller and
ultrasound scan passed their recorded rollouts. The organs needle-lift also
reported success with finite state/actions, but its room camera is occluded
by the torso, leaving visual outcome qualification incomplete.

The ultrasound rig now selects Arena's tiled renderer because it requests
RGB, depth and segmentation. The independent renderer accepts RGB only and
caused the first launch to fail. The same controller then passed at the
original step cap. Focused CPU checks passed (26 tests), as did pinned static
checks and catalog lint.

Default Zenoh discovery failed between the isolated policy and simulator
environments on this host. Cross-environment probes passed with explicit TCP
or multicast restricted to loopback. The learned-policy qualification runs use:

```bash
export ZENOH_CONFIG="$PWD/runs/surgical_lift_block/20261119_140424/zenoh-loopback.json5"
```

That file contains `{"scouting":{"multicast":{"interface":"lo"}}}`.
It is for same-host services; use the explicit policy endpoint for a remote
server. Trocar assembly passed after applying this transport configuration,
with all five milestones and 164 recorded frames.

Exact run directories, checkpoint revisions, outcomes, retained failures,
recording inspections and contact sheets are indexed in
`qualification-report.md` (retained locally at `runs/surgical_lift_block/20261119_140424/qualification-report.md`)
and its adjacent `recording-inspection.json`. The original upgrade evidence
above remains a separate snapshot. Probe reach was skipped at the user's
request because no exported checkpoint was available. Alternative N1.7/OpenPI
modes, XR/teleoperation and broad replay qualification are not claimed here.

The learned tray-placement and scissor policies also passed with clean exits
on their first attempts (1231 and 595 recorded samples). The cart's clean-exit
repeat succeeded on attempt 2, with 1519 samples; attempt 1 hit the unchanged
1800-step cap and is retained as a failure. Its earlier 1423-sample successful
run has an invalid helper exit capture and is excluded from clean-exit
qualification. Thus cart success was observed, but reliable first-attempt
success is not established. All recorded RGB frames were decoded, and action,
state, initial-state and segment checks passed, including the failed attempt.
