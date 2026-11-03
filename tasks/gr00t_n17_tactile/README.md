# gr00t_n17_tactile

GR00T N1.7 with fingertip tactile conditioning for the Unitree H2 with
SharpaWave hands, on the real robot. The narrative, the data format, the
training and deployment flow and the roadmap are in
[docs/workflows/h2_tactile.md](../../docs/workflows/h2_tactile.md); this file
is the project map.

## Layout

```text
tasks/gr00t_n17_tactile/
├── pyproject.toml                       own venv; pins the patched Isaac-GR00T checkout
├── i4h_tasks/gr00t_n17_tactile/
│   ├── manifest/h2_sharpa_tactile.yaml  the one task: embodiment + sensor configuration
│   ├── config.py                        modality config incl. `tactile` and `tactile_force`
│   ├── server.py                        GR00T PolicyServer with tactile options + server_info
│   ├── train.py                         finetune launcher argv from the manifest; --dry-run
│   ├── tactile/
│   │   ├── interface.py                 TactileSource / TactileEncoder / FingertipKinematics
│   │   ├── sharpa.py                    SharpaWave implementations (public sources)
│   │   └── sensor/{wire,subscriber}.py  ZMQ stream from the robot-side bridge
│   └── client/
│       ├── protocol.py                  ZMQ client for Isaac-GR00T's PolicyServer protocol
│       ├── observation.py               observation builder: cameras, joint state, tactile maps, force window
│       ├── execution.py                 chunk execution, smoothing, async / same-plan workers
│       ├── robot.py                     H2 + SharpaWave + camera adapters over the teleop checkout; FakeRobot
│       └── deploy.py                    the deployment CLI (i4h-tasks-gr00t-n17-tactile-deploy)
├── patches/README.md                    what the GR00T patch series changes
└── tests/                               light-venv tests (no gr00t, no torch)
    └── fake_policy_server.py            PolicyServer stand-in for smokes without a robot (not collected)
```

| Part | State |
| --- | --- |
| Manifest, modality config, sensor contracts, wire format, subscriber, live source, `server_info`, trainer argv | complete |
| Deform encoder, fingertip kinematics, force encoder wrappers | complete; thin wrappers over the modules of the patched checkout |
| Policy loading, server binding, finetune launch | complete; need the checkouts that `setup.sh` creates |
| Deployment client (cameras, robot state, action execution on the H2) | complete; run on the robot from this checkout (check mode, then synchronous and asynchronous episodes of the pads checkpoint on the standing H2, 2026-11-03) and against the served policy with `--fake-robot`; runs in a ROS 2 + Unitree DDS Python 3.10 environment, see [Deployment client](#deployment-client) |

## Setup

```bash
# The patched Isaac-GR00T checkout, the public sources of the frozen encoders
# and the hand URDFs, and the teleoperation checkout plus Unitree Python SDK
# that the deployment client drives the robot through; all pinned by revision
# (see Dependencies).
./third_party/setup.sh tasks/gr00t_n17_tactile
uv sync --project tasks/gr00t_n17_tactile
```

`setup.sh` creates `third_party/Isaac-GR00T-1.7-tactile` (the N1.7 revision
`tasks/gr00t_n17` pins, plus `third_party/gr00t_n17_tactile.patch`),
`third_party/sharpawave-deform-encoder`, `third_party/T-Rex` (its
`tactile_vqvae` package only) and `third_party/sharpa-urdf-usd-xml` (`wave_01`
only). Neither encoder repository is installed as a package: the Sharpa one
ships no `pyproject` and T-Rex's pins a full VLA stack (Python 3.10, torch
2.6), so the server and the trainer put both on the import path.

For the deployment client it also creates `third_party/xr_teleoperate` (the
official `unitreerobotics/xr_teleoperate` at `ebe257f0`, its `teleop/`
directory only, plus `third_party/xr_teleoperate_h2_sharpa.patch`) and
`third_party/unitree_sdk2_python` (`794fb2b3` plus
`third_party/unitree_sdk2_network_config.patch`). Neither goes into the task
venv; the client runs in its own environment (see
[Environment](#environment)).

## Commands

```bash
# What a client learns on connect, resolved from the manifest alone (the served
# version is corrected by the loaded checkpoint: its cameras, fingertips, force window).
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile --print-server-info

# The exact launcher invocation for a dataset, without running it.
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile-train \
  --dataset /data/h2_sharpa_pads_lerobot --dry-run

# Finetune on a LeRobot v2.x dataset with the tactile columns.
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile-train \
  --dataset /data/h2_sharpa_pads_lerobot --output-dir /ckpt/h2_sharpa_pads

# Serve a checkpoint to the robot. The tactile bridge runs on the onboard computer.
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile \
  --checkpoint /ckpt/h2_sharpa_pads --tactile-host <robot-onboard-computer> \
  --tactile-refresh per-step [--cascade --tactile-source none] [--disable-tactile]

# Drive the robot from the served checkpoint: see Deployment client below.

# Tests: pure Python, run in the light `workflows` venv (no model stack needed);
# also part of the light suite in scripts/run_tests.py.
uv run --project workflows --extra dev pytest tasks/gr00t_n17_tactile/tests
```

`--tactile-refresh per-step` re-reads the live stream before every denoising
step; `per-chunk` reads it once per request. `--cascade` re-plans the tail of
every chunk from the tactile data in each client observation (a checkpoint
trained with delay augmentation is required) and does not read the live stream.
`--disable-tactile` serves a tactile checkpoint with its tactile path switched
off: the vision-only ablation on the same weights. The switch is inside the
model, so the client must still send the tactile and force keys the
checkpoint was trained with (the policy's strict input check requires them);
`server_info` lists them under `required_observation_keys` and repeats this
note. A live stream is not opened in this mode.

The server waits two seconds at start-up for every fingertip of the checkpoint
to appear on the live stream and refuses to serve if one stays silent, naming
it; a fingertip that goes quiet later is logged (at most every five seconds)
and the previous frame is kept.

## Deployment client

`client/deploy.py` is the program on the workstation that reads the H2's
cameras and joint state, subscribes to the tactile stream, sends one
observation per chunk to the policy server and executes the returned action
chunks on the arms and hands. Its flags are those of the N1.7 client it was
ported from, so existing launch recipes carry over; two legacy paths are not
in it because the pinned third-party set does not carry them: the ZMQ camera
server (`--image-source zmq`; ROS 2 is the camera path) and the direct
SharpaWave SDK hand path (`--hands-source sdk`; the hands are driven through
the robot-side DDS bridge).

### Environment

The client talks Unitree DDS and ROS 2, which the task venv cannot (Python
3.11, no ROS), so it runs in a separate Python 3.10 environment:

- ROS 2 Humble sourced (`rclpy`, `sensor_msgs`, `rmw_cyclonedds_cpp`);
- `pip install -e third_party/unitree_sdk2_python`, which brings
  `cyclonedds==0.10.2` through its `setup.py`;
- `pip install logging_mp pyzmq msgpack numpy PyYAML opencv-python-headless`
  (`logging_mp` is imported by the teleop arm controller, `opencv` only by
  `--check`).

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=tasks/gr00t_n17_tactile:third_party/xr_teleoperate:$PYTHONPATH \
  python -m i4h_tasks.gr00t_n17_tactile.client.deploy --help
```

Keep `:$PYTHONPATH` at the end: ROS 2's setup script delivers `rclpy` through
`PYTHONPATH`, and an assignment without it drops ROS from the interpreter.
`XR_TELEOPERATE_PATH` overrides the location of the teleop checkout (default
`third_party/xr_teleoperate`, resolved from the repository root). The console
script `i4h-tasks-gr00t-n17-tactile-deploy` is the same entry point in the
task venv, usable there for the `--fake-robot` smokes only.

The Unitree SDK patch adds a `networkConfig` keyword to
`ChannelFactoryInitialize`: the client passes a CycloneDDS XML that names the
workstation's address on the robot network and the exact peers (the onboard
computer and the two H2 body participants, discovered through the onboard
computer unless `--body-dds-peer` names them), and the SDK uses it as the DDS
configuration instead of its auto-determined or interface-name template. That
is how a workstation with several network interfaces reaches the robot
without multicast; `--network-interface` bypasses it and uses the SDK's own
template.

### Two terminals

```bash
# Terminal A: the server (task venv, GPU).
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile \
  --checkpoint /ckpt/h2_sharpa_pads --tactile-host <robot-onboard-computer> \
  --tactile-refresh per-step

# Terminal B: the client (ROS 2 + Unitree DDS environment).
PYTHONPATH=tasks/gr00t_n17_tactile:third_party/xr_teleoperate:$PYTHONPATH \
  python -m i4h_tasks.gr00t_n17_tactile.client.deploy \
  --tactile-host <robot-onboard-computer> --task "pick up the gauze pad" --motion \
  --camera-network-address <workstation-ip-on-camera-network> \
  --camera-network-peer <onboard-computer-ip-on-camera-network> \
  --init-state-yaml /path/to/init_state.yaml --steps 300
```

The client connects the hands (they move to neutral), the arm controller
(its constructor drives both arms to the zero pose and the head to
`--head-pitch-home` as soon as it starts) and the cameras, drives the arms
home and holds (`--hold-s`), stages the init
posture from the YAML (`--init-state-hold-s`), connects to the server,
calibrates the tactile sensors (below) and waits for Enter before the first
chunk. Each episode runs `--steps` control steps, then home and init posture
again. Ctrl+C stops the loop, drives the arms home and releases the hands.

`--head-mode` defaults to `mono`, the `/head/left` camera alone, which is
what the three-camera launcher on the onboard computer publishes and what
the `high` view of the checkpoints was recorded from; `binocular` reads the
stitched `/head/left` + `/head/right` pair for eye-view checkpoints and needs
the fourth camera running. `--motion` publishes the arm commands on `rt/arm_sdk` for a standing robot
(default `rt/lowcmd`). `--steps` defaults to 30, one second at 30 Hz, so a
real run passes its own value. `--right-arm-hold-qpos` fixes the right arm
for a single-arm checkpoint (default: the YAML's right arm, else the measured
pose after going home). `--no-arm` and `--no-hands` run the plumbing without
commanding that part; `--verbose` enables debug logging.

### What the client derives from `server_info`

On connect the client reads the server's `server_info` and configures itself
from it, so nothing can drift from the checkpoint actually being served:

| From `server_info` | Decides |
| --- | --- |
| `required_observation_keys.video` (else `video_modality_keys`) | which head and wrist views are sent; eye views select `--head-eye both` |
| `required_observation_keys.state` (else `state_groups`) | which hand state keys are sent; a single-hand checkpoint gets one hand |
| `tactile.fingertips`, `checkpoint.use_tactile` | the deform keys and the fingertips the live source subscribes to |
| `tactile.force_window`, `checkpoint.use_tactile_force` | whether the force window is sent and its length |
| `control_hz`, `execution_steps`, `action_horizon` | the control rate and the steps executed per chunk |
| `cascade.enabled`, `checkpoint.force_delay_max` | whether same-plan refines are available and how far they may lead |

The explicit flags `--wrist-cams`, `--head-eye`, `--state-hands`,
`--checkpoint-tactile` / `--no-checkpoint-tactile`,
`--checkpoint-tactile-force` / `--no-checkpoint-tactile-force`, `--frequency`
and `--action-horizon` override the server. The two checkpoint flags log a
warning when they contradict `server_info`; the others replace the server's
choice silently (`--action-horizon` warns only when it exceeds the served
`action_horizon`). `--state-hands` selects the hand state keys and, with them,
the deform keys and force columns of that hand. Against a server without the endpoint the legacy
defaults apply with a warning: tactile keys on, force window off, both hands,
`right_wrist_view` only.

The tactile and force keys are sent whenever the checkpoint requires them:
with `--no-tactile` (the ablation arm) they are zeros, and when the server runs
`--disable-tactile` they are still required by the policy's input check. With
the server's `--tactile-source live` the server re-reads its own stream before
every denoising step and falls back to the request's maps when its stream is
incomplete, so a client without `--tactile` is warned.

### Check mode

```bash
PYTHONPATH=tasks/gr00t_n17_tactile:third_party/xr_teleoperate:$PYTHONPATH \
  python -m i4h_tasks.gr00t_n17_tactile.client.deploy --check --check-dir /tmp/check \
  --tactile-host <robot-onboard-computer> \
  --camera-network-address <workstation-ip-on-camera-network> \
  --camera-network-peer <onboard-computer-ip-on-camera-network>
```

`--check` builds one observation and never contacts the server, but it
connects the robot like a run: the hands move to neutral, and the arm
controller drives both arms to the zero pose and the head to
`--head-pitch-home` as soon as it starts. For a camera and tactile check with
no robot motion add `--no-arm --no-hands`. It prints every field (shape,
dtype, range), writes `check_<key>.png`
per camera and `check_tactile_<key>.png` per fingertip into `--check-dir`, and
logs the nonzero-pixel count and maximum of each tactile map. An all-black
fingertip image means the bridge on the onboard computer is not running or
the host and port are wrong. Without a server the check resolves with the
legacy defaults, so pass `--wrist-cams`, `--state-hands` and
`--checkpoint-tactile[-force]` as the checkpoint needs them.

Wrist-camera routing: by default the `/wrist/left` stream becomes
`left_wrist_view` and `/wrist/right` becomes `right_wrist_view`, which is how
the training data of the current checkpoints was recorded through the ROS
pipeline, so identity routing reproduces training (verified by camera
fingerprint against training frames). On this rig that means
`right_wrist_view` shows the left hand, because the camera on the left wrist
is wired as the right stream; it is what the policies expect. Confirm it in
check mode before every live run: `check_right_wrist_view.png` must show the
left hand. `--swap-wrist-cams` exchanges the two streams; it exists for the
legacy checkpoints recorded through the ZMQ camera server.

### Real-time chunking

By default the control loop is synchronous: capture an observation, wait for
the server, execute the chunk, so the robot idles during inference.
`--async-policy` moves inference to a background thread that keeps producing
chunks; the loop takes the newest cached chunk and skips the steps whose
execution time already passed during inference. Consecutive chunks are
sampled independently, so switching between them can jerk; these controls
soften the boundary:

| Flag | Default | Effect |
| --- | --- | --- |
| `--async-policy` | off | run inference concurrently with execution |
| `--async-cache-poll-s` | 0.005 | how often the loop polls for a fresh cached chunk |
| `--smooth-alpha` | 1.0 | EMA on per-step targets, `alpha * raw + (1 - alpha) * prev`; 1.0 = off, 0.3-0.6 quiets jitter at the cost of lag |
| `--boundary-smooth-steps` | 0 | ticks after a chunk switch that use `--boundary-smooth-alpha` instead of `--smooth-alpha`; typical 3-6 |
| `--boundary-smooth-alpha` | 1.0 | heavier EMA during that window; lower = stronger crossfade |
| `--boundary-blend-steps` | 0 | cross-chunk blend: fades the old chunk's unexecuted future into the new chunk's first steps; typical 5-8, needs a chunk longer than `--action-horizon` |
| `--boundary-blend-alpha` | 2.0 | decay of the arm blend, `w_old(t) = exp(-alpha * t / N)`; the hands blend linearly |

Two T-Rex-style execution modes refresh the unexecuted tail of the running
chunk with fresher tactile data instead of waiting for the next boundary:

- **New-plan tail refresh** (`--trex-action-execution`, async only): every
  `--trex-refine-every` steps the tail is overwritten, time-aligned, from the
  newest independent chunk the worker produced, without resetting the march
  through the chunk.
- **Same-plan refines** (`--trex-refine-mode same-plan`; requires
  `--async-policy` and a server started with `--cascade`): one slow pass plans
  each chunk and, at plan indices that are multiples of `--trex-refine-every`
  up to `--trex-max-refine-offset`, the client asks the server to re-run the
  fast tail of the same cached plan with the fresh tactile data and splices
  the result into the tail (overwrite, or an EMA with `--trex-refine-blend`
  and `--trex-refine-blend-alpha`). `--trex-slow-lead` submits the next slow
  pass that many ticks before the boundary so it is ready in time;
  `--trex-contact-gate` fires refines only while the left hand's largest force
  magnitude is at or above `--trex-contact-threshold`. Keep
  `--trex-max-refine-offset` within the checkpoint's `force_delay_max`; the
  client warns when it is not.

Bring-up order: run synchronously first, add `--async-policy`, then
`--boundary-blend-steps 5` (a 32-step chunk against 16 executed steps leaves
room to blend), then a refine mode.

### Tactile calibration

The fingertip sensors are tared before every episode through the onboard
computer: over SSH the client runs
`docker exec -w /workspace <container> python3 calibrate_sharpa_tactile.py`
(`--tactile-calib-ssh`, default `unitree@<--tactile-host>`;
`--tactile-calib-container`, default `sharpa180`), retries
`--tactile-calib-attempts` times and settles `--tactile-calib-settle-s`
after success so the calibrated frames reach the stream. The hands must be
stationary with no fingertip touching anything, which the home and init
postures provide. SSH authentication is key-based, or set `THOR_SSH_PASS` with
`sshpass` installed. When every attempt fails the client asks: Enter retries,
`skip` continues uncalibrated, `q` quits. `--tactile-calib` /
`--no-tactile-calib` force it; the default is on when tactile keys are sent
and `--tactile` is on.

### Episode scoring

`--num-episodes N` stops after N labelled episodes (default: until Ctrl+C) and
`--log-results <file.jsonl>` appends one record per episode. After each
episode the client asks `[s]uccess / [f]ail / [a]bort / [r]etry`: `s` and `f`
log and continue, `a` logs and ends the run, `r` repeats the episode without
logging. A record holds `timestamp`, `episode_idx`, `tactile` (whether the
live stream was fed), `task`, `steps` and `label`, so the tactile and
`--no-tactile` arms of an ablation can be aggregated from two files; the run
prints a summary at the end.

### Smoke test without a robot

Nothing here touches hardware or the robot network. Terminal A runs the
stand-in server (task venv or the client environment; it needs `pyzmq` and
`msgpack`):

```bash
python tasks/gr00t_n17_tactile/tests/fake_policy_server.py --port 5599 [--hands left|both] [--no-force] [--cascade]
```

Terminal B runs the client against it with `--fake-robot`, which replaces the
cameras, the arm controller and the hands by deterministic stand-ins that
record every command; it imports nothing from `teleop` or `unitree_sdk2py`,
so the same command also runs in the task venv through the console script:

```bash
PYTHONPATH=tasks/gr00t_n17_tactile:third_party/xr_teleoperate:$PYTHONPATH \
  python -m i4h_tasks.gr00t_n17_tactile.client.deploy --fake-robot --no-tactile \
  --policy-port 5599 --task "smoke" --steps 32 --num-episodes 1 --hold-s 0 --init-state-hold-s 0
```

Expected: the served `server_info` is logged, every observation is accepted,
one line per executed chunk is logged (the fake robot records all 32 arm
targets internally), and the run exits cleanly on completion and on Ctrl+C. Repeat with `--async-policy`, then with
`--async-policy --trex-refine-mode same-plan` against
`fake_policy_server.py --cascade`. `--check --fake-robot --check-dir /tmp/check`
writes the PNGs (or `.npy` files when `cv2` is absent). The real
`required_observation_keys` and the policy's strict input check are only
exercised against the real server:
`uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile --checkpoint <dir> --tactile-source none --port 5599`
with the same client command.

### Safety notes

- The hands move to neutral when the client connects, the arm controller
  drives both arms to the zero pose and the head to `--head-pitch-home` as soon
  as it is constructed (also with `--check`), and `go_home` drives the arms to
  the zero pose again; clear the workspace before starting.
- Keep a hand on the emergency stop; Ctrl+C homes the arms but the robot keeps
  moving until it gets there.
- Use `--no-arm` and `--no-hands` for plumbing runs, and `--check` before the
  first live run of a session.
- The arm controller blocks until the robot's `rt/lowstate` arrives; a client
  that logs "Waiting to subscribe dds..." forever is not on the robot network.

### Troubleshooting

- **"Waiting to subscribe dds..." never ends** although the body peers were
  discovered: the body's replies are not routed back to the workstation. The
  workstation reaches the body through the onboard computer, which must hold a
  host route for the workstation's robot-network address over the
  workstation-facing interface (the start scripts on the onboard computer
  install it; it is lost at reboot). Check `ip route get <workstation-robot-address>`
  there.
- **No tactile frames on port 7779 although the producer runs**: the relay
  publishes a fingertip only once it has that hand's joint telemetry, which the
  Sharpa DDS bridge supplies. Without the bridge the stream is silent; the
  server and the client report every fingertip as missing. Start the bridge
  (the hands open to the zero pose when it connects them).
- **A camera topic goes silent mid-run** (the gscam node is alive but idle,
  `ros2 topic hz` shows nothing, the client logs that a video key is missing):
  stop the camera nodes, run the onboard computer's camera setup script
  (`setup_sensing_cameras_boot.sh`, needs sudo: it resets the deserializer and
  reloads the camera modules), then restart the camera launcher. The
  asynchronous worker stops after 60 consecutive failed iterations and the
  episode ends with the error; the synchronous loop fails on the first one.

## Dependencies

Everything third-party is public and pinned by revision in `third_party/setup.sh`.

| Component | Source | Revision | License |
| --- | --- | --- | --- |
| GR00T N1.7 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), plus the patch series | `4b1dca9d` (the `tasks/gr00t_n17` pin) | Apache-2.0 |
| Base weights | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | | NVIDIA model license |
| SharpaWave deform encoder | [sharpa-robotics/sharpawave-deform-encoder](https://github.com/sharpa-robotics/sharpawave-deform-encoder), weights at [Sharpa-Robotics/sharpawave-deform-encoder](https://huggingface.co/Sharpa-Robotics/sharpawave-deform-encoder) | `7a30b37b` | Apache-2.0 |
| SharpaWave hand URDFs | [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) | `0d19cac6` | Apache-2.0 |
| SharpaWave SDK (on the robot) | [sharpa-robotics/sharpa-wave-sdk](https://github.com/sharpa-robotics/sharpa-wave-sdk), image [sharpadev/sharpawavesdk](https://hub.docker.com/r/sharpadev/sharpawavesdk) | | vendor |
| T-Rex force encoder | [ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex), weights inside [miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6) | `ba4047ce` | MIT |
| Unitree teleoperation (`teleop/` only) | [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate), plus `third_party/xr_teleoperate_h2_sharpa.patch` (H2 arm controller changes, SharpaWave hand DDS client, ROS 2 camera client, DDS network configuration and peer discovery, and the tactile wire format the robot-side bridge shares with teleop recording) | `ebe257f0` | Apache-2.0 |
| Unitree Python SDK | [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python), plus `third_party/unitree_sdk2_network_config.patch` (the `networkConfig` keyword); brings `cyclonedds==0.10.2` through its `setup.py` | `794fb2b3` | BSD-3-Clause |

An isaac-for-healthcare fork of the official `xr_teleoperate` repository will
carry the deployment patch as a branch; the pin then moves to that fork's
commit and the patch file goes away (`XR_TELEOPERATE_URL` in `setup.sh` is
that one variable).

The two Hugging Face files are pinned to a commit of their model repository
in the manifest (`deform_encoder.revision`, `force.encoder_revision`). The
T-Rex VQ-VAE tensors are read out of the published VLA checkpoint (`model.pt`,
8.5 GB) by key prefix; a file holding only its `tactile_vqvae.*` tensors loads
the same way. The published SharpaWave URDFs are byte-identical to the ones
the datasets were recorded against.

## Encoder weights

The stack targets the deform encoder checkpoint Sharpa publishes. The
checkpoints finetuned so far, including the ones behind the reported benchmark
numbers, were trained with an earlier, internal release of the encoder; the
published one is a separate training run of the same architecture, and for the
same map the two produce unrelated features. This does not affect serving: a
finetuned GR00T checkpoint carries its own copy of the frozen encoder, and the
file the manifest names only builds the module before the checkpoint's tensors
are loaded over it. Training a new checkpoint uses the public file throughout,
which is self-consistent; its results are not directly comparable to the
benchmark numbers.

Not in this repository: the datasets and finetuned checkpoints, which are
passed as paths; the robot-side bridge that republishes SDK frames over ZMQ
and relays the hand DDS topics (it runs on the robot's onboard computer next
to the SDK container); the fingertip calibration helper it runs; and the
teleoperation and dataset-conversion tooling, which live in their own public
repositories (see the workflow document).
