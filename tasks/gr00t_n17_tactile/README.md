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
│   └── tactile/
│       ├── interface.py                 TactileSource / TactileEncoder / FingertipKinematics
│       ├── sharpa.py                    SharpaWave implementations (public sources)
│       └── sensor/{wire,subscriber}.py  ZMQ stream from the robot-side bridge
├── patches/README.md                    what the GR00T patch series changes
└── tests/                               light-venv tests (no gr00t, no torch)
```

| Part | State |
| --- | --- |
| Manifest, modality config, sensor contracts, wire format, subscriber, live source, `server_info`, trainer argv | complete |
| Deform encoder, fingertip kinematics, force encoder wrappers | complete; thin wrappers over the modules of the patched checkout |
| Policy loading, server binding, finetune launch | complete; need the checkouts that `setup.sh` creates |
| Deployment client (cameras, robot state, action execution on the H2) | not in this repository yet, see the workflow document |

## Setup

```bash
# The patched Isaac-GR00T checkout and the public sources of the frozen
# encoders and the hand URDFs, all pinned by revision (see Dependencies).
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
(a small script that runs next to the SDK container); the deployment client;
and the teleoperation and dataset-conversion tooling, which live in their own
public repositories (see the workflow document).
