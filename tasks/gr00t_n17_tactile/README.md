# gr00t_n17_tactile

GR00T N1.7 with fingertip tactile conditioning for the Unitree H2 with
SharpaWave hands, on the real robot. The narrative, the data format, the
training and deployment flow and the roadmap are in
[docs/workflows/h2_tactile.md](../../docs/workflows/h2_tactile.md); this file
is the project map.

This is a structure-first draft: the layout, the manifest, the contracts and
the argument surfaces are final, and the parts that need the GR00T patch
series raise `NotImplementedError` until it lands (`patches/README.md`).

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

| Part | State in this draft |
| --- | --- |
| Manifest, modality config, sensor contracts, wire format, subscriber, live source, `server_info`, trainer argv | complete |
| Deform encoder, fingertip kinematics, force encoder wrappers | signatures with the loading recipe; bodies land with the patch series |
| Policy loading, server binding, finetune launch | raise until the patch series lands |

## Commands

```bash
# What a client learns on connect, resolved from the manifest. Light venv.
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile --print-server-info

# The exact launcher invocation for a dataset, without running it.
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile-train \
  --dataset /data/h2_sharpa_pads_lerobot --dry-run

# Tests; also part of the light suite in scripts/run_tests.py.
uv run --project tasks/gr00t_n17_tactile --extra dev pytest
```

The project resolves without a model stack in this draft: the `gr00t` and
`timm` dependencies are switched on in `pyproject.toml` with the code
follow-up, together with the patched checkout under `third_party/`.

Serving a checkpoint, once the series has landed:

```bash
uv run --project tasks/gr00t_n17_tactile i4h-tasks-gr00t-n17-tactile \
  --checkpoint /ckpt/h2_sharpa_pads --tactile-host <robot-onboard-computer> \
  --tactile-refresh per-step [--cascade] [--disable-tactile]
```

## Dependencies

Everything third-party is public and pinned by revision in the follow-up.

| Component | Source | License |
| --- | --- | --- |
| GR00T N1.7 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) at the revision `tasks/gr00t_n17` pins, plus the patch series | Apache-2.0 |
| Base weights | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | NVIDIA model license |
| SharpaWave deform encoder | [sharpa-robotics/sharpawave-deform-encoder](https://github.com/sharpa-robotics/sharpawave-deform-encoder), weights at [Sharpa-Robotics/sharpawave-deform-encoder](https://huggingface.co/Sharpa-Robotics/sharpawave-deform-encoder) | Apache-2.0 |
| SharpaWave hand URDFs | [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml) | Apache-2.0 |
| SharpaWave SDK (on the robot) | [sharpa-robotics/sharpa-wave-sdk](https://github.com/sharpa-robotics/sharpa-wave-sdk), image [sharpadev/sharpawavesdk](https://hub.docker.com/r/sharpadev/sharpawavesdk) | vendor |
| T-Rex force encoder | [ZhuoyangLiu2005/T-Rex](https://github.com/ZhuoyangLiu2005/T-Rex), weights at [miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6) | MIT |

Not in this repository: the datasets and finetuned checkpoints, which are
passed as paths; the robot-side bridge that republishes SDK frames over ZMQ
(a small script that runs next to the SDK container); and the teleoperation
and dataset-conversion tooling, which live in their own public repositories
(see the workflow document).
