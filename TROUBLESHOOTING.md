# Troubleshooting

Symptom first. Each entry says how to confirm the cause and what to do about it.

Keep each run's output under `runs/<workflow>/<YYYYMMDD_HHMMSS>/`: the policy runner's
`--output_base_dir` holds its per-episode results, and a policy server's own terminal or log
holds the server's output. The policy runner's output ends with the simulator's error.

## The policy runner fails at its first request to a policy server

Start the server first and wait until it is ready before starting the runner: GR00T's server
prints `Server is ready`, and openpi's accepts connections on its port once its model has
loaded and compiled. A server downloads its
checkpoint the first time it runs, before it listens. Checkpoints range from a few GB to
38 GB, so on a cold cache this takes minutes; the download in the server's output is progress,
not an error.

Each client waits a bounded time for a reply:

| Client | Wait | Option |
| --- | --- | --- |
| `Gr00tN15RemotePolicy` | 120 s per reply | `--timeout_s` |
| `Gr00tN17RemotePolicy` | GR00T's `PolicyClient`, 15 s per reply | none; start the runner once the server is ready |
| `LiverScanPi0Policy` | 300 s for the server to answer | `--ping_timeout` |

A server on another machine is reached with `--remote_host`.

## A policy server stops during inference

Look at the end of the server's output. If it stops at a native error, such as
`ptxas fatal`, `LLVM ERROR` or a `F external/xla` check, the server process died and waiting
longer cannot help. The PI0 server pins jaxlib 0.5.3, whose XLA builds Triton GEMM kernels
only for the GPUs it knows. On a newer one (DGX Station GB300, compute capability 10.3; DGX
Spark GB10, 12.1) the server passes `--xla_gpu_enable_triton_gemm=false` itself, which sends
those matrix multiplies to cuBLAS. An `XLA_FLAGS` you export that already sets the option is
left as it is. The PI0 server also compiles its model with one inference while it loads, so
the first request does not pay for compilation.

## `You are trying to access a gated repo` in a policy server's output

Some policy stacks pull a gated model. Request access on the model page, then export a token
before running:

```bash
export HF_TOKEN=hf_...
```

A repository's metadata endpoint may answer anonymously even while file access returns 403,
so being able to view the model page is not evidence that a download will succeed.

## A run fails after the simulator boots, with a traceback

Arena imports the environment only after Isaac Sim has started, so several mistakes are
reported roughly 40 seconds in rather than at once:

- **An unknown environment.** The class after `i4h_arena.environments:` must be the scene
  name in CamelCase, and the scene name follows it; `ls arena/i4h_arena/scenes/manifest`
  lists the scenes.
- **Options on the wrong side of the scene name.** Launcher and policy options, such as
  `--presets newton`, `--enable_cameras` and `--viz kit`, go before it; the environment's
  own, such as `--episode_steps` or `--patient_twin`, after it.
- **A missing recording.** `replay_demos.py --dataset_file` names a file that is not there.

A policy server that needs a checkpoint you have to train first stops at once instead: it
reports that its task names no published checkpoint and asks for `--model-path` or
`--checkpoint`.

## `Failed to create device!` from Slang fluoroscopy

Patient-CT fluoroscopy on the CUDA device, which endoluminal RL uses by default, and the
standalone `xray-simulator` CLI, whose default device is CUDA, stop with slangpy's
`RuntimeError: Failed to create device!` when Slang cannot load the CUDA Toolkit's NVRTC. The
endoluminal environments render on Vulkan and do not need NVRTC.

Slang compiles its shaders with NVRTC and loads the toolkit's unversioned `libnvrtc.so`. The
pip wheels ship only the versioned `libnvrtc.so.N`, so they are not enough. Fix it in one of
these ways:

- **Render on Vulkan.** Vulkan does not use NVRTC; it reads each frame back through the
  host, at roughly twice the time per frame.
- **Install a CUDA Toolkit.** CUDA 12.x or 13.x works if it is no newer than the CUDA version
  `nvidia-smi` reports: shaders that a newer NVRTC emits are rejected by an older driver. CUDA
  12.1 with driver 580 and CUDA 13.2 with driver 610 are known to work. Slang looks under
  `CUDA_PATH` and `/usr/local/cuda*` and on the loader path; set `CUDA_PATH` to choose a
  toolkit when several are installed.

To see what is installed, run `ls /usr/local/cuda*/lib64/libnvrtc.so*`. A toolkit directory
without `libnvrtc.so` is a partial install, such as one that only provides driver headers.

## The e2e annotate stage fails episodes the simulator passed

The `annotate` stage of `scripts/e2e/run.sh` ends with `0/N succeeded` and `ValueError: no
successful episodes`, although the `record` stage reported every episode succeeded. The VLM
sees only a few sampled frames, so first look at what it was shown. Dump the frames for one
demo, from every camera, with the same sampling the annotator uses:

```bash
uv run --project tools/annotator python - runs/<workflow>/<run>/data/expanded.hdf5 <<'EOF'
import sys, h5py, numpy as np
from PIL import Image
obs = h5py.File(sys.argv[1], "r")["data/demo_0/obs"]
for cam in obs:
    v = obs[cam]
    if v.dtype == np.uint8 and v.ndim == 4:
        idx = np.linspace(0, len(v) - 1, 6).astype(int)
        Image.fromarray(np.concatenate([v[i] for i in idx], axis=1)).save(f"/tmp/annot_{cam}.png")
EOF
```

The driver grades against the `annotate` block of the scene manifest
(`arena/i4h_arena/scenes/manifest/<scene>.yaml`), and against the policy task's prompt when
there is none. If the subject is a few pixels in the graded camera, name a better view in its
`cameras`; any camera the scene records will do. If the prompt names no state a camera can
see, as `Perform a liver ultrasound.` does not, write one as the `criterion`. If success is a
motion rather than a resting place, as for a probe sweep, set `judge: trajectory` so the model
is not asked to find it in the last frame. Grade the same file by hand with `--camera`,
`--judge` and `--task` until the verdicts match what the frames show, then put those values in
the manifest.

## PI0 fine-tuning stops with `wandb.errors.errors.UsageError: No API key configured`

openpi's training config enables Weights & Biases by default, and logs in before the first
step. The `openpi_pi0` trainer turns it off unless you ask for it with `--wandb` or with
`train.wandb: true` in the task manifest; an exported `WANDB_API_KEY` or a `wandb login` does
not turn it on. Its `--dry-run` prints the resolved `wandb` value and what decided it. Seeing
this error means logging was requested without a working login: run `wandb login`, set
`WANDB_MODE=offline` to log locally without an account, or drop `--wandb`. A checkout that
predates the opt-in needs `WANDB_MODE=disabled` to skip the login. Resuming, with `--wandb`,
a run that was trained without it keeps logging off and says so, because that run has no
W&B id to resume; pass `--overwrite` or a new `--output-dir` to start a logged run.

## The same `--seed` produces a different outcome

Episode outcomes can differ between runs at a fixed seed: PhysX contact and GPU scheduling
are not bit-reproducible. Treat one run as an existence proof
rather than a success rate: for anything you intend to report or compare, run several trials
and quote the ratio.

## Processes or GPU memory left behind

Stop a run with Ctrl+C. Stop a background run or a policy server by the process ID you
started it with; matching on a tool or module name also stops other people's runs on a shared
machine. Stop a live scene session with `./local-agent/bridge.sh stop <scene>`. To find what is
holding a GPU, run `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

## Docker: the first command takes much longer than expected

The container creates its environments into the `i4h-state` volume on first use, and how much
it creates depends on the command. `i4h-policy <stack>/<task>` installs one policy stack and
downloads its checkpoint, on the order of ten minutes and 30 GB; a simulator run installs the
arena environment and every component, which needs substantially more time and disk.

To install only what a given workflow needs, name the projects. Add one more `-e` to the
`i4h-docker` function in [docker/README.md](docker/README.md):

```bash
-e I4H_SETUP_PROJECTS="common arena tasks/gr00t_n15"
```

An unknown name is rejected before anything is installed, so a typo costs nothing. The state
volume is persistent, so whatever does get installed is paid for once per machine rather than
once per container.

## Docker: `cannot attach stdin to a TTY-enabled container`

The `i4h-docker` helper in [docker/README.md](docker/README.md) passes `-it`, which requires a
terminal. From a script, cron, or CI, drop `-t`:

```bash
docker run --rm -i ...
```
