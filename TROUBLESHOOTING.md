# Troubleshooting

Symptom first. Each entry says how to confirm the cause and what to do about it.

Every run writes its own directory under `runs/<workflow>/<YYYYMMDD_HHMMSS>/`, and that is
the first place to look: `run.json` records the resolved mode and options, `backend-*.log`
holds the policy backend's own output, and the arena log holds the simulator's. The
launcher prints only a short failure line, so the actual error is usually in one of those.

## `backend did not become ready`

Two unrelated causes produce this same message.

### The checkpoint is still downloading

A policy backend downloads its checkpoint the first time it runs, inside the readiness
window. Checkpoints range from a few GB to 38 GB, so a first run on a cold cache can hit the
window while the download is still healthy and in progress. Confirm it by looking at
`runs/.../backend-*.log`: a download in flight is progress, not an error.

Raise the window until the cache is warm:

```bash
I4H_BACKEND_READY_TIMEOUT_S=600 ./run.sh <workflow> --policy --episodes 1
```

The same run typically takes a few minutes once the checkpoint is resident.

### Peer discovery selected the wrong network interface

The runner and its policy backend are separate processes that find each other over Zenoh. On
a host carrying several interfaces — Docker bridges, a VPN, a second NIC — discovery can
settle on one that cannot carry the session, and the handshake never completes even though
`backend-*.log` says the backend is ready. That contradiction is the signal: the backend
reports readiness and the runner still times out.

The most reliable fix is to serve the policy yourself and connect to it explicitly, which
skips discovery. Its preload also moves the checkpoint download out of the readiness window,
so this addresses both causes at once. Run each command in its own terminal:

```bash
./docker/policy-server.sh <workflow>
./run.sh <workflow> --policy --policy-endpoint 127.0.0.1:7448 --episodes 1
```

The server listens on `tcp/0.0.0.0:7448`; `--listen` and `--connect` move it, and the Docker
image exposes the same script as `i4h-policy`. To keep the single-command form instead, point
`ZENOH_CONFIG` at a Zenoh config file that pins scouting to loopback:

```bash
printf '{ mode: "peer", scouting: { multicast: { interface: "lo" } } }\n' > /tmp/zenoh-lo.json5
ZENOH_CONFIG=/tmp/zenoh-lo.json5 ./run.sh <workflow> --policy --episodes 1
```

For a policy and an arena on different machines, see [docker/README.md](docker/README.md).

## `remote task ... produced no action for 30.0s`

The backend accepted the session but did not answer an observation within the action
window. Look at the end of `runs/.../backend-*.log` to tell the two causes apart.

If the log stops at a native error, such as `ptxas fatal`, `LLVM ERROR` or a `F external/xla`
check, the backend process died and a longer window cannot help. The PI0 backend pins
jaxlib 0.5.3, whose XLA builds Triton GEMM kernels only for the GPUs it knows. On a newer one
(DGX Station GB300, compute capability 10.3; DGX Spark GB10, 12.1) the backend now passes
`--xla_gpu_enable_triton_gemm=false` itself, which sends those matrix multiplies to cuBLAS. An
`XLA_FLAGS` you export that already sets the option is left as it is.

Otherwise inference is slower than the window. The PI0 backend compiles its model with one
inference while it loads, before the simulator starts, so the first frame does not pay for
compilation. A backend without that warm-up, or a slow GPU, may still need a longer window:

```bash
I4H_BACKEND_ACTION_TIMEOUT_S=300 ./run.sh <workflow> --policy --episodes 1
```

## `You are trying to access a gated repo` in a backend log

Some policy stacks pull a gated model. Request access on the model page, then export a token
before running:

```bash
export HF_TOKEN=hf_...
```

A repository's metadata endpoint may answer anonymously even while file access returns 403,
so being able to view the model page is not evidence that a download will succeed.

## A run fails after the simulator boots, with a traceback

Two preconditions are only reported once the simulator is up, roughly 40 seconds in:

- **A missing recording.** `--replay` raises `FileNotFoundError: no recording at <path>`.
  Check the path before rerunning.
- **A missing checkpoint.** A mode that needs a policy you have to train and export first
  raises `ValueError: ... requires --checkpoint pointing to exported policy.pt`. The workflow
  catalog in [README.md](README.md) marks those modes; pass `--checkpoint <path>` to the
  exported artifact.

By contrast, an unknown workflow or an unknown mode is rejected immediately, with the valid
options listed, so a fast rejection means the command was wrong and a slow one means the
environment was.

## `Failed to create device!` from Slang fluoroscopy

Patient-CT fluoroscopy on the CUDA device (`--fluoro-device cuda`) and the standalone
`xray-simulator` CLI, whose default device is CUDA, stop with slangpy's
`RuntimeError: Failed to create device!` when Slang cannot load the CUDA Toolkit's NVRTC.
The launcher checks this before Isaac Sim starts and reports
`--fluoro-device cuda: Slang could not create a CUDA device`, together with the first
`libnvrtc.so` it found and that library's version.

Slang compiles its shaders with NVRTC and loads the toolkit's unversioned `libnvrtc.so`. The
pip wheels ship only the versioned `libnvrtc.so.N`, so they are not enough. Fix it in one of
these ways:

- **Render on Vulkan.** `--fluoro-device vulkan`, the default, does not use NVRTC.
- **Install a CUDA Toolkit.** CUDA 12.x or 13.x works if it is no newer than the CUDA version
  `nvidia-smi` reports: shaders that a newer NVRTC emits are rejected by an older driver. CUDA
  12.1 with driver 580 and CUDA 13.2 with driver 610 are known to work. Slang looks under
  `CUDA_PATH` and `/usr/local/cuda*` and on the loader path; set `CUDA_PATH` to choose a
  toolkit when several are installed.

To see what is installed, run `ls /usr/local/cuda*/lib64/libnvrtc.so*`. A toolkit directory
without `libnvrtc.so` is a partial install, such as one that only provides driver headers.

## The same `--seed` produces a different outcome

Episode outcomes can differ between runs at a fixed seed, and the runner logs
`Seed not set for the environment` when that applies. Treat one run as an existence proof
rather than a success rate: for anything you intend to report or compare, run several trials
and quote the ratio.

## Processes or GPU memory left behind

```bash
./stop.sh all
```

This stops every i4h run in the checkout, not just the one that failed, so avoid it while a
second run or a live authoring session is active on another GPU. It matches on i4h module
paths, so an unrelated Isaac Sim you started yourself is left alone.

Each `./run.sh` records the backends it starts in a file of its own under `.run/pids.d/` and
stops them when it exits, so concurrent runs in one checkout, or the test suite run beside
one, neither stop nor orphan each other's policy servers. A launcher killed outright, for
example with `kill -9`, cannot clean up; `./stop.sh all` stops every backend listed under
`.run/pids.d/` and then sweeps for any remaining i4h process.

## Docker: the first command takes much longer than expected

The container creates its environments into the `i4h-state` volume on first use, and how much
it creates depends on the command. A discovery command such as `./run.sh list` needs seconds;
`i4h-policy <task>` installs one policy stack and downloads its checkpoint, on the order of
ten minutes and 30 GB; a full `./run.sh <workflow>` installs every component, which needs
substantially more time and disk.

To install only what a given workflow needs, name the projects. Add one more `-e` to the
`i4h-docker` function in [docker/README.md](docker/README.md):

```bash
-e I4H_SETUP_PROJECTS="common engine workflows arena tasks/gr00t_n15"
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
