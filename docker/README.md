# Docker

The default `i4h-workflows` image contains the source and every pinned third-party checkout. The entrypoint creates persistent virtual environments automatically, so a user who pulls the image does not run `setup.sh`.

An optional full image also includes every environment created by `setup.sh`. It is much larger, but its first command starts without installing dependencies.

The examples use the local tag `i4h-workflows:local`.

## Build

```bash
docker build \
  --cache-to type=inline \
  -f docker/Dockerfile \
  -t i4h-workflows:local .
```

Build the optional full image when download size matters less than immediate startup:

```bash
docker build \
  --target full \
  --cache-to type=inline \
  -f docker/Dockerfile \
  -t i4h-workflows:full .
```

The Dockerfile uses NVIDIA's Ubuntu 24.04 base for both amd64 and arm64. Isaac Sim, PyTorch, and their CUDA libraries come from the component locks instead of being duplicated by a larger base image.

## Run

Define this function once in the terminal:

```bash
i4h-docker() {
  local image="${I4H_DOCKER_IMAGE:-i4h-workflows:local}"
  local uv_cache_args=()
  local xauthority="${XAUTHORITY:-$HOME/.Xauthority}"
  if [ ! -f "$xauthority" ]; then
    echo "i4h-docker: Xauthority file not found: $xauthority" >&2
    return 1
  fi
  if [ -n "${I4H_HOST_UV_CACHE:-}" ]; then
    uv_cache_args=(
      -e UV_LINK_MODE=copy
      -v "$I4H_HOST_UV_CACHE:/opt/i4h-state/uv-cache"
    )
  fi
  docker run --rm -it \
    --runtime=nvidia \
    --network host \
    --ipc host \
    -e DISPLAY \
    -e HF_TOKEN \
    -e XAUTHORITY=/root/.Xauthority \
    -v "$xauthority:/root/.Xauthority:ro" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v i4h-state:/opt/i4h-state \
    "${uv_cache_args[@]}" \
    "$image" \
    "$@"
}
```

Commands use the default image. To use the full image, change only the image selection:

```bash
I4H_DOCKER_IMAGE=i4h-workflows:full i4h-docker ./run.sh scissor_pick_and_place --policy --episodes 1
```

## Commands

| Container command | Use it for |
| --- | --- |
| `./run.sh ...` | Full workflow with arena, Isaac Sim, and any local policy backend selected by the workflow |
| `i4h-policy <task-id>` | Policy inference only, including on Jetson AGX Thor |
| `i4h-annotator ...` | Visual validation against an OpenAI-compatible VLM service |
| `bash` | Interactive work inside the image |

Use the same named `i4h-state` volume in every command. It preserves model downloads, simulator caches, recordings, and the default image's uv cache and virtual environments.

With the default image, the first command downloads only the environment it needs and later containers reuse it automatically. The full image skips dependency setup, but policy models still download on first use and persist in `i4h-state`.

`HF_TOKEN` is forwarded when it is set, which allows gated model downloads to use the same helper.

To reuse an existing host uv cache during the default image's first setup, prefix a command with `I4H_HOST_UV_CACHE="$HOME/.cache/uv"`. The writable bind may create root-owned cache entries on the host, so leave it unset for the isolated Docker-managed cache.

## Full workflow

On x86 or DGX Spark, `run.sh` behaves like it does in a host checkout and starts the policy locally when the selected mode needs one.

```bash
i4h-docker ./run.sh scissor_pick_and_place --policy --episodes 1
```

## Policy on Thor

Run only the policy on Thor. The task ID selects the policy stack and the Zenoh namespace, and the server listens on port `7448` by default.

```bash
i4h-docker i4h-policy gr00t_n15/scissor_pick_and_place
```

Run the arena on an x86 or DGX Spark host and point the same workflow at Thor:

```bash
i4h-docker ./run.sh scissor_pick_and_place --policy --policy-endpoint 192.168.1.50:7448 --episodes 1
```

## Visual validation

Record a workflow to a known path:

```bash
i4h-docker ./run.sh scissor_pick_and_place --policy --episodes 1 --record /workspace/runs/demos.hdf5
```

Start vLLM as a sibling container:

```bash
docker run --rm -d \
  --name i4h-workflows-vllm \
  --runtime=nvidia \
  --network host \
  --ipc host \
  -e HF_HOME=/opt/i4h-state/huggingface \
  -v i4h-state:/opt/i4h-state \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --dtype auto --gpu-memory-utilization 0.4 --max-model-len 32768 --trust-remote-code
```

Grade a recording with the i4h image:

```bash
i4h-docker i4h-annotator --task "the scissors reach and remain in the target tray" offline /workspace/runs/demos.hdf5 --write
```

## Shell

```bash
i4h-docker bash
```

Running the image sets `ACCEPT_EULA=Y` and acknowledges the [NVIDIA Isaac Sim EULA](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/).
