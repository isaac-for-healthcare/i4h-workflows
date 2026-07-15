# Local i4h Agent

Run this repo's i4h skills with a local SGLang model server and OpenCode.

## Requirements

- NVIDIA GPU with Docker GPU support.
- `qwen3-coder-30b-fp8` profile: one 48 GB GPU.
- Default `qwen3-coder-next` profile: one 96 GB GPU for SGLang, preferring GPU 1 with the FP8 checkpoint when available.
- `qwen3-coder-next-bf16` profile: two 96 GB GPUs.
- GPU 0 should stay free if you plan to run Isaac Sim at the same time.

Install the host tools once:

```bash
sudo apt update
sudo apt install npm tmux xclip -y
sudo npm install -g opencode-ai
```

## Usage

```bash
./local-agent/run.sh start
./local-agent/run.sh agent
```

`agent` with no prompt opens an interactive OpenCode session. To run one prompt non-interactively:

```bash
./local-agent/run.sh agent "run i4h-workflow-setup"
```

The default profile is `qwen3-coder-next`, which prefers GPU 1 so GPU 0 remains available for Isaac Sim, and falls back to GPU 0 on single-GPU hosts. Override settings with environment variables from `local-agent/config.env`, for example:

```bash
I4H_AGENT_PROFILE=qwen3-coder-30b-fp8 ./local-agent/run.sh start
I4H_AGENT_PROFILE=qwen3-coder-next-bf16 ./local-agent/run.sh start
```

To use an already-running OpenAI-compatible model server instead of starting the local Docker model, point the agent at that endpoint and run `agent` directly:

```bash
I4H_AGENT_BASE_URL=http://10.111.19.0:8001 \
I4H_AGENT_VL_BASE_URL=http://10.111.19.0:8000 \
./local-agent/run.sh agent
```

## Remote (NVIDIA-hosted) models

Use `nvidia-hosted` to run the coding model on NVIDIA's hosted inference APIs instead of local SGLang. The portal URL where you get a key is not always the same as the OpenAI-compatible API base URL that `run.sh` calls.  Default model is `azure/openai/gpt-5.5`.

Public NVIDIA Build keys from <https://build.nvidia.com/> use:

```bash
export I4H_AGENT_API_KEY=nvapi-...
I4H_AGENT_PROFILE=nvidia-hosted \
I4H_AGENT_BASE_URL=https://integrate.api.nvidia.com \
./local-agent/run.sh agent
```

Internal NVIDIA Inference keys from <https://inference.nvidia.com/> use:

```bash
export I4H_AGENT_API_KEY=sk-...
I4H_AGENT_PROFILE=nvidia-hosted \
I4H_AGENT_BASE_URL=https://inference-api.nvidia.com \
./local-agent/run.sh agent
```

Optional model override:

```bash
I4H_AGENT_PROFILE=nvidia-hosted \
I4H_AGENT_BASE_URL=https://inference-api.nvidia.com \
I4H_AGENT_MODEL=nvidia/nvidia/nemotron-3-ultra \
./local-agent/run.sh agent
```

Keep API keys out of git. VL defaults: `I4H_AGENT_VL_BASE_URL=https://inference-api.nvidia.com`, `I4H_AGENT_VL_MODEL=nvidia/nvidia/nemotron-nano-12b-v2-vl`, `I4H_AGENT_VL_API_KEY=${I4H_AGENT_API_KEY}`.
