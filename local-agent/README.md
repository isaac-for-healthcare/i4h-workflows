# Local i4h Agent

Run this repository's root-level i4h workflow skills with a local SGLang model server and OpenCode.

## Requirements

- NVIDIA GPU with Docker GPU support.
- One 96 GB Blackwell GPU, or one DGX Spark.
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

This starts the accuracy-first [Nemotron 3.5 Lightning BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) model and opens an interactive agent. DGX Spark settings and GPU selection are automatic.

To run a single prompt:

```bash
./local-agent/run.sh agent "run i4h-workflow-setup"
```

[Nemotron 3.5 Lightning NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) is an optional lower-memory mode for DGX Spark:

```bash
I4H_AGENT_PROFILE=nemotron-3.5-lightning-nvfp4 ./local-agent/run.sh start
```

Run `./local-agent/run.sh` without arguments to see server-management commands.

Agent shell commands share one tmux-backed session, so working-directory changes and exported variables remain available to later commands in the same agent workflow.

## Remote (NVIDIA-hosted) models

Use `nvidia-hosted` to run the coding and vision models on NVIDIA's hosted inference APIs instead of local SGLang. Model IDs are kept in the launch command because they differ between the public and internal endpoints.

Internal NVIDIA Inference keys from <https://inference.nvidia.com/> use:

```bash
export I4H_AGENT_API_KEY=sk-...
I4H_AGENT_PROFILE=nvidia-hosted \
I4H_AGENT_BASE_URL=https://inference-api.nvidia.com \
I4H_AGENT_MODEL=switchyard/openai/gpt-5.6-sol \
I4H_AGENT_VL_BASE_URL=https://inference-api.nvidia.com \
I4H_AGENT_VL_MODEL=nvidia/nvidia/nemotron-nano-12b-v2-vl \
./local-agent/run.sh agent
```

Public NVIDIA Build keys from <https://build.nvidia.com/> use [Nemotron 3 Ultra](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b):

```bash
export I4H_AGENT_API_KEY=nvapi-...
I4H_AGENT_PROFILE=nvidia-hosted \
I4H_AGENT_BASE_URL=https://integrate.api.nvidia.com \
I4H_AGENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b \
I4H_AGENT_VL_BASE_URL=https://integrate.api.nvidia.com \
I4H_AGENT_VL_MODEL=nvidia/nemotron-nano-12b-v2-vl \
./local-agent/run.sh agent
```

Keep API keys out of git. The vision endpoint reuses `I4H_AGENT_API_KEY` unless `I4H_AGENT_VL_API_KEY` is set separately.
