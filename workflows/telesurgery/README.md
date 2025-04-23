# Telesurgery

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Development

Use `./run vscode` to start a new VS Code window for development.

Note that the first launch could take a couple of minutes while IsaacSim is compiling shaders. These
shaders are cached and the second launch is much quicker.
