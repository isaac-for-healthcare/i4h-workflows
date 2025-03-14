# Scripts for all robotic surgery simulations

## System Requirements

- Ubuntu 22.04
- NVIDIA GPU with compute capability 8.6 and 32GB of memory
    - GPUs without RT Cores (A100, H100) are not supported.
- NVIDIA Driver Version >= 555
- 50GB of disk space

## Environment Setup

The robotic surgery workflow is built on the following dependencies:
- [IsaacSim 4.1.0.0](https://docs.isaacsim.omniverse.nvidia.com/4.1.0/index.html)
- [IsaacLab 1.2.0](https://isaac-sim.github.io/IsaacLab/v1.2.0/source/setup/installation/index.html)

We recommend using `conda` to create a new environment and install all the dependencies:

Install other dependencies by running:
```bash
cd <path-to-i4h-workflows>
python tools/install_deps.py --workflow robotic_surgery
```

### Download the I4H assets

```sh
i4h-asset-retrieve
```

## Run the scripts

Navigate to these sub-directories and run the scripts.

- [Simulation](./simulation)
