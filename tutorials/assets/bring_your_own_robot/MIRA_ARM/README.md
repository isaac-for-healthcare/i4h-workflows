# MIRA ARM TEle

## Environment Setup

The tutorial is built on the following dependencies:
- [IsaacSim 4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
- [IsaacLab 2.0.2](https://isaac-sim.github.io/IsaacLab/v2.0.2/index.html)

If you already installed dependencies of a workflow (such as `robotic_ultrasound`, or `robotic_surgery`), you can use them directly. Otherwise, you can install dependencies according to the following subsection.

### (Optional) Install Dependencies

#### Create a new conda environment

Conda is suggested for virtual environment setup, install `Miniconda` from [Miniconda website](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) and create a new virtual environment with Python 3.10.

```sh
# Create a new conda environment
conda create -n robotic_mira python=3.10 -y
# Activate the environment
conda activate robotic_mira
```

#### Install IsaacSim and IsaacLab

```sh
cd <path-to-i4h-workflows>
bash tools/install_isaac.sh
```