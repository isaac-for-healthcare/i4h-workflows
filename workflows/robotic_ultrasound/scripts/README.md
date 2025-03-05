# Scripts for all the simulation and physical world logic

## Setup
1. [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)
is suggested for virtual environment setup, install `Miniconda`, then execute:
```sh
# Create a new conda environment
conda create -n robotic_ultrasound python=3.10 -y
# Activate the environment
conda activate robotic_ultrasound
```
2. RTI DDS is the common communication package for all the scripts,
please refer to [DDS website](https://www.rti.com/products) for registration.
Install `RTI DDS`:
```sh
pip install rti.connext
```
