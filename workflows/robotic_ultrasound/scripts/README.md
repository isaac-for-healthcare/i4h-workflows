# Scripts for all the simulation and physical world logic

## Setup
1. [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install)
is suggested for virtual environment setup, install `Miniconda`, then execute:
```sh
# Create a new conda environment
conda create -n robotic_ultrasound python=3.10 -y
# Activate the environment
conda activate robotic_ultrasound
# Might be needed if your system GLIBC doesn't match what's used by Holoscan
conda install -c conda-forge gcc=13.3.0
```
2. RTI DDS is the common communication package for all the scripts,
please refer to [DDS website](https://www.rti.com/products) for registration.
Install `RTI DDS`:
```sh
pip install rti.connext
```
3. Make sure `PYTHONPATH` and `RTI_LICENSE_FILE` is set
```sh
export PYTHONPATH=`/path-to-i4h-workflows/workflows/robotic_ultrasound/scripts`
export RTI_LICENSE_FILE=<path-to-rti-license-file>
```
