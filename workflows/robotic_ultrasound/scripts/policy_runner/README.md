# Policy runner for both the simulation and physical world

## Run PI0 policy with DDS communication

### Prepare Model Weights

[FIXME: change to `asset-catalog`] Download the v0.3 model weights from [GDrive](https://drive.google.com/drive/folders/1sL4GAETSMbxxcefsTsOkX7wXkTsbDqhW?usp=sharing)

### Prepare USD Assets

Please check the [I4H asset catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog) for assets downloading, put the USD assets as "./assets".

### Install Dependencies

Follow the [Environment Setup](../README.md#environment-setup) instructions to setup the environment and dependencies.


### Setup Python Path

Please move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```

### Run Policy

Please move back to the current folder and run the following command:

```sh
python run_policy.py  \
    --rti_license_file <path to>/rti_license.dat \
    --ckpt_path <path to policy model> \
    --repo_id i4h/sim_liver_scan \
    --domain_id <domain id> \
    --height <input image height> \
    --width <input image width>
```
