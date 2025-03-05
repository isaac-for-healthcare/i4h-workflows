# Policy runner for both the simulation and physical world

## Run PI0 policy with DDS communication
1. [FIXME: change to `asset-catalog`] Download the v0.3 model weights from [GDrive](https://drive.google.com/drive/folders/1sL4GAETSMbxxcefsTsOkX7wXkTsbDqhW?usp=sharing)

2. Please check the [I4H asset catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog) for assets downloading, put the USD assets as "./assets".

3. To install `openpi` in python 3.10 without `uv` environment, and support `IsaacSim 4.2`, we need below minor steps:
- `git clone git@github.com:Physical-Intelligence/openpi.git`
- Changes for `openpi/src/openpi/shared/download.py` (just temp workaround, will not need it after upgrading to IsaacSim 4.5):
  ```py
  -import boto3.s3.transfer as s3_transfer
  +# import boto3.s3.transfer as s3_transfer

  -import s3transfer.futures as s3_transfer_futures
  +# import s3transfer.futures as s3_transfer_futures

  -from types_boto3_s3.service_resource import ObjectSummary
  +# from types_boto3_s3.service_resource import ObjectSummary

  -) -> s3_transfer.TransferManager:

  -date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
  +date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
  ```
- Change the python requirement in `pyproject.toml` to `>=3.10`.
- Install `lerobot`, `openpi-client` and `openpi`:
  First of all, create a python visual environment, referring to the [python setup](../README.md), then execute:
  ```sh
  conda activate robotic_ultrasound
  pip install toml
  pip install git+https://github.com/huggingface/lerobot@6674e368249472c91382eb54bb8501c94c7f0c56
  pip install -e packages/openpi-client/
  pip install -e .
  ```

4 Install `RTI` for communication, refer to [DDS Setup](../README.md).

5. Now that move to the [scripts](../) folder and specify python path:
```sh
export PYTHONPATH=`pwd`
```

5. Return to this folder and run the following command:
```sh
python run_policy.py  \
    --rti_license_file <path to>/rti_license.dat \
    --ckpt_path <path to>/pi0_aortic_scan_v0.3/19000 \
    --repo_id hf/chiron_aortic \
    --domain_id <domain id> \
    --height <input image height> \
    --width <input image width>
```
