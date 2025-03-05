import os
import tempfile
import zipfile
import logging
from typing import Literal


import omni.client
import shutil


SHA256_HASH = {
    "0.1": "9cecada4748a203e3880a391924ce5f1343494530753368dd9be65aeb0535cf9"
}

I4H_ASSET_ROOT = {
    "nucleus": "https://isaac-dev.ov.nvidia.com/omni/web3/omniverse://isaac-dev.ov.nvidia.com",
    "staging": "",  # FIXME: Add staging asset root
    "production": ""  # FIXME: Add production asset root
}

DEFAULT_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache", "i4h-assets")


def get_i4h_asset_path(version: Literal["0.1"] = "0.1") -> str:
    """
    Get the path to the i4h asset for the given version.
    """
    asset_root = I4H_ASSET_ROOT.get(os.environ.get("ISAAC_ENV", "nucleus"))  # FIXME: Add production asset root
    sha256_hash = SHA256_HASH.get(version, None)
    if sha256_hash is None:
        raise ValueError(f"Invalid version: {version}")
    remote_path = f"{asset_root}/Library/IsaacHealthcare/{version}/i4h-assets-v{version}-{sha256_hash}.zip"
    if not omni.client.stat(remote_path)[0] == omni.client.Result.OK:
        raise ValueError(f"Asset not found: {remote_path}")
    return remote_path


def retrieve_asset(version: Literal["0.1"] = "0.1", download_dir: str | None = None, force_download: bool = False) -> str:
    """
    Download the asset from the remote path to the download directory.
    """
    if download_dir is None:
        download_dir = DEFAULT_DOWNLOAD_DIR

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    remote_path = get_i4h_asset_path(version)
    sha256_hash = SHA256_HASH.get(version)  # Already checked by get_i4h_asset_path

    # If the asset hash is a folder in download_dir, skip the download
    local_path = os.path.join(download_dir, sha256_hash)
    if os.path.exists(local_path) and not force_download:
        return local_path

    # Force download
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    os.makedirs(local_path)

    result, _, file_content = omni.client.read_file(remote_path)

    if result != omni.client.Result.OK:
        raise ValueError(f"Failed to download asset: {remote_path}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "i4h-assets-v{version}.zip"), "wb") as f:
                f.write(file_content)
            # TODO: Check sha256 hash
            with zipfile.ZipFile(os.path.join(temp_dir, "i4h-assets-v{version}.zip"), "r") as zip_ref:
                zip_ref.extractall(local_path)
            return local_path
    except Exception as e:
        raise ValueError(f"Failed to extract asset: {remote_path}") from e
