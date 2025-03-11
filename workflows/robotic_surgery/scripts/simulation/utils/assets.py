import os
from dataclasses import dataclass

from i4h_asset_helper import get_i4h_local_asset_path


@dataclass
class Enums:
    """Enums for the assets in the robotic surgery workflow."""
    dVRK_PSM = "Robots/dVRK/PSM/psm_col.usd"


class Assets(Enums):
    """Assets for the robotic surgery workflow customized for the user download directory."""
    def __init__(self, download_dir: str | None = None):
        """
        Initialize the assets

        Args:
            download_dir: The directory to download the assets to
        """
        if download_dir is None:
            download_dir = get_i4h_local_asset_path()
        # Update all the paths
        for attr in dir(self):
            if not attr.startswith("_"):
                setattr(self, attr, os.path.join(download_dir, getattr(self, attr)))


# singleton object for the assets
robotic_surgery_assets = Assets()
