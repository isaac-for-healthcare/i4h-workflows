import os
from dataclasses import dataclass

from i4h_asset_helper import get_i4h_local_asset_path


@dataclass
class Enums:
    """Enums for the assets in the robotic ultrasound workflow."""

    basic = "Test/basic.usda"
    panda = "Robots/Franka/Collected_panda_assembly/panda_assembly.usda"
    phantom = "Props/ABDPhantom/phantom.usda"
    table_with_cover = "Props/VentionTableWithBlackCover/table_with_cover.usd"
    policy_ckpt = "Policies/AorticScan"


class Assets(Enums):
    """
    Assets for the robotic ultrasound workflow customized for the user download directory.

    This class is a singleton that will download the assets to the user's local asset directory.
    It has the following attributes:
    - basic: the path to the basic.usda asset
    - panda: the path to the panda_assebly.usda asset
    - phantom: the path to the phantom.usda asset
    - table_with_cover: the path to the table_with cover_.usd asset

    """

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
robotic_ultrasound_assets = Assets()
