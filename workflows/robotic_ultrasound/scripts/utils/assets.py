import os
from dataclasses import dataclass

from i4h_asset_helper import get_i4h_local_asset_path


@dataclass
class Enums:
    basic = "Test/basic.usda"
    panda = "Robots/Franka/Collected_panda_assembly/panda_assembly.usda"
    phantom = "Props/ABDPhantom/phantom.usda"
    table_with_cover = "Props/VentionTableWithBlackCover/table_with_cover.usd"


class Assets(Enums):
    def __init__(self, download_dir: str | None = None):
        if download_dir is None:
            download_dir = get_i4h_local_asset_path()
        # Update all the paths
        for attr in dir(self):
            if not attr.startswith("_"):
                setattr(self, attr, os.path.join(download_dir, getattr(self, attr)))


# singleton
robotic_ultrasound_assets = Assets()
