# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned tablecloth robot assets and recoverable local H2 cache."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.7.0/724f82e"
G1_INSPIRE_USD_PATH = f"{_ASSET_ROOT}/Robots/UnitreeG1/g1_29dof_with_inspire_rev_1_0/g1_29dof_with_inspire_rev_1_0.usd"
H2_SHARPA_USD_PATH = f"{_ASSET_ROOT}/Robots/UnitreeH2/h2_with_sharpa/H2_with_sharpa_flat.usd"
# ---------------------------------------------------------------------------
# H2 + Sharpa non-USD asset resolution (urdf/ + teleop_configs/)
# ---------------------------------------------------------------------------
_H2_SHARPA_ASSET_ROOT = os.path.expanduser(
    os.environ.get(
        "RHEO_H2_SHARPA_ASSETS_DIR",
        "~/.cache/i4h_workflows/spread_tablecloth/h2_with_sharpa",
    )
)
H2_SHARPA_URDF_PATH = os.path.join(_H2_SHARPA_ASSET_ROOT, "urdf", "H2_with_sharpa_hands.urdf")
H2_SHARPA_HAND_URDF_DIR = os.path.join(_H2_SHARPA_ASSET_ROOT, "urdf", "sharpa_standalone")
H2_SHARPA_TELEOP_CONFIG_DIR = os.path.join(_H2_SHARPA_ASSET_ROOT, "teleop_configs")

_H2_SHARPA_REQUIRED_FILES = {
    "urdf": (
        "H2_with_sharpa_hands.urdf",
        "sharpa_standalone/left_sharpa_wave.urdf",
        "sharpa_standalone/right_sharpa_wave.urdf",
    ),
    "teleop_configs": (
        "sharpa_wave_left_dexpilot.yml",
        "sharpa_wave_right_dexpilot.yml",
    ),
}


def _cache_complete(directory: Path, required: tuple[str, ...]) -> bool:
    """Check entry points and the relative geometry referenced by the pinned URDFs."""
    try:
        for relative in required:
            path = directory / relative
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if path.suffix == ".urdf":
                for element in ET.parse(path).iter():
                    if element.tag not in {"mesh", "texture"}:
                        continue
                    dependency = path.parent / element.attrib["filename"]
                    if not dependency.is_file() or dependency.stat().st_size == 0:
                        return False
    except (OSError, ET.ParseError, KeyError):
        return False
    return True


def ensure_h2_sharpa_assets(usd_url: str = H2_SHARPA_USD_PATH) -> None:
    """Repair missing H2 URDFs, geometry, and retargeting configs in the local cache."""
    for subdir, required in _H2_SHARPA_REQUIRED_FILES.items():
        local = Path(_H2_SHARPA_ASSET_ROOT) / subdir
        if _cache_complete(local, required):
            continue
        if not usd_url.startswith(("omniverse://", "http://", "https://")):
            raise FileNotFoundError(
                f"Incomplete H2 assets in {local}; {usd_url!r} is not a remote URL. "
                "Set RHEO_H2_SHARPA_ASSETS_DIR to a complete local bundle."
            )
        import omni.client  # noqa: PLC0415 (Kit is only needed for a cache miss)

        local.mkdir(parents=True, exist_ok=True)
        remote = usd_url.rsplit("/", 1)[0] + "/" + subdir + "/"
        print(f"[tablecloth] downloading {remote} -> {local}", flush=True)
        result = omni.client.copy(remote, str(local), omni.client.CopyBehavior.OVERWRITE)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"omni.client.copy {remote} -> {result}")
        if not _cache_complete(local, required):
            raise RuntimeError(f"H2 asset download is incomplete: {local}; retry to repair the cache")
