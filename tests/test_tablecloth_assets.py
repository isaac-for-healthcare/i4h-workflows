# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H2 cache recovery must work without loading Isaac Sim into CPU CI."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).parents[1] / "arena/i4h_arena/embodiments/_tablecloth_assets.py"


def _write_bundle(root):
    for relative in (
        "urdf/H2_with_sharpa_hands.urdf",
        "urdf/sharpa_standalone/left_sharpa_wave.urdf",
        "urdf/sharpa_standalone/right_sharpa_wave.urdf",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh = path.with_suffix(".stl")
        mesh.write_text("mesh data")
        path.write_text(f'<robot name="h2"><link name="body"><mesh filename="{mesh.name}"/></link></robot>')
    directory = root / "teleop_configs"
    directory.mkdir(parents=True)
    for side in ("left", "right"):
        (directory / f"sharpa_wave_{side}_dexpilot.yml").write_text("retargeting: {}\n")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("tablecloth_assets", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    _write_bundle(remote)
    shutil.copytree(remote, local)
    monkeypatch.setattr(module, "_H2_SHARPA_ASSET_ROOT", str(local))

    def copy(url, destination, behavior):
        shutil.copytree(remote / url.rstrip("/").rsplit("/", 1)[1], destination, dirs_exist_ok=True)
        return "OK"

    client = ModuleType("omni.client")
    client.copy = Mock(side_effect=copy)
    client.CopyBehavior = SimpleNamespace(OVERWRITE="overwrite")
    client.Result = SimpleNamespace(OK="OK")
    omni = ModuleType("omni")
    omni.client = client
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.client", client)
    return SimpleNamespace(module=module, local=local, client=client)


def test_complete_local_bundle_needs_no_network(cache):
    cache.module.ensure_h2_sharpa_assets("/offline/H2.usd")
    cache.client.copy.assert_not_called()


@pytest.mark.parametrize(
    "relative,contents",
    [
        ("urdf/H2_with_sharpa_hands.urdf", None),
        ("urdf/sharpa_standalone/left_sharpa_wave.urdf", None),
        ("urdf/sharpa_standalone/right_sharpa_wave.urdf", None),
        ("urdf/H2_with_sharpa_hands.stl", None),
        ("urdf/sharpa_standalone/right_sharpa_wave.stl", None),
        ("teleop_configs/sharpa_wave_left_dexpilot.yml", None),
        ("teleop_configs/sharpa_wave_right_dexpilot.yml", None),
        ("urdf/H2_with_sharpa_hands.urdf", "<robot"),
        ("urdf/sharpa_standalone/left_sharpa_wave.stl", ""),
        ("teleop_configs/sharpa_wave_right_dexpilot.yml", ""),
    ],
)
def test_partial_cache_is_repaired(cache, relative, contents):
    path = cache.local / relative
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    cache.module.ensure_h2_sharpa_assets()
    assert path.stat().st_size > 0
    cache.client.copy.assert_called_once()
    cache.module.ensure_h2_sharpa_assets()
    assert cache.client.copy.call_count == 1


@pytest.mark.parametrize("result", ["INTERRUPTED", "OK"])
def test_interrupted_or_incomplete_download_is_retried(cache, result):
    shutil.rmtree(cache.local / "urdf")
    successful_copy = cache.client.copy.side_effect

    def partial_copy(url, destination, behavior):
        # The old single marker exists, but the hand descriptions and meshes do not.
        (Path(destination) / "H2_with_sharpa_hands.urdf").write_text('<robot name="h2"/>')
        return result

    cache.client.copy.side_effect = partial_copy
    with pytest.raises(RuntimeError):
        cache.module.ensure_h2_sharpa_assets()
    cache.client.copy.side_effect = successful_copy
    cache.module.ensure_h2_sharpa_assets()
    assert cache.client.copy.call_count == 2
    assert (cache.local / "urdf/sharpa_standalone/right_sharpa_wave.stl").is_file()
