# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers to register Workflow-specific assets and environments."""

from isaaclab_arena_environments.cli import ExampleEnvironments
from scripts.utils.blackwell_render import ensure_blackwell_safe_kit_args
from simulation.environments.g1_locomanip_observe_object_environment import ObserveObjectEnvironment
from simulation.environments.g1_locomanip_push_cart_environment import G1LocomanipPushCartEnvironment
from simulation.environments.g1_locomanip_tray_pick_and_place_environment import G1LocomanipTrayPickAndPlaceEnvironment


def register_workflow_cli():
    """Register into the global registries for CLI before the simulation app is started."""
    ensure_blackwell_safe_kit_args()
    if G1LocomanipTrayPickAndPlaceEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                G1LocomanipTrayPickAndPlaceEnvironment.name: G1LocomanipTrayPickAndPlaceEnvironment,
            }
        )
    if G1LocomanipPushCartEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                G1LocomanipPushCartEnvironment.name: G1LocomanipPushCartEnvironment,
            }
        )
    if ObserveObjectEnvironment.name not in ExampleEnvironments:
        ExampleEnvironments.update(
            {
                ObserveObjectEnvironment.name: ObserveObjectEnvironment,
            }
        )


def _patch_lightwheel_cache_writes():
    """Ensure nested Lightwheel cache dirs exist before the SDK writes metadata.

    The SDK calls ``open(path, "w")`` for ``{object}/{version}.txt`` without always
    creating ``{object}/`` first. Host caches may also mix flat and nested layouts.
    """
    import builtins
    import os

    if getattr(builtins, "_rheo_lightwheel_open_patch", False):
        return

    _orig_open = builtins.open

    def _open_with_lightwheel_cache_dirs(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = os.fspath(file)
            marker = f"{os.sep}.cache{os.sep}lightwheel_sdk{os.sep}object{os.sep}"
            if marker in path:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
        return _orig_open(file, *args, **kwargs)

    builtins.open = _open_with_lightwheel_cache_dirs
    builtins._rheo_lightwheel_open_patch = True


def register_workflow_assets():
    """Register Rheo-specific assets into the global AssetRegistry.

    Arena's ``@register_asset`` decorator calls ``is_registered()``, which normally
    triggers ``ensure_assets_registered()`` and imports the full Arena library
    (Lightwheel SDK downloads, G1/Pinocchio modules, etc.). That cascade while
    SimulationApp is running has caused SIGSEGV in Kit Python. We register workflow
    assets first; the full Arena registry loads later via ``get_asset_by_name()``.
    """
    from contextlib import contextmanager

    import isaaclab_arena.assets.asset_registry as asset_registry

    @contextmanager
    def _skip_eager_arena_registry():
        def _is_registered_local(self, name: str) -> bool:
            return name in self._components

        orig_asset = asset_registry.AssetRegistry.is_registered
        orig_device = asset_registry.DeviceRegistry.is_registered
        asset_registry.AssetRegistry.is_registered = _is_registered_local
        asset_registry.DeviceRegistry.is_registered = _is_registered_local
        try:
            yield
        finally:
            asset_registry.AssetRegistry.is_registered = orig_asset
            asset_registry.DeviceRegistry.is_registered = orig_device

    with _skip_eager_arena_registry():
        _patch_lightwheel_cache_writes()
        import scripts.teleop_devices.motion_controllers  # noqa: F401
        import simulation.assets.background_library  # noqa: F401
        import simulation.assets.object_library  # noqa: F401
        import simulation.embodiments.g1_patched  # noqa: F401
