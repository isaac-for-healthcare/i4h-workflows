# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets import registries
from isaaclab_arena.assets.register import register_asset


def test_asset_decorator_does_not_load_global_catalog(monkeypatch) -> None:
    # The pinned Arena checks local registration without loading its global catalog.
    registry = registries.AssetRegistry()
    name = "i4h_test_local_registration"
    registry._components.pop(name, None)

    def fail_if_called() -> None:
        raise AssertionError("asset decorator loaded the global catalog")

    monkeypatch.setattr(registries, "ensure_assets_registered", fail_if_called)

    class LocalAsset:
        pass

    LocalAsset.registry_name = name
    try:
        assert register_asset(LocalAsset) is LocalAsset
        assert registry.is_registered(name, ensure_loaded=False)
    finally:
        registry._components.pop(name, None)
