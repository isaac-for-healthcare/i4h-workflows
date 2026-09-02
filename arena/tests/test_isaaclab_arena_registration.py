# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.assets import asset_registry
from isaaclab_arena.assets.register import register_asset


def test_asset_decorator_does_not_load_global_catalog(monkeypatch) -> None:
    registry = asset_registry.AssetRegistry()
    name = "i4h_test_local_registration"
    registry._components.pop(name, None)

    def fail_if_called() -> None:
        raise AssertionError("asset decorator loaded the global catalog")

    monkeypatch.setattr(asset_registry, "ensure_assets_registered", fail_if_called)

    class LocalAsset:
        pass

    LocalAsset.name = name
    try:
        assert register_asset(LocalAsset) is LocalAsset
        assert registry.is_registered_local(name)
    finally:
        registry._components.pop(name, None)


def test_local_asset_lookup_does_not_load_global_catalog(monkeypatch) -> None:
    registry = asset_registry.AssetRegistry()
    name = "i4h_test_local_lookup"

    class LocalAsset:
        pass

    registry._components[name] = LocalAsset

    def fail_if_called() -> None:
        raise AssertionError("local asset lookup loaded the global catalog")

    monkeypatch.setattr(asset_registry, "ensure_assets_registered", fail_if_called)
    try:
        assert registry.get_asset_by_name(name) is LocalAsset
    finally:
        registry._components.pop(name, None)
