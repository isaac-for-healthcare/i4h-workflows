# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material regression tests; require OpenUSD and Pillow, but not Isaac Sim."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
from PIL import Image
from pxr import Sdf, Usd

ASSETS = Path(__file__).parents[1] / "i4h_arena" / "assets"
LAYER = ASSETS / "usd" / "organs" / "organs.usda"
UPSTREAM = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/Healthcare/0.5.0/132c82d/Props/Organs/"
)
CHANNELS = {
    "diffuseReflectionColor_texture": ("diffuseReflectionColor", "sRGB"),
    "normalmap_texture": ("geometryNormal", "raw"),
    "specularReflectionRoughness_texture": ("specularReflectionRoughness", "raw"),
    "subsurfaceWeight_texture": ("subsurfaceWeight", "raw"),
}
MATERIALS = ("c_organ_graph_bladder", "c_organ_graph_prostate")
WALL = "/World/OR_Room/Over_GRP_Room_Additions_merged/Looks/M_Plastic_Wall_BLUE/Shader"


@pytest.fixture
def composed_stage(tmp_path):
    """Compose the real override over a small offline stand-in for the remote scene."""
    source = Usd.Stage.CreateNew(str(tmp_path / "source.usda"))
    world = source.DefinePrim("/World", "Xform")
    source.SetDefaultPrim(world)
    mesh = source.DefinePrim("/World/Geometry", "Cube")
    mesh.CreateAttribute("size", Sdf.ValueTypeNames.Double).Set(0.2)
    mesh.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0.75)
    for material in MATERIALS:
        for shader in (*CHANNELS, "normalmap_texture_01"):
            path = f"/World/organ_shaders/Looks/{material}/{shader}"
            prim = source.DefinePrim(path, "Shader")
            prim.CreateAttribute("inputs:texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("missing.png"))
            if shader in CHANNELS and shader != "normalmap_texture":
                child = source.DefinePrim(path + "/file_texture", "Shader")
                child.CreateAttribute("inputs:texture", Sdf.ValueTypeNames.Asset).SetConnections(
                    [Sdf.Path(path + ".inputs:texture")]
                )
    wall = source.DefinePrim(WALL, "Shader")
    wall.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("missing.mdl"))
    source.GetRootLayer().Save()

    package = tmp_path / "package"
    shutil.copytree(LAYER.parent, package)
    override = Sdf.Layer.FindOrOpen(str(package / LAYER.name))
    override.GetPrimAtPath("/World").referenceList.prependedItems = [
        Sdf.Reference(source.GetRootLayer().identifier, "/World")
    ]
    override.Save()
    return Usd.Stage.Open(override)


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("shader,channel", CHANNELS.items())
def test_replacement_maps_resolve_and_keep_connections(composed_stage, material, shader, channel):
    name, space = channel
    path = f"/World/organ_shaders/Looks/{material}/{shader}"
    attr = composed_stage.GetAttributeAtPath(path + ".inputs:texture")
    asset = attr.Get()
    assert Path(asset.resolvedPath).is_file()
    assert Path(asset.path).name == f"Bladder_topo_blender_{name}.1001.png"
    assert "<UDIM>" not in asset.path
    assert attr.GetColorSpace() == space
    with Image.open(asset.resolvedPath) as image:
        image.verify()
    if shader != "normalmap_texture":
        child = composed_stage.GetAttributeAtPath(path + "/file_texture.inputs:texture")
        assert child.GetConnections() == [Sdf.Path(path + ".inputs:texture")]
        assert child.GetColorSpace() == space


def test_external_material_paths_and_geometry_are_preserved(composed_stage):
    for material in MATERIALS:
        attr = composed_stage.GetAttributeAtPath(
            f"/World/organ_shaders/Looks/{material}/normalmap_texture_01.inputs:texture"
        )
        assert attr.Get().path == UPSTREAM + "materials/human_skin_normal_detail.jpg"
    wall = composed_stage.GetAttributeAtPath(WALL + ".info:mdl:sourceAsset")
    assert wall.Get().path == UPSTREAM + "materials/operating_room/Base/Plastics/Plastic.mdl"
    assert composed_stage.GetAttributeAtPath("/World/Geometry.size").Get() == 0.2
    assert composed_stage.GetAttributeAtPath("/World/Geometry.physics:mass").Get() == 0.75


def test_workflow_uses_packaged_layer_and_original_scene():
    spec = importlib.util.spec_from_file_location("asset_constants", ASSETS / "constants.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    assert Path(constants.ORGANS_USD) == LAYER.resolve()
    layer = Sdf.Layer.FindOrOpen(constants.ORGANS_USD)
    assert layer.defaultPrim == "World"
    assert layer.GetPrimAtPath("/World").referenceList.prependedItems == [
        Sdf.Reference(UPSTREAM + "organs.usd", "/World")
    ]


def test_override_only_authors_material_attributes():
    layer = Sdf.Layer.FindOrOpen(str(LAYER))
    attributes = []

    def check(path):
        obj = layer.GetObjectAtPath(path)
        if isinstance(obj, Sdf.AttributeSpec):
            attributes.append(obj)
            assert obj.name in ("inputs:texture", "info:mdl:sourceAsset")
        elif isinstance(obj, Sdf.PrimSpec) and path != Sdf.Path("/World"):
            assert obj.specifier == Sdf.SpecifierOver
            assert not obj.typeName

    layer.Traverse("/", check)
    # Ten organ texture values, one wall material, and six connected-input color-space opinions.
    assert len(attributes) == 17
    assert sum(attr.HasDefaultValue() for attr in attributes) == 11
