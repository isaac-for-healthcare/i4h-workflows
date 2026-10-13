# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply the material repair recipe to an original organs.usd for publication.

The source package must already contain the four replacement PNGs and original
dependencies. Write to a new sibling file, inspect it, then replace organs.usd
in the staging copy only. Requires usd-core or Isaac Sim's USD runtime.
"""

import argparse
from pathlib import Path

UPSTREAM = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/Healthcare/0.5.0/132c82d/Props/Organs/"
)


def apply_material_repairs(layer):
    """Copy only material values/color spaces, never the recipe's scene reference."""
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(layer)
    recipe = Sdf.Layer.FindOrOpen(str(Path(__file__).with_name("organs.usda")))
    attributes = []

    def collect(path):
        spec = recipe.GetObjectAtPath(path)
        if isinstance(spec, Sdf.AttributeSpec):
            attributes.append(spec)

    recipe.Traverse("/", collect)
    # Check the input scene before authoring anything.
    for spec in attributes:
        if not stage.GetAttributeAtPath(spec.path):
            raise ValueError(f"Missing original material attribute: {spec.path}")

    for spec in attributes:
        attr = stage.GetAttributeAtPath(spec.path)
        if spec.HasDefaultValue():
            value = spec.default
            if isinstance(value, Sdf.AssetPath) and value.path.startswith(UPSTREAM):
                value = Sdf.AssetPath("./" + value.path.removeprefix(UPSTREAM))
            attr.Set(value)
        if spec.HasInfo("colorSpace"):
            attr.SetColorSpace(spec.GetInfo("colorSpace"))
    return len(attributes)


def main(argv=None):
    from pxr import Sdf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file() or source.parent != output.parent:
        parser.error("Source must exist and output must be a sibling to preserve relative dependencies")
    if output.exists() or args.output.is_symlink():
        parser.error("Output already exists")
    if output.suffix not in {".usd", ".usda", ".usdc"}:
        parser.error("Output must be a USD layer")
    # Keep the original layer's identifier while resolving its relative references.
    layer = Sdf.Layer.FindOrOpen(str(source))
    apply_material_repairs(layer)
    with output.open("xb"):
        pass
    try:
        if not layer.Export(str(output)):
            raise RuntimeError(f"Could not export {output}")
    except Exception:
        output.unlink()
        raise
    print(f"Prepared {output}; original source file is unchanged")


if __name__ == "__main__":
    main()
