# Organ material replacements

The surgical organ scene uses a public USD file (a file describing a 3D scene).
That file points to four missing Bladder texture maps, a skin texture on an
artist's Windows computer, and a malformed path for a wall material.

`organs.usda` is a small override layer. It loads the original scene and changes
only the affected material inputs. `ORGANS_USD` in `assets/constants.py` selects
this layer automatically; no manual asset download or cache editing is needed.
The original scene and its remaining assets still require network access.

## What is included

Four RGB PNG images live under `materials/organs/Bladder/`. Each is 1254 × 1254
pixels. The filenames retain the upstream naming scheme and the `1001` suffix.
The override uses these explicit filenames, rather than a `<UDIM>` pattern
that asks the renderer to discover missing numbered image tiles.

- `diffuseReflectionColor`: the pale pink surface color, read as sRGB.
- `geometryNormal`: small surface details, encoded as a tangent-space normal map.
- `specularReflectionRoughness`: how smooth or rough reflections appear.
- `subsurfaceWeight`: how strongly the shader blends light scattering inside the tissue.

Normal, roughness, and subsurface maps are read as raw data, without the color
correction used for a photograph. The existing scalar inputs use the red
channel of the roughness and subsurface maps.

The source scene points both its bladder and prostate materials at the missing
Bladder maps. This layer preserves that sharing. It does not claim that a
bladder texture is an accurate prostate material.

The existing skin-detail image and wall `Plastic.mdl` are referenced through
their working URLs inside the original versioned asset package.

## Texture provenance and limits

These are new, AI-generated **synthetic visual replacements**, not recovered
source textures or measurements of real tissue. They were generated with the
built-in image generation tool on 2026-10-05. The color map was generated first;
the other three maps used it as a reference. The original PNG outputs are
included without pixel edits. Exact prompts are in `generation-prompts.json`.

The images are art-directed approximations. Their normal directions, material
values, and seams need visual review in the intended scene. They are not a
validated biological material, and should be reviewed before producing training
or evaluation datasets. Restore the original maps if they become available.

The generated maps are contributed under this repository's Apache-2.0 license.
This directory does not redistribute the upstream scene or its other assets.

## Check the fix

With the Arena environment installed, run:

```bash
./run.sh surgical_lift_needle_organs --rule-based --episodes 1
```

Check the appearance and confirm that the log no longer reports missing Bladder
textures, the Windows skin-normal path, or the blue wall's `Plastic.mdl` path.
The episode passing by itself does not prove that the materials loaded.

The CPU tests exercise USD composition and local texture resolution without
starting Isaac Sim:

```bash
arena/.venv/bin/python -m pytest arena/tests/test_organs_materials.py
```
