#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a blank workflow: a scene that runs as an IsaacLab-Arena environment.

The scene, its ground-and-light assets and its manifest are all a new
environment needs: ``i4h_arena.environments`` exposes every scene in the
manifest as an Arena external environment named after it in CamelCase. The
workflow declaration under the specialty names the scene, so ``./run.sh list``
lists it and ``./run.sh <workflow>`` opens it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
SPECIALTIES = (
    "laparoscopic-robotics",
    "ultrasound-robotics",
    "endoluminal-robotics",
    "hospital-automation-robotics",
)


def _identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise argparse.ArgumentTypeError("use a lowercase snake_case identifier")
    return value


def _class_name(identifier: str) -> str:
    return "".join(part.capitalize() for part in identifier.split("_"))


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "arena" / "i4h_arena" / "scenes" / "manifest").is_dir():
        raise SystemExit(f"{root} does not contain the i4h scenes")
    return root


def _asset_source(scene_id: str) -> str:
    class_name = _class_name(scene_id)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground and lighting for the blank {scene_id} scene."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.asset import Asset


class ConfigAsset(Asset):
    """Arena wrapper around an Isaac Lab scene config."""

    def __init__(self, name: str, cfg: Any):
        super().__init__(name=name, tags=["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> tuple[str, Any]:
        return self.name, self._cfg

    def get_event_cfg(self) -> tuple[str, None]:
        return self.name, None


@configclass
class {class_name}SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),
    )


def make_assets() -> list[ConfigAsset]:
    """Return the minimal blank-world assets."""
    source = {class_name}SceneCfg(env_spacing=4.0)
    return [ConfigAsset(name, deepcopy(getattr(source, name))) for name in ("ground", "light")]
'''


def _scene_source(scene_id: str) -> str:
    class_name = _class_name(scene_id)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blank scene for {scene_id}."""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes.base import Scene


class {class_name}Scene(Scene):
    name = {scene_id!r}

    def register_assets(self) -> None:
        import i4h_arena.assets.{scene_id}  # noqa: F401

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.{scene_id} import make_assets

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=None,
            scene=ArenaScene(assets=make_assets()),
            task=None,
        )
'''


def _scene_manifest(scene_id: str, description: str) -> str:
    class_name = _class_name(scene_id)
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

impl: i4h_arena.scenes.{scene_id}:{class_name}Scene
description: {json.dumps(description)}
embodiment: none
action_space: joint_position
dof: 0
gripper: false
cameras: []
objects: []
robots: []
max_steps: 600
control_hz: 60.0
"""


def _workflow_declaration(scene_id: str, description: str) -> str:
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

description: {json.dumps(description)}
scene: {scene_id}
# Without modes the workflow runs idle, live and replay; declare rule-based, policy and
# teleop modes here as the scene gains a robot and behavior.
"""


def _contract_test(scene_id: str) -> str:
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec


def test_scene_contract() -> None:
    manifest = Path(__file__).parents[1] / "arena" / "i4h_arena" / "scenes" / "manifest" / {f"{scene_id}.yaml"!r}
    spec = load_scene_spec(manifest)
    assert spec.impl == "i4h_arena.scenes.{scene_id}:{_class_name(scene_id)}Scene"
    assert spec.max_steps > 0
    assert spec.control_hz > 0.0
    assert spec.dof >= 0
    if spec.embodiment == "none":
        assert spec.dof == 0
        assert spec.robots == ()
    else:
        assert spec.dof > 0
        assert spec.robots
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_blank_environment.py",
        description="Create a blank i4h workflow whose scene runs as an IsaacLab-Arena environment.",
    )
    parser.add_argument("scene_id", type=_identifier)
    parser.add_argument(
        "--specialty",
        choices=SPECIALTIES,
        required=True,
        help="the specialty that lists the workflow: writes workflows/<specialty>/<scene_id>.yaml",
    )
    parser.add_argument("--description", default="Blank i4h scene.")
    parser.add_argument("--dry-run", action="store_true", help="print the files without creating them")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the generated contract test, the scene tests and the declaration tests after creation",
    )
    return parser


def _validate(root: Path, scene_id: str) -> None:
    common_python = root / "common" / ".venv" / "bin" / "python"
    arena_python = root / "arena" / ".venv" / "bin" / "python"
    for python in (common_python, arena_python):
        if not python.is_file():
            raise SystemExit(f"cannot validate without {python}; run ./setup.sh first")
    commands = (
        ("contract test", [common_python, "-m", "pytest", root / "tests" / f"test_{scene_id}_contract.py", "-q"]),
        ("scene tests", [arena_python, "-m", "pytest", root / "arena" / "tests" / "test_scene.py", "-q"]),
        (
            "workflow declaration",
            [common_python, "-m", "pytest", root / "tests" / "test_run_workflow.py", "-q", "-k", scene_id],
        ),
    )
    for label, command in commands:
        print(f"validating: {label}", flush=True)
        try:
            subprocess.run([str(part) for part in command], cwd=root, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{label} failed with exit code {exc.returncode}") from None


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.dry_run and args.validate:
        parser.error("--dry-run and --validate cannot be used together")
    root = _repo_root()
    scene_id = args.scene_id
    manifest = root / "arena" / "i4h_arena" / "scenes" / "manifest" / f"{scene_id}.yaml"
    if manifest.exists():
        raise SystemExit(
            f"scene {scene_id!r} already exists: {manifest.relative_to(root)}\n"
            "edit the existing scene or choose a different id"
        )
    # Workflow ids are unique across specialties: run.sh finds a workflow by its id alone.
    declared = sorted((root / "workflows").glob(f"*/{scene_id}.yaml"))
    if declared:
        raise SystemExit(
            f"workflow {scene_id!r} already exists: {declared[0].relative_to(root)}\n"
            "edit the existing workflow or choose a different id"
        )
    files = {
        root / "arena" / "i4h_arena" / "assets" / f"{scene_id}.py": _asset_source(scene_id),
        root / "arena" / "i4h_arena" / "scenes" / f"{scene_id}.py": _scene_source(scene_id),
        manifest: _scene_manifest(scene_id, args.description),
        root / "workflows" / args.specialty / f"{scene_id}.yaml": _workflow_declaration(scene_id, args.description),
        root / "tests" / f"test_{scene_id}_contract.py": _contract_test(scene_id),
    }

    print("blank workflow files:")
    for path in files:
        print(f"  {path.relative_to(root)}")
    if args.dry_run:
        print("dry run; no files created")
        return 0

    collisions = [path for path in files if path.exists()]
    if collisions:
        joined = "\n".join(f"  {path}" for path in collisions)
        raise SystemExit(f"refusing to overwrite existing files:\n{joined}")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(
        f"created blank workflow {scene_id!r} in {args.specialty}; open it with ./run.sh {scene_id}; "
        f"its scene is the Arena environment i4h_arena.environments:{_class_name(scene_id)} {scene_id}"
    )
    if args.validate:
        _validate(root, scene_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
