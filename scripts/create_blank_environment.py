#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a complete blank, idle-only Workflow scaffold."""

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
    if not (root / "workflows" / "i4h_workflows").is_dir():
        raise SystemExit(f"{root} does not contain the workflow runtime")
    return root


def _existing_workflow_sources(workflows_root: Path, workflow_id: str) -> tuple[Path, ...]:
    """Return every specialty source already claiming the public workflow id."""
    authored_root = workflows_root / "workflows" / "i4h_workflows"
    if not authored_root.is_dir():
        return ()
    return tuple(
        sorted(
            specialty / f"{workflow_id}.py"
            for specialty in authored_root.iterdir()
            if specialty.is_dir() and not specialty.name.startswith("_") and (specialty / f"{workflow_id}.py").is_file()
        )
    )


def _asset_source(workflow_id: str) -> str:
    class_name = _class_name(workflow_id)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ground and lighting for the blank {workflow_id} scene."""

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


def _scene_source(workflow_id: str) -> str:
    class_name = _class_name(workflow_id)
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blank scene for {workflow_id}."""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes.base import Scene


class {class_name}Scene(Scene):
    name = {workflow_id!r}

    def register_assets(self) -> None:
        import i4h_arena.assets.{workflow_id}  # noqa: F401

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.{workflow_id} import make_assets

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=None,
            scene=ArenaScene(assets=make_assets()),
            task=None,
        )
'''


def _scene_manifest(workflow_id: str, description: str) -> str:
    class_name = _class_name(workflow_id)
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

impl: i4h_arena.scenes.{workflow_id}:{class_name}Scene
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


def _workflow_source(workflow_id: str) -> str:
    return f'''# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idle-only Workflow for the blank {workflow_id} Scene."""

from i4h_engine.interface import Workflow
from i4h_workflow_modes.idle import idle

WORKFLOW = Workflow(
    scene={workflow_id!r},
    modes={{"idle": idle}},
)
'''


def _contract_test(workflow_id: str) -> str:
    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from i4h_common.manifest import load_scene_spec
from i4h_engine.loader import load_workflow_module, resolve_workflow


def test_workflow_contract() -> None:
    source = load_workflow_module({workflow_id!r})
    resolved = resolve_workflow({workflow_id!r}, "idle")
    assert source.scene == {workflow_id!r}
    assert "idle" in source.modes
    assert resolved.scene == {workflow_id!r}
    assert resolved.mode == "idle"

    manifest = (
        Path(__file__).parents[2]
        / "arena"
        / "i4h_arena"
        / "scenes"
        / "manifest"
        / {f"{workflow_id}.yaml"!r}
    )
    spec = load_scene_spec(manifest)
    assert spec.impl == "i4h_arena.scenes.{workflow_id}:{_class_name(workflow_id)}Scene"
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
        description="Create a complete blank, idle-only i4h Workflow.",
    )
    parser.add_argument("workflow_id", type=_identifier)
    parser.add_argument(
        "--specialty",
        choices=SPECIALTIES,
        required=True,
        help="product specialty used to group the workflow source",
    )
    parser.add_argument("--description", default="Blank i4h Workflow scene.")
    parser.add_argument("--dry-run", action="store_true", help="print the files without creating them")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run workflow show/lint and the generated contract test after creation",
    )
    return parser


def _validate(workflows_root: Path, workflow_id: str) -> None:
    test_python = workflows_root / "workflows" / ".venv" / "bin" / "python"
    arena_python = workflows_root / "arena" / ".venv" / "bin" / "python"
    if not test_python.is_file():
        raise SystemExit(f"cannot validate without {test_python}; run ./setup.sh first")
    if not arena_python.is_file():
        raise SystemExit(f"cannot validate without {arena_python}; run ./setup.sh first")
    commands = (
        ("show", [workflows_root / "run.sh", "show", workflow_id, "--mode", "idle"]),
        ("lint", [workflows_root / "run.sh", "lint", workflow_id, "--mode", "idle"]),
        ("lint --all", [workflows_root / "run.sh", "lint", "--all"]),
        (
            "contract test",
            [
                test_python,
                "-m",
                "pytest",
                workflows_root / "workflows" / "tests" / f"test_{workflow_id}_contract.py",
                "-q",
            ],
        ),
        (
            "blank-scene runtime test",
            [
                arena_python,
                "-m",
                "pytest",
                workflows_root / "arena" / "tests" / "test_scene.py",
                "-q",
            ],
        ),
    )
    for label, command in commands:
        print(f"validating: {label}", flush=True)
        try:
            subprocess.run([str(part) for part in command], cwd=workflows_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"{label} failed with exit code {exc.returncode}") from None


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.dry_run and args.validate:
        parser.error("--dry-run and --validate cannot be used together")
    root = _repo_root()
    workflows_root = root
    workflow_id = args.workflow_id
    existing_sources = _existing_workflow_sources(workflows_root, workflow_id)
    if existing_sources:
        joined = "\n".join(f"  {path.relative_to(root)}" for path in existing_sources)
        raise SystemExit(
            f"workflow id {workflow_id!r} already exists:\n{joined}\n"
            "edit the existing workflow or choose a different id"
        )
    files = {
        workflows_root / "arena" / "i4h_arena" / "assets" / f"{workflow_id}.py": _asset_source(workflow_id),
        workflows_root / "arena" / "i4h_arena" / "scenes" / f"{workflow_id}.py": _scene_source(workflow_id),
        workflows_root / "arena" / "i4h_arena" / "scenes" / "manifest" / f"{workflow_id}.yaml": _scene_manifest(
            workflow_id, args.description
        ),
        workflows_root / "workflows" / "i4h_workflows" / args.specialty / f"{workflow_id}.py": _workflow_source(
            workflow_id
        ),
        workflows_root / "workflows" / "tests" / f"test_{workflow_id}_contract.py": _contract_test(workflow_id),
    }

    print("blank Workflow files:")
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
    print(f"created blank Workflow {workflow_id!r}")
    if args.validate:
        _validate(workflows_root, workflow_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
