# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guardrails for the architectural invariants in DESIGN.md §3.

Static — these read source, they never import — so they run anywhere, including
in a checkout with nothing synced. Breaking one of these does not break a test
somewhere later; it breaks the property the whole design rests on.

**Two kinds of forbidden import**, because a function-local import means two
different things in this tree:

``never``
    Illegal at any scope. These are *layering* violations: ``common`` reaching
    up to ``arena``, a policy stack reaching sideways to ``arena``. Hiding one
    inside a function does not make it legal — it makes it harder to find.

``lazy_only``
    Illegal at module scope, permitted inside a function. These are *heavy or
    optional* dependencies whose import has global side effects or costs
    seconds, such as the simulator and model stacks the RL launcher reaches only
    after its preflight.

Run with:  uv run --project common --extra dev pytest tests/
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_gr00t_n17_finetune_resolves_its_local_modality_config() -> None:
    """Keep the fine-tune shim's source-relative path valid after tree moves."""
    source = ROOT / "tasks" / "gr00t_n17" / "i4h_tasks" / "gr00t_n17" / "_finetune.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_modality_config_path"
    )
    namespace = {"Path": Path, "__file__": str(source)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)

    resolved = namespace["_modality_config_path"]()
    assert resolved == source.parent / "config.py"
    assert resolved.is_file()


@dataclass(frozen=True)
class LayerRule:
    """What one project may not import, and at which scope."""

    source: str
    #: Illegal anywhere, including inside a function body.
    never: frozenset[str] = field(default_factory=frozenset)
    #: Illegal at module scope; a deliberate lazy import is fine.
    lazy_only: frozenset[str] = field(default_factory=frozenset)

    def __str__(self) -> str:
        return self.source


#: Internal packages, in layering order. Anything at or above a project's own
#: level is `never` for that project.
RULES = [
    LayerRule(
        "common/i4h_common",
        never=frozenset({"i4h_tasks", "i4h_tools", "i4h_arena", "i4h_rl", "isaacsim", "isaaclab", "torch"}),
    ),
    LayerRule(
        # THE invariant: the arena process must never import a policy stack.
        # Their torch pins conflict with Isaac's, which is why each checkpoint is
        # served from its own environment. Only the policy clients Arena itself
        # uses (GR00T's gr00t.policy.server_client, openpi-client) cross over; the
        # model, its configs and its training code stay out.
        "arena/i4h_arena",
        never=frozenset(
            {
                "i4h_tools",
                "i4h_tasks.gr00t_n15",
                "i4h_tasks.gr00t_n16",
                "i4h_tasks.gr00t_n17",
                "i4h_tasks.openpi_pi0",
                "gr00t.configs",
                "gr00t.eval",
                "gr00t.experiment",
                "gr00t.model",
                "openpi",
            }
        ),
    ),
    *[
        LayerRule(
            f"tools/{name}/i4h_tools",
            never=frozenset({"i4h_arena", "i4h_rl", "isaacsim", "isaaclab"}),
        )
        for name in ("mimic", "dataset", "cosmos", "annotator")
    ],
    *[
        LayerRule(
            # A policy server is a leaf: nothing may depend on it, and it needs
            # nothing of ours but `common`. `isaaclab_arena` is listed because
            # reaching into the Arena source tree for joint-remap helpers forces a
            # multi-GB checkout on a process that never touches the simulator.
            f"tasks/{name}/i4h_tasks",
            never=frozenset(
                {
                    "i4h_arena",
                    "i4h_tools",
                    "i4h_rl",
                    "isaacsim",
                    "isaaclab",
                    "isaaclab_arena",
                    "isaaclab_arena_gr00t",
                    "isaaclab_arena_g1",
                }
            ),
        )
        # gr00t_n16 uses the Arena G1 WBC joint-remapping implementation.
        for name in ("gr00t_n15", "gr00t_n17", "openpi_pi0")
    ],
    LayerRule(
        "tasks/gr00t_n16/i4h_tasks",
        never=frozenset({"i4h_arena", "i4h_tools", "i4h_rl", "isaacsim", "isaaclab"}),
    ),
    LayerRule(
        # Online RL is the deliberate integration boundary: the lightweight
        # resolver stays importable without the simulator/model stack, while
        # the launcher and worker extension may import both after preflight.
        "rl/i4h_rl",
        never=frozenset({"i4h_tools"}),
        lazy_only=frozenset({"i4h_arena", "gr00t", "isaacsim", "isaaclab", "isaaclab_arena", "rlinf", "torch"}),
    ),
]


#: Neither is ours to police: .venv is installed output, third_party is
#: upstream source vendored at a pinned revision.
SKIP = (".venv", "third_party")


def ours(path: Path) -> bool:
    return not any(part in SKIP for part in path.parts)


def _python_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.rglob("*.py") if ours(p)]


def _is_type_checking_guard(node: ast.stmt) -> bool:
    """``if TYPE_CHECKING:`` bodies never execute, so they cannot violate layering."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _imported_modules(node: ast.stmt) -> list[str]:
    """Dotted module names an import statement pulls in."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        # `from a.b import c` may import a module `a.b.c`; check both.
        return [node.module, *[f"{node.module}.{alias.name}" for alias in node.names]]
    return []


def collect_imports(path: Path) -> tuple[set[str], set[str]]:
    """Return ``(module_level, function_level)`` dotted module names.

    ``if TYPE_CHECKING:`` bodies are excluded from both — they are annotations,
    not imports.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    module_level: set[str] = set()
    function_level: set[str] = set()

    def walk(body: list[ast.stmt], *, lazy: bool) -> None:
        target = function_level if lazy else module_level
        for node in body:
            if _is_type_checking_guard(node):
                continue
            if isinstance(node, ast.Import | ast.ImportFrom):
                target.update(_imported_modules(node))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(node.body, lazy=True)
            elif isinstance(node, ast.ClassDef):
                walk(node.body, lazy=lazy)
            elif isinstance(node, ast.If | ast.Try | ast.While | ast.For | ast.With):
                walk(node.body, lazy=lazy)
                walk(getattr(node, "orelse", []), lazy=lazy)
                walk(getattr(node, "finalbody", []), lazy=lazy)
                for handler in getattr(node, "handlers", []):
                    walk(handler.body, lazy=lazy)

    walk(tree.body, lazy=False)
    return module_level, function_level


def _matches(module: str, forbidden: str) -> bool:
    """``i4h_tasks.gr00t_n15`` matches itself and anything beneath it, not ``i4h_tasks.basic``."""
    return module == forbidden or module.startswith(f"{forbidden}.")


def _violations(modules: set[str], forbidden: frozenset[str]) -> list[str]:
    """Offending modules, collapsed to their shortest form.

    ``from a.b import c`` records both ``a.b`` and ``a.b.c`` because ``c`` might
    be a submodule; for reporting, only ``a.b`` is interesting.
    """
    hits = {m for m in modules for f in forbidden if _matches(m, f)}
    return sorted(m for m in hits if not any(other != m and _matches(m, other) for other in hits))


@pytest.mark.parametrize("rule", RULES, ids=[r.source for r in RULES])
def test_no_forbidden_import_at_any_scope(rule: LayerRule):
    """`never` imports are illegal even inside a function.

    A lazy import is the escape hatch for a *heavy* dependency, not for a
    layering violation. Wrapping one in a function hides it from review without
    making it legal.
    """
    problems: list[str] = []
    for path in _python_files(ROOT / rule.source):
        module_level, function_level = collect_imports(path)
        where = path.relative_to(ROOT)
        for module in _violations(module_level, rule.never):
            problems.append(f"{where} imports {module!r}")
        for module in _violations(function_level, rule.never):
            problems.append(f"{where} lazily imports {module!r} (still a layering violation)")
    assert not problems, f"{rule.source} must never import these:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("rule", [r for r in RULES if r.lazy_only], ids=[r.source for r in RULES if r.lazy_only])
def test_heavy_dependencies_are_imported_lazily(rule: LayerRule):
    """`lazy_only` imports must not be at module scope.

    Importing ``isaacsim`` at module scope would boot Kit before the RL launcher has
    resolved and checked its profile, which its preflight is arranged to prevent.
    """
    problems: list[str] = []
    for path in _python_files(ROOT / rule.source):
        module_level, _ = collect_imports(path)
        for module in _violations(module_level, rule.lazy_only):
            problems.append(f"{path.relative_to(ROOT)} imports {module!r} at module scope")
    assert not problems, (
        f"{rule.source} must import these lazily, inside the function that needs them:\n  " + "\n  ".join(problems)
    )


# -- the checker itself has to work --------------------------------------


def test_checker_separates_module_and_function_scope(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def f():\n"
        "    import json\n"
        "    from i4h_arena.scenes.base import scene_specs\n"
        "\n"
        "class C:\n"
        "    import csv\n"
        "    def m(self):\n"
        "        import sqlite3\n"
    )
    module_level, function_level = collect_imports(source)
    assert {"os", "pathlib"} <= module_level
    assert "csv" in module_level  # class body executes at import time
    assert {"json", "i4h_arena.scenes.base", "sqlite3"} <= function_level
    assert "json" not in module_level


def test_checker_catches_a_lazy_layering_violation(tmp_path):
    """A lazy import hidden in a function is still a layering violation."""
    source = tmp_path / "training.py"
    source.write_text(
        "def task_spec(task_id):\n"
        "    from i4h_arena.scenes.base import scene_specs\n"
        "    return scene_specs()[task_id]\n"
    )
    _, function_level = collect_imports(source)
    assert _violations(function_level, frozenset({"i4h_arena"})) == [
        "i4h_arena.scenes.base"
    ], "a lazy import that crosses a layer must be reported, collapsed to the module"


def test_checker_ignores_type_checking_blocks(tmp_path):
    source = tmp_path / "annotated.py"
    source.write_text(
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from i4h_arena.scenes.base import Scene\n"
    )
    module_level, function_level = collect_imports(source)
    assert not _violations(module_level | function_level, frozenset({"i4h_arena"}))


def test_internal_imports_use_i4h_namespaces():
    """Generic top-level package names are retired; every i4h import is explicit."""
    retired = {"common", "arena", "tasks", "tools"}
    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if not ours(path):
            continue
        module_level, function_level = collect_imports(path)
        stale = sorted(module for module in module_level | function_level if module.split(".", 1)[0] in retired)
        if stale:
            offenders.append(f"{path.relative_to(ROOT)} imports {stale}")
    assert not offenders, "retired generic import namespaces:\n  " + "\n  ".join(offenders)


def test_matcher_respects_submodule_boundaries():
    assert _matches("i4h_tasks.gr00t_n15", "i4h_tasks.gr00t_n15")
    assert _matches("i4h_tasks.gr00t_n15.server", "i4h_tasks.gr00t_n15")
    assert not _matches("i4h_tasks.basic", "i4h_tasks.gr00t_n15")
    # A prefix must be a whole path segment, not a string prefix.
    assert not _matches("i4h_tasks_extra", "i4h_tasks")
    assert _matches("i4h_tasks.basic.motion", "i4h_tasks")


# -- manifest and project integrity --------------------------------------


def test_init_files_contain_no_code():
    """``__init__.py`` carries the licence header and nothing else.

    A re-export shim gives every symbol two truthful import paths, and the
    aliased one hides where the code actually lives — ``from i4h_arena.policies import
    HoldPolicy`` says nothing about `policies/hold.py`. Without shims every
    import states its own target, and there is no export list to fall out of
    date with the modules beneath it.
    """
    offenders: list[str] = []
    for init in sorted(p for p in ROOT.rglob("__init__.py") if ours(p)):
        tree = ast.parse(init.read_text())
        # A module docstring is the only statement allowed to survive, and even
        # that is discouraged; anything executable is not.
        code = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            )
        ]
        if code:
            kinds = sorted({type(n).__name__ for n in code})
            offenders.append(f"{init.relative_to(ROOT)} contains {kinds}")
    assert not offenders, "__init__.py must hold no code:\n  " + "\n  ".join(offenders)


def test_nothing_imports_from_a_bare_package():
    """Imports must name the module that defines the symbol.

    With empty ``__init__.py`` files, ``from i4h_arena.scenes import Scene`` would
    fail, and only at run time. Catching it statically keeps the failure at review.
    """
    # import path -> directory on disk, so we can tell a submodule from a symbol.
    package_dirs: dict[str, Path] = {}
    package_roots = {
        "i4h_common",
        "i4h_arena",
        "i4h_tasks",
        "i4h_tools",
        "i4h_rl",
    }
    for init in ROOT.rglob("__init__.py"):
        if not ours(init):
            continue
        parts = init.parent.parts
        for i, part in enumerate(parts):
            if part not in package_roots:
                continue
            package_dirs[".".join(parts[i:])] = init.parent
            break

    def is_submodule(package: str, name: str) -> bool:
        directory = package_dirs[package]
        return (directory / f"{name}.py").is_file() or (directory / name / "__init__.py").is_file()

    offenders: list[str] = []
    for path in ROOT.rglob("*.py"):
        if not ours(path) or path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in package_dirs):
                continue
            # `from pkg import submodule` is fine — that is a real module, not a
            # shim re-export. Only bare symbols would have needed an __init__.
            symbols = [a.name for a in node.names if not is_submodule(node.module, a.name)]
            if symbols:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} imports {', '.join(symbols)} from package "
                    f"{node.module!r} — name the module that defines it"
                )
    assert not offenders, "\n  ".join(offenders)


def test_no_namespace_package_init_files():
    """An `i4h_tasks/__init__.py` or `i4h_tools/__init__.py` breaks sibling projects."""
    offenders: list[Path] = []
    for project in ("tasks", "tools"):
        top = ROOT / project / "__init__.py"
        if top.exists():
            offenders.append(top.relative_to(ROOT))
        for path in (ROOT / project).glob(f"*/i4h_{project}/__init__.py"):
            offenders.append(path.relative_to(ROOT))
    assert not offenders, f"PEP 420 namespaces must have no __init__.py: {offenders}"


SCENE_MANIFEST = ROOT / "arena" / "i4h_arena" / "scenes" / "manifest"


def _scenes() -> list[tuple[str, dict]]:
    out = []
    for path in sorted(SCENE_MANIFEST.glob("*.yaml")):
        with path.open("rb") as handle:
            out.append((path.stem, yaml.safe_load(handle)))
    return out


def test_scene_manifest_parses():
    scenes = _scenes()
    assert scenes, f"no scene manifests under {SCENE_MANIFEST}"
    for name, scene in scenes:
        for required in ("impl", "embodiment", "action_space", "dof"):
            assert required in scene, f"scene {name} is missing {required}"


def test_projects_are_independent_uv_projects():
    """No workspace: each project resolves its own lock (DESIGN.md §12)."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    assert "workspace" not in document.get("tool", {}).get(
        "uv", {}
    ), "the repository must not be a uv workspace — arena's Isaac pins would be forced onto every light project"


def test_every_project_pins_its_siblings_by_path():
    for pyproject in sorted(ROOT.glob("*/pyproject.toml")) + sorted(ROOT.glob("*/*/pyproject.toml")):
        if not ours(pyproject) or pyproject.parent == ROOT:
            continue
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
        sources = document.get("tool", {}).get("uv", {}).get("sources", {})
        for dependency in document.get("project", {}).get("dependencies", []):
            name = dependency.split(">")[0].split("=")[0].split("[")[0].strip()
            if name.startswith("i4h-"):
                assert name in sources, f"{pyproject.relative_to(ROOT)}: {name} has no path source"


def test_policy_stacks_declare_only_what_they_import():
    """A backend must not install simulator source it does not import."""
    exempt = {"gr00t_n16"}  # uses Arena's G1 WBC joint remapping
    offenders = []
    for pyproject in sorted((ROOT / "tasks").glob("*/pyproject.toml")):
        stack = pyproject.parent.name
        if stack in exempt:
            continue
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
        declared = " ".join(document.get("project", {}).get("dependencies", []))
        if "isaaclab" in declared:
            offenders.append(f"{stack} declares an isaaclab dependency but only serves a policy")
    assert not offenders, "\n  ".join(offenders)


def test_arena_does_not_depend_on_a_policy_stack():
    """Policy stacks stay out of arena because their torch pins conflict."""
    with (ROOT / "arena" / "pyproject.toml").open("rb") as handle:
        document = tomllib.load(handle)
    declared = " ".join(document.get("project", {}).get("dependencies", []))
    for stack in ("gr00t-n15", "gr00t-n16", "gr00t-n17", "openpi-pi0"):
        assert stack not in declared, (
            f"arena declares a dependency on {stack}; their torch pins conflict with Isaac's, "
            f"which is why each one is served from its own environment"
        )


ROBOT_MANIFEST = ROOT / "arena" / "i4h_arena" / "embodiments" / "manifest"


def test_every_scene_embodiment_has_a_descriptor():
    scenes = [dict(body, name=name) for name, body in _scenes()]
    missing = sorted({s["embodiment"] for s in scenes if not (ROBOT_MANIFEST / f"{s['embodiment']}.yaml").is_file()})
    assert not missing, f"scenes name robots with no descriptor in {ROBOT_MANIFEST}: {missing}"


def test_robot_descriptors_do_not_restate_scene_facts():
    """dof / action_space / gripper belong to the scene, not the robot.

    The same arm mounted under a different controller has a different action
    width. Stating it in both places produced only an agreement test, which is
    the smell that one of the two should not exist.
    """
    # control_hz too: the same arm simulated at a different decimation has a
    # different rate, so it belongs with the step budget it converts.
    owned_by_scene = {"dof", "action_space", "gripper", "control_hz"}
    offenders = []
    for path in sorted(ROBOT_MANIFEST.glob("*.yaml")):
        with path.open("rb") as handle:
            document = yaml.safe_load(handle) or {}
        restated = sorted(owned_by_scene & document.keys())
        if restated:
            offenders.append(f"{path.relative_to(ROOT)} restates {restated}")
    assert not offenders, chr(10).join(offenders)


def test_arena_module_names_share_one_vocabulary():
    """`scenes/`, `assets/` and `envcfg/` name the same worlds the same way.

    Each holds a different facet of one scene — the Scene class, its USD props,
    its IsaacLab env cfg — so a file in one should be findable by name in the
    others. The port arrived with env-era names (`ultrasound_liver_scan.py`
    beside `panda_phantom.py`), which made the three directories read as
    unrelated trees.

    Shared bases carry a leading underscore in every directory, and not every
    scene needs its own asset or cfg module; what is checked is that a name,
    where it exists, matches a scene.
    """

    def stems(directory: str) -> set[str]:
        base = ROOT / "arena" / "i4h_arena" / directory
        shared_modules = {"__init__", "base", "constants", "_base"}
        helper_modules = {
            # The fluoroscopy module names the reusable C-arm/catheter asset
            # technology, not the renamed endoluminal Scene that consumes it.
            "assets": {"authoring_catalog", "config_asset", "contact", "fluoroscopy_catheter_navigation"},
        }
        return {p.stem for p in base.glob("*.py")} - shared_modules - helper_modules.get(directory, set())

    scenes = stems("scenes")
    stray = {}
    for directory in ("assets", "envcfg"):
        unknown = stems(directory) - scenes
        if unknown:
            stray[directory] = sorted(unknown)
    assert not stray, f"module names not matching any scene: {stray}. Scenes are {sorted(scenes)}."


def test_scene_manifest_names_match_scene_modules():
    """Every `[[scene]]` entry points at a module named after it."""
    scenes = [dict(body, name=name) for name, body in _scenes()]
    mismatches = []
    for scene in scenes:
        module = scene["impl"].partition(":")[0]
        expected = f"i4h_arena.scenes.{scene['name']}"
        if module != expected:
            mismatches.append(f"{scene['name']}: impl is {module}, expected {expected}")
    assert not mismatches, "\n  ".join(mismatches)


def test_scene_classes_call_env_cfgs_with_valid_keywords():
    """A scene must not pass a keyword its env cfg does not accept.

    This is the bug class the arena review turned up: `PandaPhantomEnvCfg` takes
    no `env_spacing`, and the scene passed one. Nothing catches that until Kit
    has booted and the scene is being constructed — several minutes in, with a
    `TypeError` that looks like an Isaac problem.

    Static and approximate on purpose: it only checks calls whose callee is a
    class imported from `i4h_arena.envcfg`, and only flags keywords absent from that
    class's `__init__` (following one level of inheritance within the module).
    """
    envcfg_dir = ROOT / "arena" / "i4h_arena" / "envcfg"

    def init_keywords(module_file: Path, class_name: str, depth: int = 0) -> set[str] | None:
        if depth > 3 or not module_file.is_file():
            return None
        tree = ast.parse(module_file.read_text())
        node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if node is None:
            return None
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            # Inherited __init__: follow the first base defined in the same module.
            for base in node.bases:
                base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if base_name:
                    inherited = init_keywords(module_file, base_name, depth + 1)
                    if inherited is not None:
                        return inherited
            return None
        if init.args.kwarg is not None:
            return None  # **kwargs accepts anything
        accepted = {a.arg for a in init.args.args + init.args.kwonlyargs} - {"self"}
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if base_name:
                inherited = init_keywords(module_file, base_name, depth + 1)
                if inherited:
                    accepted |= inherited
        return accepted

    offenders: list[str] = []
    for path in sorted((ROOT / "arena" / "i4h_arena" / "scenes").glob("*.py")):
        tree = ast.parse(path.read_text())
        # class name -> the envcfg module it was imported from
        origin: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("i4h_arena.envcfg."):
                for alias in node.names:
                    origin[alias.asname or alias.name] = node.module.rpartition(".")[2]

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            module_stem = origin.get(node.func.id)
            if module_stem is None:
                continue
            accepted = init_keywords(envcfg_dir / f"{module_stem}.py", node.func.id)
            if accepted is None:
                continue
            passed = {kw.arg for kw in node.keywords if kw.arg}
            unknown = sorted(passed - accepted)
            if unknown:
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} calls {node.func.id}({', '.join(unknown)}=...) "
                    f"but it accepts {sorted(accepted)}"
                )
    assert not offenders, "scene/env-cfg signature mismatch:\n  " + "\n  ".join(offenders)


REMOTE_STACKS = ("gr00t_n15", "gr00t_n16", "gr00t_n17", "openpi_pi0")


def _manifests():
    """Every task manifest — discovery is one glob, so this is too."""
    return sorted((ROOT / "tasks").glob("*/i4h_tasks/*/manifest/*.yaml"))


def test_every_policy_stack_declares_its_tasks():
    stacks = {p.parent.parent.name for p in _manifests()}
    assert stacks >= set(REMOTE_STACKS), f"missing manifests for {sorted(set(REMOTE_STACKS) - stacks)}"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_manifest_names_an_embodiment_and_avoids_base_models(path):
    """A policy task must say what it drives, and must not present a base
    foundation model as a trained checkpoint."""
    BASE_MODELS = {"nvidia/GR00T-N1.5-3B", "nvidia/GR00T-N1.6-3B", "nvidia/GR00T-N1.7-3B"}
    with path.open("rb") as handle:
        doc = yaml.safe_load(handle) or {}
    assert doc.get("embodiment"), f"{path.stem} names no embodiment"
    assert (doc.get("model") or {}).get(
        "repo"
    ) not in BASE_MODELS, f"{path.stem} serves a base model, not a task checkpoint"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_every_task_manifest_has_summary_and_nonduplicate_prompt(path):
    """Every task has one catalog description; prompt exists only when richer."""
    with path.open("rb") as handle:
        doc = yaml.safe_load(handle) or {}
    summary = str(doc.get("summary", "")).strip()
    prompt = str(doc.get("prompt", "")).strip()
    assert summary, f"{path.stem}: missing summary"
    assert not prompt or prompt.casefold() != summary.casefold(), f"{path.stem}: prompt duplicates summary"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_every_task_manifest_loads(path):
    """What each stack's serve and train modules read must parse."""
    from i4h_common.taskdef import load_taskdef

    assert load_taskdef(path).summary
