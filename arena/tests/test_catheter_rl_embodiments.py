# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the camera-free RSL-RL catheter observations.

The task and embodiment modules pull in ``isaaclab.sim`` and Arena and cannot
be imported without Isaac Sim, so these read the source the way
``test_franka_catheter_embodiment`` does.

Two contracts are worth pinning. The first is where the flat variant comes
from: the navigation observation group belongs to
``CatheterNavigationTask``, and its ``flat_observations`` switch is the only
thing that drops the fluoroscopy view and concatenates the rest. If an
embodiment published an observation group of its own, the actor would be handed
a second, unconcatenated group. That failure is silent at import and only shows
up as a shape error once a trainer starts.

The second is that the two RSL-RL configs stay identical. The pair exists to
measure what the coupled MJWarp solver costs against the rod-only solver, and
that only works while every hyperparameter matches; a tweak to one of them
turns the comparison into noise without anything else complaining.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ARENA = Path(__file__).parents[1] / "i4h_arena"
REPO = Path(__file__).parents[2]

CATHETER_SOURCE = ARENA / "embodiments" / "catheter.py"
FRANKA_SOURCE = ARENA / "embodiments" / "franka_catheter.py"
TASK_SOURCE = ARENA / "envcfg" / "endoluminal_navigation" / "task.py"
SCENE_SOURCE = ARENA / "scenes" / "endoluminal_navigation.py"
ARM_SCENE_SOURCE = ARENA / "scenes" / "endoluminal_navigation_arm.py"

PLAIN_CONFIG = REPO / "rl" / "config" / "endoluminal_navigation_ppo_rsl_rl.yaml"
ARM_CONFIG = REPO / "rl" / "config" / "endoluminal_navigation_arm_ppo_rsl_rl.yaml"

#: The task that owns the navigation observations, and the gate both scenes branch on.
TASK = "CatheterNavigationTask"
GATE = "_wants_flat_rl_observations"


def _class_def(source: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {source.name}")


def _base_names(class_def: ast.ClassDef) -> list[str]:
    return [base.id for base in class_def.bases if isinstance(base, ast.Name)]


def _method_names(class_def: ast.ClassDef) -> list[str]:
    return [node.name for node in class_def.body if isinstance(node, ast.FunctionDef)]


def _scene_name(source: Path) -> str:
    return "EndoluminalNavigationArmScene" if "arm" in source.stem else "EndoluminalNavigationScene"


def _method(class_def: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{class_def.name} does not define {name}")


def test_the_flat_variant_drops_the_view_and_flattens_the_rest() -> None:
    """The switch must do both halves, since either alone leaves RSL-RL broken.

    Flattening without dropping the view asks the concatenation to fold an
    image in beside fifteen scalars; dropping without flattening hands back a
    dict the actor cannot read.
    """
    body = ast.unparse(_method(_class_def(TASK_SOURCE, TASK), "get_observation_cfg"))

    assert "if self.flat_observations:" in body
    assert "config.policy.fluoroscopy_rgb = None" in body
    assert "config.policy.concatenate_terms = True" in body
    # Without a twin the task returns None for the observation, reward and
    # termination alike, which keeps a twinless scene failing on the missing
    # twin rather than on an attribute error here.
    assert "return None" in body


@pytest.mark.parametrize(
    ("source", "embodiment"),
    [
        (CATHETER_SOURCE, "CatheterEmbodiment"),
        (FRANKA_SOURCE, "FrankaCatheterEmbodiment"),
    ],
)
def test_the_embodiments_publish_no_observation_of_their_own(source: Path, embodiment: str) -> None:
    """One task has to serve both scenes, or the comparison is not like-for-like.

    The arm is a positioner rather than a second agent: servo'd to hold the
    introducer, it adds seven recorded joint columns but no observation and no
    action channel. If that ever changes and an embodiment publishes its own
    group, this fires -- and the task, both RSL-RL configs' ``obs_groups`` and
    the 15-wide actor input all need revisiting together.
    """
    methods = _method_names(_class_def(source, embodiment))
    assert "get_observation_cfg" not in methods, (
        f"{embodiment} publishes an observation group beside {TASK}'s, so the "
        f"paired runs no longer share the task's fifteen columns."
    )
    assert "get_observation_cfg" in _method_names(_class_def(TASK_SOURCE, TASK))


def test_the_gate_is_defined_once_and_used_by_both_scenes() -> None:
    """Shared so the two scenes cannot disagree about what a training run is."""
    base_scene = _class_def(SCENE_SOURCE, "EndoluminalNavigationScene")
    assert GATE in _method_names(base_scene), f"{GATE} should live on the base scene"
    assert GATE in ast.unparse(_method(base_scene, "_make_task")), f"_make_task does not gate on {GATE}"
    assert "self._make_task(" in ast.unparse(_method(base_scene, "build"))

    arm_scene = _class_def(ARM_SCENE_SOURCE, "EndoluminalNavigationArmScene")
    assert GATE not in _method_names(arm_scene), "the arm scene should inherit the gate, not restate it"
    assert "_make_task" not in _method_names(arm_scene), "the arm scene should inherit the task, not restate it"


def test_the_stock_rsl_rl_entry_points_are_registered() -> None:
    """Isaac Lab's own scripts resolve ``--task`` through these two kwargs.

    Registered on the base scene's ``build``, which the arm scene inherits
    unchanged, so both Gym IDs carry them.
    """
    build = ast.unparse(_method(_class_def(SCENE_SOURCE, "EndoluminalNavigationScene"), "build"))

    assert "rl_framework_entry_point='rsl_rl_cfg_entry_point'" in build
    assert "i4h_arena.agents.rsl_rl:ProfiledRslRlRunnerCfg" in build
    assert "build" not in _method_names(
        _class_def(ARM_SCENE_SOURCE, "EndoluminalNavigationArmScene")
    ), "the arm scene overrides build, so it no longer inherits the entry points"


def test_the_paired_configs_differ_only_in_experiment_name() -> None:
    """Identical hyperparameters are what make the two runs a measurement."""
    plain = yaml.safe_load(PLAIN_CONFIG.read_text())
    arm = yaml.safe_load(ARM_CONFIG.read_text())

    assert plain["runner"].pop("experiment_name") == "endoluminal_navigation_rsl"
    assert arm["runner"].pop("experiment_name") == "endoluminal_navigation_arm_rsl"
    assert plain == arm, (
        "the rod-only and coupled-solver configs have drifted apart; a gap "
        "between their per-term rewards is no longer attributable to the solver."
    )


@pytest.mark.parametrize(
    ("source", "embodiment"),
    [(CATHETER_SOURCE, "CatheterEmbodiment"), (FRANKA_SOURCE, "FrankaCatheterEmbodiment")],
)
def test_both_catheter_embodiments_pin_the_env_spacing_to_zero(source: Path, embodiment: str) -> None:
    """Coincident rods make a positive spacing a bug rather than a preference.

    ``add_catheter_rod_to_builder`` adds the same absolute positions
    ``num_envs`` times, so the rods sit on each other whatever the spacing
    says. A non-zero spacing then leaves the environment origins on a grid
    under coincident rods, and ``tip_position`` -- the one observation reported
    relative to its origin -- returns a per-environment constant of metres for
    one identical physical state, in a channel whose real range is centimetres.

    Pinned for both embodiments because the Franka variant zeroed it for the
    arm's sake and the plain one inherited a positive default, which is how the
    two came to disagree. The RL profile cannot express zero at all:
    ``i4h_rl.profile`` rejects a non-positive ``env_spacing``, so this override
    is the only place it can happen.
    """
    body = ast.unparse(_method(_class_def(source, embodiment), "modify_env_cfg"))

    assert "env_cfg.scene.env_spacing = 0.0" in body
