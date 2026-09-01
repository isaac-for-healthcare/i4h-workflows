# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Scene contract.

A scene is a world: assets, embodiment, cameras, and randomization. Workflows and
tasks own goals and run modes. Each scene has a YAML manifest declaring what it
provides; the layering tests keep that declaration aligned with its Python
implementation.
"""

from __future__ import annotations

import argparse
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from i4h_arena.adapters.actuation import ArenaActuation, RobotSlice
from i4h_arena.adapters.scene_view import ArenaSceneView
from i4h_common.manifest import SceneSpec


@dataclass(frozen=True, slots=True)
class SensorSliderSpec:
    """Workflow-owned live control shown beside a docked sensor image."""

    label: str
    control: str
    minimum: float
    maximum: float
    step: float
    default: float
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class SensorDisplayControlSpec:
    """Sensor-owned control over how a rendered frame becomes pixels.

    Distinct from :class:`SensorSliderSpec`, which drives the workflow: these reach the sensor
    and re-map an existing frame instead of changing what the simulation does.
    """

    label: str
    control: str
    minimum: float
    maximum: float
    step: float
    default: float


class Scene(ABC):
    """One simulated world."""

    #: Must match a scene manifest filename.
    name: str = ""

    def __init__(self, spec: SceneSpec, args: argparse.Namespace) -> None:
        self.spec = spec
        self.args = args

    # -- construction ----------------------------------------------------
    def register_assets(self) -> None:
        """Side-effect imports for ``@register_asset`` declarations. Override as needed."""

    @abstractmethod
    def build(self) -> Any:
        """Return an ``IsaacLabArenaEnvironment`` for this world."""

    def gym_spec(self) -> tuple[str, Any]:
        """Resolve ``(gym_env_id, env_cfg)`` for :func:`gymnasium.make`."""
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        print("[arena] constructing arena environment", flush=True)
        arena_env = self.build()
        print("[arena] registering with gym", flush=True)
        gym_id, env_cfg = ArenaEnvBuilder(arena_env, self.args).build_registered()
        self.configure_env_cfg(env_cfg)
        return gym_id, env_cfg

    def configure_env_cfg(self, env_cfg: Any) -> None:
        """Apply workflow-specific simulator presentation settings after Arena builds the cfg."""

    def configure_args(self, args: argparse.Namespace) -> None:
        """Adjust ``args`` before ``AppLauncher`` starts (e.g. enable cameras)."""
        if not getattr(args, "no_cameras", False) and self.spec.cameras:
            args.enable_cameras = True
        if self.spec.action_space == "ee_pose":
            args.action_device = "ik_abs"

    # -- adapters --------------------------------------------------------
    def home_joints(self, env: Any) -> np.ndarray | None:
        """Home pose for this scene's robot, or ``None`` to use IsaacLab's default."""
        return None

    def robot_slices(self, env: Any) -> tuple[RobotSlice, ...]:
        """Action-vector layout. Single-robot scenes get the whole vector."""
        width = int(env.action_space.shape[-1])
        return (RobotSlice("robot", 0, width, gripper_index=width - 1 if self.spec.gripper else None),)

    def tcp_body(self) -> str | None:
        """Body name to read as the tool frame; ``None`` means the last link."""
        return None

    def object_aliases(self) -> dict[str, str]:
        return {}

    def footprint_half_extents(self) -> dict[str, tuple[float, float]]:
        """Horizontal collision footprints owned by this Scene."""
        return {}

    def command_objects(self) -> dict[str, tuple[str, str]]:
        return {}

    def robot_assets(self) -> dict[str, str]:
        return {}

    def joint_orders(self) -> dict[str, tuple[str, ...]]:
        """Policy-facing joint order for each robot, when it differs from USD."""
        return {}

    def tcp_sensors(self) -> dict[str, str]:
        return {}

    def camera_aliases(self) -> dict[str, str]:
        return {}

    def default_sensor_views(self) -> tuple[str, ...]:
        """Camera-compatible sensors shown in docked windows for visible runs."""
        return ()

    def sensor_view_titles(self) -> dict[str, str]:
        """Operator-facing titles for docked sensor windows."""
        return {}

    def sensor_view_outputs(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Presentation outputs offered by each live sensor window."""
        return {}

    def sensor_view_keyboard_toggles(self) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
        """Keyboard toggles between presentation outputs for each sensor window."""
        return {}

    def sensor_view_projection_presets(self) -> dict[str, tuple[tuple[str, str, float], ...]]:
        """Named projection choices as ``(label, key, angle_radians)`` tuples."""
        return {}

    def sensor_view_projection_defaults(self) -> dict[str, int]:
        """Initial combo-box index for each sensor's projection choices."""
        return {}

    def sensor_view_appearances(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Selectable display looks as ``(label, appearance)`` tuples, first one being default."""
        return {}

    def sensor_view_display_controls(self) -> dict[str, tuple[SensorDisplayControlSpec, ...]]:
        """Live display-mapping controls offered beside each sensor image."""
        return {}

    def sensor_view_sliders(self) -> dict[str, tuple[SensorSliderSpec, ...]]:
        """Live workflow controls displayed beside each sensor image."""
        return {}

    def relative_ee(self) -> bool:
        return False

    def make_view(self, env: Any) -> ArenaSceneView:
        return ArenaSceneView(
            env,
            objects=self.spec.objects,
            robots=self.spec.robots,
            cameras=self.spec.cameras,
            home_joints=self.home_joints(env),
            tcp_body=self.tcp_body(),
            gripper=self.spec.gripper,
            object_aliases=self.object_aliases(),
            footprint_half_extents=self.footprint_half_extents(),
            command_objects=self.command_objects(),
            robot_assets=self.robot_assets(),
            joint_orders=self.joint_orders(),
            tcp_sensors=self.tcp_sensors(),
            camera_aliases=self.camera_aliases(),
            root_relative=self.spec.action_space == "ee_pose",
        )

    def make_actuation(self, env: Any, view: ArenaSceneView | None = None) -> ArenaActuation:
        width = int(env.action_space.shape[-1])
        actuation = ArenaActuation(
            num_envs=int(env.unwrapped.num_envs),
            action_dim=width,
            action_space=self.spec.action_space,
            device=str(getattr(self.args, "device", "cpu")),
            slices=self.robot_slices(env),
            view=view,
            relative_ee=self.relative_ee(),
        )
        # Seed the buffer so the very first tick commands a sane pose rather than
        # all-zeros, which on a floating-base humanoid means collapse.
        #
        # Read it through the view, not `self.home_joints()`: the view falls back
        # to the articulation's `default_joint_pos`, which IsaacLab sets from the
        # embodiment cfg. That is where six of seven robots actually keep their
        # home pose — relying on a descriptor override meant only `so101` was
        # ever seeded, and the rest silently started at zero.
        if self.spec.action_space == "joint_position" and width > 0:
            home = (view or self.make_view(env)).home_joints()
            if home is not None:
                robot_slices = self.robot_slices(env)
                if home.shape[-1] == width:
                    actuation.seed(home)
                elif len(robot_slices) == 1 and robot_slices[0].joint_width == home.shape[-1]:
                    initial = np.zeros((int(env.unwrapped.num_envs), width), dtype=np.float32)
                    robot_slice = robot_slices[0]
                    initial[:, robot_slice.start : robot_slice.start + home.shape[-1]] = home
                    actuation.seed(initial)
        elif self.spec.action_space == "ee_pose" and not self.relative_ee():
            scene_view = view or self.make_view(env)
            initial = np.zeros((int(env.unwrapped.num_envs), width), dtype=np.float32)
            for robot_slice in self.robot_slices(env):
                tcp = scene_view.tcp(robot_slice.name)
                quat_xyzw = tcp.quat[:, [1, 2, 3, 0]]
                pose = np.concatenate([tcp.pos, quat_xyzw], axis=-1)
                pose_width = min(7, robot_slice.width)
                initial[:, robot_slice.start : robot_slice.start + pose_width] = pose[:, :pose_width]
                if robot_slice.gripper_index is not None:
                    initial[:, robot_slice.start + robot_slice.gripper_index] = 1.0
            actuation.seed(initial)
        return actuation

    # -- episode hooks ---------------------------------------------------
    def on_reset(self, env: Any, view: ArenaSceneView) -> None:
        """After ``env.reset``: snapshot randomization, pre-roll to a start pose."""

    def describe(self) -> str:
        return self.spec.description or self.name


def scene_specs() -> dict[str, SceneSpec]:
    """Every scene declared in the manifest. No imports."""
    from i4h_engine.registry import default_registry

    return dict(default_registry().scenes)


def load_scene(name: str, args: argparse.Namespace) -> Scene:
    """Import and construct a scene by manifest name.

    This is the only place ``arena`` imports a scene class, and it happens after
    the manifest has already been validated against the workflow.
    """
    specs = scene_specs()
    spec = specs.get(name)
    if spec is None:
        import difflib

        close = difflib.get_close_matches(name, specs, n=3, cutoff=0.5)
        hint = f"; did you mean {close}?" if close else f"; known scenes: {sorted(specs)}"
        raise KeyError(f"unknown scene {name!r}{hint}")

    spec = spec.for_mode(getattr(args, "mode", None))
    module_name, _, attr = spec.impl.partition(":")
    if not attr:
        raise ValueError(f"scene {name}: impl must be 'module:Class', got {spec.impl!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr, None)
    if cls is None:
        raise AttributeError(f"scene {name}: {module_name} has no attribute {attr!r} (from {spec.source})")
    scene = cls(spec, args)
    scene.register_assets()
    return scene
