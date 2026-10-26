# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Scene contract.

A scene is a world: assets, embodiment, cameras, randomization, and the task whose
success ends an episode. ``i4h_arena.environments`` runs each scene as an
IsaacLab-Arena environment. Each scene has a YAML manifest declaring what it
provides; the layering tests keep that declaration aligned with its Python
implementation.
"""

from __future__ import annotations

import argparse
import functools
import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from i4h_arena.ui.sensor_view import SensorView
from i4h_common.manifest import SceneSpec, load_scene_manifest


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

    def arena_environment(self) -> Any:
        """This world as an Arena environment whose ``env_cfg_callback`` applies its tuning.

        Arena 0.3 applies one ``env_cfg_callback`` per environment, after the physics
        backend is resolved. It is composed here from the task's and the embodiment's
        ``tune_env_cfg`` hooks, in that order, followed by the simulation defaults i4h
        keeps from before Isaac Lab 3.0 (see :mod:`i4h_arena.simulation`) and the merged
        RTX settings for every camera renderer and for env startup (see
        :mod:`i4h_arena.rendering`).
        """
        from i4h_arena import rendering, simulation  # noqa: PLC0415

        arena_env = self.build()
        owners = (arena_env.task, arena_env.embodiment)
        tuners = [owner.tune_env_cfg for owner in owners if callable(getattr(owner, "tune_env_cfg", None))]
        rtx_settings = rendering.merge_rtx_settings(
            rendering.SCENE_DEFAULTS,
            *(getattr(owner, "rtx_settings", None) for owner in owners),
            self.scene_rtx_settings(),
        )
        previous = arena_env.env_cfg_callback

        def env_cfg_callback(env_cfg: Any) -> Any:
            if previous is not None:
                env_cfg = previous(env_cfg)
            for tune in tuners:
                env_cfg = tune(env_cfg)
            simulation.apply_simulation_defaults(env_cfg)
            rendering.apply_to_env_cfg(env_cfg, rtx_settings)
            return env_cfg

        arena_env.env_cfg_callback = env_cfg_callback
        return arena_env

    def gym_spec(self) -> tuple[str, Any]:
        """Resolve ``(gym_env_id, env_cfg)`` for :func:`gymnasium.make`."""
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        print("[arena] constructing arena environment", flush=True)
        arena_env = self.arena_environment()
        print("[arena] registering with gym", flush=True)
        gym_id, env_cfg, _env_kwargs = ArenaEnvBuilder(arena_env, arena_builder_cfg(self.args)).build_registered()
        self.configure_env_cfg(env_cfg)
        return gym_id, env_cfg

    def scene_rtx_settings(self) -> dict[str, Any]:
        """RTX settings this world adds on top of its task's and embodiment's. Override as needed."""
        return {}

    def configure_env_cfg(self, env_cfg: Any) -> None:
        """Apply workflow-specific simulator presentation settings after Arena builds the cfg."""

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Flags this scene reads from the parsed CLI, added to its Arena environment's parser."""

    def configure_args(self, args: argparse.Namespace) -> None:
        """Adjust ``args`` before ``AppLauncher`` starts (e.g. enable cameras)."""
        if getattr(args, "presets", None) is None:
            args.presets = "physx"
        if not getattr(args, "no_cameras", False) and self.spec.cameras:
            args.enable_cameras = True
        if self.spec.action_space == "ee_pose":
            args.action_device = "ik_abs"

    def default_sensor_views(self) -> tuple[str, ...]:
        """Camera-compatible sensors shown in docked windows for visible runs."""
        return ()

    def make_sensor_view(self, env: Any) -> SensorView:
        """What the docked windows read the sensors in :meth:`default_sensor_views` through."""
        return SensorView(env, self.default_sensor_views())

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

    def sensor_view_readouts(self, env: Any) -> dict[str, Callable[[], str]]:
        """Per-frame status line shown under each sensor image.

        Teleop is where this earns its place: the operator is driving toward a
        termination criterion they cannot otherwise see.
        """
        return {}


@functools.cache
def scene_specs() -> dict[str, SceneSpec]:
    """Every scene declared in ``scenes/manifest``. No imports."""
    return {spec.name: spec for spec in load_scene_manifest(Path(__file__).parent / "manifest")}


def arena_builder_cfg(args: argparse.Namespace) -> Any:
    """Arena's ``ArenaEnvBuilderCfg`` from the fields of the CLI namespace that it names."""
    import dataclasses  # noqa: PLC0415

    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg  # noqa: PLC0415

    names = {field.name for field in dataclasses.fields(ArenaEnvBuilderCfg)}
    values = {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}
    # Arena seeds the env cfg with 42 when told nothing. i4h runs are unseeded unless
    # ``--seed`` is given, so separate runs randomize differently, as they did on Arena 0.2.
    values["seed"] = getattr(args, "seed", None)
    return ArenaEnvBuilderCfg(**values)


def scene_class(name: str) -> type[Scene]:
    """Import a scene's class by manifest name."""
    specs = scene_specs()
    spec = specs.get(name)
    if spec is None:
        import difflib

        close = difflib.get_close_matches(name, specs, n=3, cutoff=0.5)
        hint = f"; did you mean {close}?" if close else f"; known scenes: {sorted(specs)}"
        raise KeyError(f"unknown scene {name!r}{hint}")
    module_name, _, attr = spec.impl.partition(":")
    if not attr:
        raise ValueError(f"scene {name}: impl must be 'module:Class', got {spec.impl!r}")
    module = importlib.import_module(module_name)
    cls = getattr(module, attr, None)
    if cls is None:
        raise AttributeError(f"scene {name}: {module_name} has no attribute {attr!r} (from {spec.source})")
    return cls


def load_scene(name: str, args: argparse.Namespace, *, register_assets: bool = True) -> Scene:
    """Import and construct a scene by manifest name."""
    cls = scene_class(name)
    scene = cls(scene_specs()[name], args)
    if register_assets:
        scene.register_assets()
    return scene
