# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class for agentic Arena environments."""

from __future__ import annotations

import argparse
import importlib
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any

from common.config import get_env_metadata


class AgenticEnvironmentBase(ABC):
    """Common contract for every agentic environment."""

    name: str | None = None
    description: str = ""

    @abstractmethod
    def get_env(self, args: argparse.Namespace) -> Any:
        """Build and return an ``IsaacLabArenaEnvironment``.

        Envs that already have a fully-formed gym task (e.g. ported tasks
        registered via :func:`gymnasium.register`) can override :meth:`build`
        instead and leave :meth:`get_env` raising ``NotImplementedError`` —
        the default :meth:`build` is the only consumer.
        """

    def build(self, args: argparse.Namespace) -> tuple[str, Any]:
        """Resolve ``(gym_env_id, env_cfg)`` for :func:`gymnasium.make`.

        Default implementation composes the result of :meth:`get_env` via
        :class:`isaaclab_arena.environments.arena_env_builder.ArenaEnvBuilder`.
        Override for envs that build their own ``env_cfg`` directly.
        """
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        arena_env = self.get_env(args)
        return ArenaEnvBuilder(arena_env, args).build_registered()

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add env-specific CLI arguments. Override as needed."""

    def configure_args(self, args: argparse.Namespace) -> None:
        """Tweak ``args`` before ``AppLauncher`` starts (e.g. enable cameras)."""
        if _is_zero_action_run(args):
            args.disable_cameras = False
            args.enable_cameras = True
        elif not getattr(args, "disable_cameras", False):
            args.enable_cameras = True

    def register_assets(self) -> None:
        """Side-effect imports for ``@register_asset`` registrations.

        Default no-op. Override to import background/object/embodiment modules
        that should be present in :class:`isaaclab_arena.assets.AssetRegistry`
        before :meth:`get_env` runs.
        """

    @abstractmethod
    def run(
        self,
        args: argparse.Namespace,
        env: Any,
        app: Any,
        controller: Any,
    ) -> None:
        """Run the simulation loop (policy episodes, teleop, replay, …)."""

    def build_idle_action(self, args: argparse.Namespace, env: Any, obs: Any) -> Any:
        """Return a per-step hold-pose action for the bridge keep-open loop.

        Default returns ``None`` (no action applied; the keep-open loop just
        renders). Floating-base humanoid envs override this to seed a 23D
        WBC command so the robot does not collapse in edit mode.
        """
        return None

    def import_runtime_module(self) -> ModuleType:
        """Import this env's Arena runtime module."""
        if not self.name:
            raise ValueError(f"{type(self).__name__} must define a non-empty name")
        return importlib.import_module(get_env_metadata(str(self.name)).runtime_module)


def _is_zero_action_run(args: argparse.Namespace) -> bool:
    return (
        not getattr(args, "teleop", False)
        and not getattr(args, "replay_dataset_path", None)
        and getattr(args, "episodes", 0) <= 0
    )


def policy_io_factory(runtime: ModuleType) -> type:
    from arena.runtimes.core.base import PolicyIO

    factories = [
        value
        for value in vars(runtime).values()
        if isinstance(value, type) and issubclass(value, PolicyIO) and value is not PolicyIO
    ]
    if len(factories) != 1:
        raise RuntimeError(f"expected exactly one PolicyIO subclass in {runtime.__name__}, found {len(factories)}")
    return factories[0]
