# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for the dVRK / STAR surgical scenes.

Six scenes sit on the same ``Props/Table/table.usd`` and differ by one swapped
object and which arm is mounted.

``star_reach`` is the exception: it mounts a ``SeattleLabTable``, so its asset
mode selects a different surface rather than inheriting the default.

Two vocabularies meet here and are deliberately kept apart:

* ``asset_mode`` selects props in :func:`i4h_arena.assets._surgical.make_assets`
  (``reach_psm``, ``lift_needle_organs``, …).
* the env cfg is chosen by *kind* — a reach scene calls one of
  :class:`SurgicalReachEnvCfg`'s classmethod factories, a lift scene builds
  :class:`SurgicalLiftEnvCfg` with an ``organs`` flag.

Collapsing them into one string is how a scene ends up with the right props and
the wrong reward terms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config


class SurgicalScene(Scene):
    """A table, one manipulandum, and one surgical arm."""

    #: Mode string for i4h_arena.assets._surgical.make_assets.
    asset_mode: str = "reach_psm"
    sim_dt: float = 1.0 / 60.0
    sim_decimation: int = 2
    render_interval: int = 2

    def register_assets(self) -> None:
        import i4h_arena.assets._surgical  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415

        from i4h_arena.assets._surgical import make_assets  # noqa: PLC0415

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self._embodiment(),
            scene=ArenaScene(assets=make_assets(self.asset_mode)),
            task=self.envcfg(),
        )

    def _embodiment(self) -> Any:
        """Instantiate this scene's arm with its contact-sensitive simulation settings."""
        enable_cameras = bool(getattr(self.args, "enable_cameras", True))
        action_device = str(getattr(self.args, "action_device", "joint_position"))
        shared = {
            "enable_cameras": enable_cameras,
            "action_device": action_device,
            "sim_dt": self.sim_dt,
            "sim_decimation": self.sim_decimation,
            "render_interval": self.render_interval,
            "enable_material_randomization": False,
        }

        if self.spec.embodiment == "star":
            from i4h_arena.embodiments.star import StarEmbodiment  # noqa: PLC0415

            return StarEmbodiment(**shared)
        if self.spec.embodiment == "dvrk_dual_psm":
            from i4h_arena.embodiments.psm import DualPsmEmbodiment  # noqa: PLC0415

            return DualPsmEmbodiment(**shared)

        from i4h_arena.embodiments.psm import PsmEmbodiment  # noqa: PLC0415

        # Original reach scenes expose only the seven-value Cartesian command;
        # lift scenes append the binary jaw action. The block uses the slightly
        # wider close pose from its source task.
        shared["include_gripper_action"] = self.asset_mode != "reach_psm"
        shared["gripper_close"] = 0.1 if self.asset_mode == "lift_block" else 0.09
        return PsmEmbodiment(**shared)

    def envcfg(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must supply an env cfg")

    def home_joints(self, env: Any) -> np.ndarray | None:
        home = get_robot_config(self.spec.embodiment).home_joint_pos_rad
        if not home:
            return None
        return np.tile(np.asarray(home, dtype=np.float32), (int(env.unwrapped.num_envs), 1))

    def tcp_body(self) -> str | None:
        return "endo360_needle" if self.spec.embodiment == "star" else "psm_tool_tip_link"

    def tcp_sensors(self) -> dict[str, str]:
        return {"robot": "ee_frame"}

    def object_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        if self.asset_mode == "lift_block":
            aliases["block"] = "object"
        elif self.asset_mode in ("lift_needle", "lift_needle_organs"):
            aliases["needle"] = "object"
        if self.asset_mode == "reach_star":
            aliases["star_table"] = "table"
        return aliases


class SurgicalReachScene(SurgicalScene):
    """Servo the tool tip through sampled reach poses."""

    #: Which SurgicalReachEnvCfg factory to call: psm | dual_psm | star.
    reach_mode: str = "psm"

    def envcfg(self) -> Any:
        from i4h_arena.envcfg._surgical import SurgicalReachEnvCfg  # noqa: PLC0415

        factory = getattr(SurgicalReachEnvCfg, self.reach_mode, None)
        if factory is None:
            raise ValueError(
                f"{type(self).__name__}: SurgicalReachEnvCfg has no {self.reach_mode!r} factory; "
                f"expected one of psm, dual_psm, star"
            )
        return factory()

    def command_objects(self) -> dict[str, tuple[str, str]]:
        if self.reach_mode == "dual_psm":
            return {
                "reach_target_1": ("ee_1_pose", "psm1"),
                "reach_target_2": ("ee_2_pose", "psm2"),
            }
        return {"reach_target": ("ee_pose", "robot")}


class SurgicalLiftScene(SurgicalScene):
    """Grasp and lift one object off the table."""

    # Match robotic_surgery's contact-sensitive lift integration settings:
    # 200 Hz physics with four substeps per 50 Hz control tick.
    sim_dt: float = 1.0 / 200.0
    sim_decimation: int = 4
    render_interval: int = 4
    task_description: str = "Lift the object with the dVRK PSM."
    #: Whether the manipulandum sits on an organ bed.
    organs: bool = False

    def envcfg(self) -> Any:
        from i4h_arena.envcfg._surgical import SurgicalLiftEnvCfg  # noqa: PLC0415

        return SurgicalLiftEnvCfg(
            task_description=self.task_description,
            organs=self.organs,
            env_spacing=float(getattr(self.args, "env_spacing", 2.5)),
        )
