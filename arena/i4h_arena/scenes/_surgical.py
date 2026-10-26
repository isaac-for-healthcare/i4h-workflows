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

from i4h_arena.scenes.base import Scene


class SurgicalScene(Scene):
    """A table, one manipulandum, and one surgical arm."""

    #: Mode string for i4h_arena.assets._surgical.make_assets.
    asset_mode: str = "reach_psm"
    sim_dt: float = 1.0 / 60.0
    sim_decimation: int = 2
    render_interval: int = 2

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        parser.add_argument(
            "--action_device",
            choices=("ik_abs", "joint_position"),
            default="ik_abs",
            help="Command the arm by absolute tool-tip pose (default) or joint positions.",
        )

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

            return DualPsmEmbodiment(**shared, newton=self._newton())

        from i4h_arena.embodiments.psm import PsmEmbodiment  # noqa: PLC0415

        # Original reach scenes expose only the seven-value Cartesian command;
        # lift scenes append the binary jaw action. The block uses the slightly
        # wider close pose from its source task.
        shared["include_gripper_action"] = self.asset_mode != "reach_psm"
        shared["gripper_close"] = 0.1 if self.asset_mode == "lift_block" else 0.09
        if self._newton():
            from i4h_arena.embodiments import _psm_newton  # noqa: PLC0415

            shared["newton"] = True
            if self.asset_mode != "reach_psm":
                shared["ik_body_offset"] = (0.0, 0.0, _psm_newton.GRASP_DEPTH_M)
            if self.asset_mode in ("lift_needle", "lift_needle_organs"):
                shared["gripper_close"] = _psm_newton.NEEDLE_GRIPPER_CLOSE
        return PsmEmbodiment(**shared)

    def _newton(self) -> bool:
        """Whether this run uses the Newton (MuJoCo Warp) physics preset."""
        return getattr(self.args, "presets", None) == "newton"

    def configure_env_cfg(self, env_cfg: Any) -> None:
        super().configure_env_cfg(env_cfg)
        if not self._newton():
            return
        from i4h_arena.embodiments._psm_newton import compensate_gravity_if_disabled  # noqa: PLC0415

        manipulandum = getattr(env_cfg.scene, "object", None)
        if manipulandum is not None:
            compensate_gravity_if_disabled(manipulandum)
            if self.asset_mode != "reach_psm":
                from i4h_arena.newton_meshes import decompose_bodies  # noqa: PLC0415

                decompose_bodies("/Object")
        organs = getattr(env_cfg.scene, "organs", None)
        if organs is not None:
            from i4h_arena.newton_meshes import keep_static_meshes, spawn_without_unreadable_payloads  # noqa: PLC0415

            organs.spawn.func = spawn_without_unreadable_payloads(organs.spawn.func)
            keep_static_meshes()

    def envcfg(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must supply an env cfg")


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


class SurgicalLiftScene(SurgicalScene):
    """Grasp and lift one object off the table."""

    # Match robotic_surgery's contact-sensitive lift integration settings:
    # 200 Hz physics with four substeps per 50 Hz control tick.
    sim_dt: float = 1.0 / 200.0
    sim_decimation: int = 4
    render_interval: int = 4
    task_description: str = "Lift the object with the dVRK PSM."
    #: Object height in the PSM root frame that counts as lifted [m]. The root sits at
    #: world z = 0.15 m, so -0.085 is world z = 0.065 m.
    success_height: float = -0.085
    #: Whether the manipulandum sits on an organ bed.
    organs: bool = False

    def envcfg(self) -> Any:
        from i4h_arena.envcfg._surgical import SurgicalLiftEnvCfg  # noqa: PLC0415

        return SurgicalLiftEnvCfg(
            task_description=self.task_description,
            success_height=self.success_height,
            organs=self.organs,
            env_spacing=float(getattr(self.args, "env_spacing", 2.5)),
        )
