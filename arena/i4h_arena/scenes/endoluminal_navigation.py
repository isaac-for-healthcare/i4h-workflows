# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-managed interactive catheter fluoroscopy scene."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from i4h_arena.scenes.base import Scene, SensorDisplayControlSpec, SensorSliderSpec
from i4h_arena.ui.sensor_view import SensorView


def resolve_fluoroscopy_backend(requested: str | None, patient_twin: str | None) -> str:
    """Select real DRR rendering when a patient twin is supplied."""
    return requested or ("slang" if patient_twin else "synthetic")


class EndoluminalNavigationScene(Scene):
    name = "endoluminal_navigation"

    _embodiment: Any | None = None

    def register_assets(self) -> None:
        import i4h_arena.assets.fluoroscopy_catheter_navigation  # noqa: F401

    @property
    def _wants_flat_rl_observations(self) -> bool:
        """Whether this run is an online RSL-RL trainer rather than a workflow mode.

        Gated rather than always on so that teleop, replay and the N1.7
        rollouts keep the named, image-bearing observation group their bridge
        reads. ``rl_observations`` is what the RSL-RL registration callback
        sets; ``rl_training_mode`` is Isaac Lab's own flag on the stock
        scripts, and either alone is enough.

        Read by the arm-borne subclass too, so the two scenes cannot drift into
        disagreeing about what counts as a training run.
        """
        return bool(getattr(self.args, "rl_observations", False)) or bool(getattr(self.args, "rl_training_mode", False))

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        parser.add_argument(
            "--patient_twin",
            default=None,
            help="Patient-twin YAML manifest: the vessel route, its goal and the CT the fluoroscopy renders.",
        )
        parser.add_argument(
            "--fluoro_backend",
            choices=("synthetic", "slang"),
            default=None,
            help="Fluoroscopy renderer (default: Slang with --patient_twin, otherwise synthetic).",
        )

    def configure_args(self, args: Any) -> None:
        """Select Newton: the catheter rod solver is a Newton solver whatever ``--presets`` says.

        Arena 0.3 resolves the backend from ``--presets`` and rejects an environment
        callback that swaps it, so the scene names Newton up front rather than letting
        the embodiment replace PhysX afterwards, as it did on Arena 0.2.
        """
        if getattr(args, "presets", None) not in (None, "newton"):
            print(
                f"[arena] {self.name} simulates the catheter with Newton; ignoring --presets {args.presets}", flush=True
            )
        args.presets = "newton"
        super().configure_args(args)

    def _make_embodiment(self) -> Any:
        """Build the embodiment this scene drives.

        Overridden by the arm-borne variant, which swaps in a drive carried on a
        robot flange and takes the scene onto the coupled MJWarp + rod solver.
        """
        from i4h_arena.embodiments.catheter import CatheterEmbodiment

        return CatheterEmbodiment(patient_twin_manifest=getattr(self.args, "patient_twin", None))

    def _make_task(self, embodiment: Any) -> Any:
        """Bind the navigation goal to the route the embodiment seeded its rod on.

        Online RSL-RL takes the camera-free variant, whose observation group is
        one flat vector. Both scenes share this, so the same fifteen columns,
        four action channels and reward describe the armless and the arm-borne
        run, and a gap between their training curves is attributable to the
        solver rather than to the objective.
        """
        from i4h_arena.envcfg.endoluminal_navigation.task import CatheterNavigationTask

        # The manifest's step cap, as time. No ``+ 1`` as the other scenes use: the
        # workflow runner strips the time-out and enforces its own cap, so the only
        # consumers are the RL trainers, which should truncate at the cap itself. The
        # division is exact at these numbers: 600 / 30 is 20.0, and Isaac Lab recovers
        # 600 from it, where 601 / 30 rounds up to 602. ``getattr`` because the
        # RSL-RL interop parser does not define the arena CLI's ``--episode-steps``.
        steps = getattr(self.args, "episode_steps", None) or self.spec.max_steps
        return CatheterNavigationTask.for_embodiment(
            embodiment,
            episode_length_s=steps / self.spec.control_hz,
            flat_observations=self._wants_flat_rl_observations,
        )

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene
        from isaaclab_arena.utils.physics_backend import PhysicsBackend

        from i4h_arena.assets.fluoroscopy_catheter_navigation import make_assets

        self._embodiment = self._make_embodiment()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self._embodiment,
            scene=ArenaScene(
                assets=make_assets(
                    fluoro_backend=resolve_fluoroscopy_backend(
                        getattr(self.args, "fluoro_backend", None), getattr(self.args, "patient_twin", None)
                    ),
                    # Slang on CUDA has to be imported before Kit starts, which Arena's and
                    # Isaac Lab's runners do not allow for; their runs render on Vulkan.
                    fluoro_device=getattr(self.args, "fluoro_device", None) or "vulkan",
                    patient_twin_manifest=getattr(self.args, "patient_twin", None),
                )
            ),
            task=self._make_task(self._embodiment),
            # Registers the Gym ID under the kwarg Isaac Lab's stock RSL-RL
            # scripts read their agent config from, so `train_rsl_rl.py --task
            # endoluminal_navigation` resolves without a second env definition.
            # Harmless for the other modes: nothing reads it unless the trainer
            # asks for it.
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg="i4h_arena.agents.rsl_rl:ProfiledRslRlRunnerCfg",
            # The rod solver is a Newton solver. Arena's and Isaac Lab's runners never
            # call ``configure_args``, so without --presets they take this default.
            default_physics_backend=PhysicsBackend.NEWTON,
        )

    def configure_env_cfg(self, env_cfg: Any) -> None:
        from isaaclab.devices.device_base import DevicesCfg
        from isaaclab.envs.common import ViewerCfg

        from i4h_arena.devices.catheter import CatheterKeyboardCfg

        # A wider three-quarter view keeps the detector, arc, support, patient, and table
        # visible together in the viewport's narrower docked layout.
        env_cfg.viewer = ViewerCfg(eye=(2.45, -1.65, 1.65), lookat=(-0.25, 0.12, 0.78))
        # Isaac Lab's record_demos.py --teleop_device catheter_keyboard finds the device here.
        env_cfg.teleop_devices = DevicesCfg(
            devices={"catheter_keyboard": CatheterKeyboardCfg(sim_device=env_cfg.sim.device)}
        )

    def scene_rtx_settings(self) -> dict[str, Any]:
        return {"enable_translucency": True}

    def make_sensor_view(self, env: Any) -> SensorView:
        from i4h_arena.medical.fluoroscopy_sources import bind_fluoroscopy_sources

        # The detector draws the catheter and follows the C-arm through these sources.
        bind_fluoroscopy_sources(env.unwrapped)
        return super().make_sensor_view(env)

    def default_sensor_views(self) -> tuple[str, ...]:
        return ("fluoroscopy",)

    def sensor_view_titles(self) -> dict[str, str]:
        return {"fluoroscopy": "C-arm Sensor"}

    def sensor_view_outputs(self) -> dict[str, tuple[tuple[str, str], ...]]:
        return {
            "fluoroscopy": (
                ("DSA + Guidance", "dsa_guidance"),
                ("DSA Raw", "dsa"),
                ("DRR + Guidance", "guidance"),
                ("DRR Raw", "rgb"),
            )
        }

    def sensor_view_keyboard_toggles(self) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
        return {
            "fluoroscopy": {
                # Preserve the guidance/raw selection while toggling the
                # simulated contrast bolus, like the reference viewport.
                "X": (("dsa_guidance", "guidance"), ("dsa", "rgb")),
            }
        }

    def sensor_view_projection_presets(self) -> dict[str, tuple[tuple[str, str, float], ...]]:
        return {
            "fluoroscopy": (
                ("1 AP", "1", 0.0),
                ("2 LAO-45", "2", math.radians(45.0)),
                ("3 Lateral", "3", math.radians(90.0)),
                ("4 RAO-30", "4", math.radians(-30.0)),
            )
        }

    def sensor_view_projection_defaults(self) -> dict[str, int]:
        return {"fluoroscopy": 1}

    def sensor_view_appearances(self) -> dict[str, tuple[tuple[str, str], ...]]:
        # Same render either way, so the operator can pick the cath-lab look or the
        # radiograph look while the catheter is moving.
        return {
            "fluoroscopy": (
                ("Fluoroscopy", "fluoro"),
                ("X-ray", "xray"),
            )
        }

    def sensor_view_readouts(self, env: Any) -> dict[str, Callable[[], str]]:
        # Arrival is the only thing that ends a teleop episode and none of it is
        # visible on the detector: the target is an unmarked centerline point and
        # the tip leaves the frame on the way to it. Without this the operator is
        # driving blind to the criterion, and overshooting reads the same as
        # closing in.
        from i4h_arena.medical.navigation_goal import arrival_status

        return {"fluoroscopy": lambda: arrival_status(env.unwrapped)}

    def sensor_view_display_controls(self) -> dict[str, tuple[SensorDisplayControlSpec, ...]]:
        # Multiples of the window fitted from the first frame, so the same bounds suit any twin.
        return {
            "fluoroscopy": (
                SensorDisplayControlSpec(
                    label="Window level",
                    control="window_level",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.05,
                    default=0.0,
                ),
                SensorDisplayControlSpec(
                    label="Window width",
                    control="window_width",
                    minimum=0.25,
                    maximum=4.0,
                    step=0.05,
                    default=1.0,
                ),
            )
        }

    def sensor_view_sliders(self) -> dict[str, tuple[SensorSliderSpec, ...]]:
        return {
            "fluoroscopy": (
                SensorSliderSpec(
                    label="Velocity (mm/s)",
                    control="catheter_insertion_speed_mps",
                    minimum=1.0,
                    # Tracks the action terms' insertion ceiling. Anything above
                    # it would move the handle without moving the catheter,
                    # since the term clamps what the slider asks for.
                    maximum=60.0,
                    step=1.0,
                    # Fast enough to feel direct, slow enough that the shaft can
                    # shed the length being fed into it. Measured on this scene:
                    # a sustained hold at 30 mm/s drove wall penetration from
                    # +0.55 to +4.25 mm with 8 of 41 particles outside the lumen
                    # and never recovered, while 9 mm/s held penetration
                    # negative and 0 of 41 outside for 10,000 steps. The ceiling
                    # stays reachable for anyone who wants it.
                    default=9.0,
                    scale=0.001,
                ),
            )
        }
