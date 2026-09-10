# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-managed interactive catheter fluoroscopy scene."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from i4h_arena.adapters.scene_view import ArenaSceneView
from i4h_arena.scenes.base import Scene, SensorDisplayControlSpec, SensorSliderSpec


def resolve_fluoroscopy_backend(requested: str | None, patient_twin: str | None) -> str:
    """Select real DRR rendering when a patient twin is supplied."""
    return requested or ("slang" if patient_twin else "synthetic")


class EndoluminalNavigationScene(Scene):
    name = "endoluminal_navigation"

    def register_assets(self) -> None:
        import i4h_arena.assets.fluoroscopy_catheter_navigation  # noqa: F401

    def _make_embodiment(self) -> Any:
        """Build the embodiment this scene drives.

        Overridden by the arm-borne variant, which swaps in a drive carried on a
        robot flange and takes the scene onto the coupled MJWarp + rod solver.
        """
        from i4h_arena.embodiments.catheter import CatheterEmbodiment

        return CatheterEmbodiment(patient_twin_manifest=self.args.patient_twin)

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.fluoroscopy_catheter_navigation import make_assets

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self._make_embodiment(),
            scene=ArenaScene(
                assets=make_assets(
                    fluoro_backend=resolve_fluoroscopy_backend(self.args.fluoro_backend, self.args.patient_twin),
                    fluoro_device=self.args.fluoro_device,
                    patient_twin_manifest=self.args.patient_twin,
                )
            ),
            task=None,
        )

    def configure_env_cfg(self, env_cfg: Any) -> None:
        from isaaclab.envs.common import ViewerCfg

        # A wider three-quarter view keeps the detector, arc, support, patient, and table
        # visible together in the viewport's narrower docked layout.
        env_cfg.viewer = ViewerCfg(eye=(2.45, -1.65, 1.65), lookat=(-0.25, 0.12, 0.78))
        env_cfg.sim.render.enable_translucency = True

    def make_view(self, env: Any) -> ArenaSceneView:
        catheter = env.unwrapped.scene["catheter"]
        carm_orbit = env.unwrapped.action_manager.get_term("carm_orbit")
        fluoroscopy = env.unwrapped.scene["fluoroscopy"]
        fluoroscopy.bind_catheter_provider(catheter)
        from i4h_arena.medical.carm import (
            ReferenceProjectionCArmStateProvider,
            SceneCArmStateProvider,
            follow_tip_enabled,
        )

        detector_size_m = (0.6144, 0.6144)
        if self.args.patient_twin:
            from i4h_arena.medical.patient_twin import PatientTwin
            from i4h_arena.medical.patient_volume import PatientVolume

            carm_provider = ReferenceProjectionCArmStateProvider(
                PatientVolume.load(PatientTwin.load(self.args.patient_twin)),
                carm_orbit,
                detector_size_m=detector_size_m,
                # The detector covers 307 mm of a 510 mm route, so a fixed
                # isocenter leaves roughly 40% of every episode with the tip off
                # the frame. An operator can still work from the distance
                # readout; a policy trained on those frames cannot, since the
                # action has no visible cause in the image it is paired with.
                # Opt-in until a live run confirms the frame, because panning is
                # the first thing to give the renderer a non-zero pose
                # translation and the first attempt rendered an unusable image.
                tip_source=catheter if follow_tip_enabled() else None,
            )
        else:
            carm_provider = self._scene_data_carm_provider(env, detector_size_m) or SceneCArmStateProvider(
                env.unwrapped.scene["xray_source"],
                env.unwrapped.scene["detector"],
                detector_size_m=detector_size_m,
            )
        fluoroscopy.bind_carm_provider(carm_provider)

        return ArenaSceneView(
            env,
            objects=self.spec.objects,
            robots=self.spec.robots,
            cameras=self.spec.cameras,
            gripper=False,
            joint_state_providers=self._joint_state_providers(env, catheter, carm_orbit),
        )

    def _joint_state_providers(self, env: Any, catheter: Any, carm_orbit: Any) -> dict[str, Any]:
        """Recorded procedure state for this scene.

        A hook rather than a literal because the arm-borne variant appends the
        servo'd arm joints, and the order it appends them in has to stay in step
        with what its embodiment manifest declares.
        """
        from i4h_arena.embodiments.catheter import CatheterCArmJointStateProvider

        return {"robot": CatheterCArmJointStateProvider(catheter, carm_orbit)}

    @staticmethod
    def _scene_data_carm_provider(env: Any, detector_size_m: tuple[float, float]) -> Any | None:
        """Read C-arm poses through SceneDataProvider when one is available.

        The provider is the backend-agnostic path for body transforms, so it is
        preferred over per-asset ``get_world_poses()``. It returns ``None`` when
        no provider is present or the prims are not registered with it, leaving
        the caller to fall back rather than losing the C-arm entirely.
        """
        try:
            from isaaclab.sim import SimulationContext

            from i4h_arena.medical.newton_providers import SceneDataCArmStateProvider

            provider = SimulationContext.instance().get_scene_data_provider()
            if provider is None:
                return None
            num_envs = int(env.unwrapped.num_envs)
            root = env.unwrapped.scene.env_prim_paths
            return SceneDataCArmStateProvider(
                provider,
                source_paths=[f"{root[index]}/CArm/Orbit/Source" for index in range(num_envs)],
                detector_paths=[f"{root[index]}/CArm/Orbit/Detector" for index in range(num_envs)],
                detector_size_m=detector_size_m,
            )
        except Exception:
            return None

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
