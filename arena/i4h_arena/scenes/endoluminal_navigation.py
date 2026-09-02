# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-managed interactive catheter fluoroscopy scene."""

from __future__ import annotations

import math
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

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.fluoroscopy_catheter_navigation import make_assets
        from i4h_arena.embodiments.catheter import CatheterEmbodiment

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=CatheterEmbodiment(patient_twin_manifest=self.args.patient_twin),
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
        from i4h_arena.medical.carm import ReferenceProjectionCArmStateProvider, SceneCArmStateProvider

        detector_size_m = (0.6144, 0.6144)
        if self.args.patient_twin:
            from i4h_arena.medical.patient_twin import PatientTwin
            from i4h_arena.medical.patient_volume import PatientVolume

            carm_provider = ReferenceProjectionCArmStateProvider(
                PatientVolume.load(PatientTwin.load(self.args.patient_twin)),
                carm_orbit,
                detector_size_m=detector_size_m,
            )
        else:
            carm_provider = SceneCArmStateProvider(
                env.unwrapped.scene["xray_source"],
                env.unwrapped.scene["detector"],
                detector_size_m=detector_size_m,
            )
        fluoroscopy.bind_carm_provider(carm_provider)
        from i4h_arena.embodiments.catheter import CatheterCArmJointStateProvider

        return ArenaSceneView(
            env,
            objects=self.spec.objects,
            robots=self.spec.robots,
            cameras=self.spec.cameras,
            gripper=False,
            joint_state_providers={"robot": CatheterCArmJointStateProvider(catheter, carm_orbit)},
        )

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
                    maximum=30.0,
                    step=1.0,
                    default=16.0,
                    scale=0.001,
                ),
            )
        }
