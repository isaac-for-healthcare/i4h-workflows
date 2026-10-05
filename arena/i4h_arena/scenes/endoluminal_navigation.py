# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-managed interactive catheter fluoroscopy scene."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from i4h_arena.adapters.scene_view import ArenaSceneView
from i4h_arena.scenes.base import Scene, SensorDisplayControlSpec, SensorSliderSpec

logger = logging.getLogger("i4h_arena.scene")


def resolve_fluoroscopy_backend(requested: str | None, patient_twin: str | None) -> str:
    """Select real DRR rendering when a patient twin is supplied."""
    return requested or ("slang" if patient_twin else "synthetic")


class EndoluminalNavigationScene(Scene):
    name = "endoluminal_navigation"

    _embodiment: Any | None = None

    def register_assets(self) -> None:
        import i4h_arena.assets.fluoroscopy_catheter_navigation  # noqa: F401

    @property
    def _navigation_target_world_m(self) -> tuple[float, float, float] | None:
        """Distal end of the planned route, or ``None`` without a patient twin.

        A phantom scene has no centerline and therefore no goal, which is why
        this is optional rather than an error: the recording then carries the
        tip's position without a distance to anything.
        """
        return getattr(self._embodiment, "navigation_target_world_m", None)

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

    def _make_embodiment(self) -> Any:
        """Build the embodiment this scene drives.

        Overridden by the arm-borne variant, which swaps in a drive carried on a
        robot flange and takes the scene onto the coupled MJWarp + rod solver.

        Online RSL-RL takes the camera-free variant instead, whose observation
        group is one flat vector.
        """
        from i4h_arena.embodiments.catheter import CatheterEmbodiment, CatheterRLEmbodiment

        embodiment = CatheterRLEmbodiment if self._wants_flat_rl_observations else CatheterEmbodiment
        return embodiment(patient_twin_manifest=self.args.patient_twin)

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.fluoroscopy_catheter_navigation import make_assets

        # Kept so ``make_view`` can record the tip's distance to the same target
        # arrival is judged against, rather than a second copy of it.
        self._embodiment = self._make_embodiment()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=self._embodiment,
            scene=ArenaScene(
                assets=make_assets(
                    fluoro_backend=resolve_fluoroscopy_backend(self.args.fluoro_backend, self.args.patient_twin),
                    fluoro_device=self.args.fluoro_device,
                    patient_twin_manifest=self.args.patient_twin,
                )
            ),
            task=None,
            # Registers the Gym ID under the kwarg Isaac Lab's stock RSL-RL
            # scripts read their agent config from, so `train_rsl_rl.py --task
            # endoluminal_navigation` resolves without a second env definition.
            # Harmless for the other modes: nothing reads it unless the trainer
            # asks for it.
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg="i4h_arena.agents.rsl_rl:ProfiledRslRlRunnerCfg",
        )

    def configure_env_cfg(self, env_cfg: Any) -> None:
        from isaaclab.envs.common import ViewerCfg

        # From the manifest rather than left at the framework's 50 s default,
        # which at this scene's 30 Hz is 1500 control steps against the 600 the
        # manifest and both PPO configs agree on. The `time_out` term turns
        # this length into truncation, so a wrong one here is a third step
        # budget that silently outvotes the other two.
        #
        # No ``+ 1`` as the other scenes use. That extra step leaves the
        # workflow runner's own cap to end the episode first, but `app.py`
        # strips the term on that path, so the only consumers are the RL
        # trainers, which should truncate at the cap itself. The division is
        # exact at these numbers: 600 / 30 is 20.0, and Isaac Lab recovers 600
        # from it, where 601 / 30 rounds up to 602.
        #
        # ``getattr`` rather than the attribute the other scenes read directly:
        # ``--episode-steps`` is an arena CLI option, and this scene also runs
        # under the RSL-RL interop parser, which does not define it.
        steps = getattr(self.args, "episode_steps", None) or self.spec.max_steps
        env_cfg.episode_length_s = steps / self.spec.control_hz
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
                # On by default, and opt-out through the environment.
                # ``follow_tip_enabled`` holds why, along with the render that
                # confirmed the panned frame -- restating it here is how this
                # comment came to describe the feature as still opt-in long
                # after that render had settled it.
                tip_source=catheter if follow_tip_enabled() else None,
            )
        else:
            carm_provider = self._scene_data_carm_provider(env, detector_size_m) or SceneCArmStateProvider(
                env.unwrapped.scene["xray_source"],
                env.unwrapped.scene["detector"],
                detector_size_m=detector_size_m,
            )
        fluoroscopy.bind_carm_provider(carm_provider)

        from i4h_arena.medical.catheter_diagnostics import CatheterEpisodeDiagnostics

        return ArenaSceneView(
            env,
            objects=self.spec.objects,
            robots=self.spec.robots,
            cameras=self.spec.cameras,
            gripper=False,
            joint_state_providers=self._joint_state_providers(env, catheter, carm_orbit),
            # The four commanded joints and the projection cannot tell a clean
            # run from one where the wire coiled, so a recording needs the rod's
            # own shape alongside them to be judged after the fact.
            diagnostics_provider=CatheterEpisodeDiagnostics(catheter, target_world_m=self._navigation_target_world_m),
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

        Any construction failure falls back, because the fallback is a supported
        path and refusing to build the scene over it would be worse. It is
        logged, though: without that, a run that quietly stopped using the
        provider is indistinguishable from one that never had it.
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
        except Exception:  # noqa: BLE001 - every construction failure is a fall back
            # Warning rather than debug, and with the traceback, because
            # ``SceneDataCArmStateProvider`` raises during construction on
            # purpose and the message is the diagnosis: a Lab revision whose
            # ``create_mapping`` does not restrict the output reports the
            # transform count it actually got. Rendering continues on the
            # per-prim path, so this line is the only trace of the downgrade.
            logger.warning(
                "C-arm SceneDataProvider unavailable; reading the source and detector prims directly",
                exc_info=True,
            )
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
