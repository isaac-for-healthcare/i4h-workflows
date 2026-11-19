# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endoluminal navigation with the catheter drive carried on a robot flange.

A separate scene rather than a flag on the existing one, because the arm changes
facts the scene manifest is the source of truth for: the embodiment gains seven
recorded joints, the model gains an articulation, and the rod-only solver gains
a MJWarp half. Those belong in a manifest that lint can read, not in an argument
that lint never sees.

Everything else -- assets, fluoroscopy, C-arm, patient twin, viewport -- is
inherited unchanged.
"""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes.endoluminal_navigation import EndoluminalNavigationScene


class EndoluminalNavigationArmScene(EndoluminalNavigationScene):
    """Catheter fluoroscopy scene whose drive unit rides on a Franka hand."""

    name = "endoluminal_navigation_arm"

    def configure_env_cfg(self, env_cfg: Any) -> None:
        super().configure_env_cfg(env_cfg)
        # Take gravity out of the world, because the arm is the only body in it
        # that gravity would act on and it must not sag.
        #
        # The arm is a positioner: it is asked to hold the introducer on the
        # access site, and the introducer carries the wire's entry in the hand
        # frame, so a hand that sags takes the wire with it. Under the servo
        # gains this scene runs, that sag was 16 cm -- enough to pull the entry
        # clean off the patient. Stiffening the servo does not buy it back; at
        # 60x the gains the arm oscillates instead of drooping.
        #
        # ``disable_gravity`` on the spawn config cannot do this job. It is a
        # PhysX-namespace attribute, and this scene runs Newton, whose backend
        # has no per-body gravity at all -- only ``Model.set_gravity``. The flag
        # was authored, ignored, and left the arm drooping anyway.
        #
        # Nothing else in the scene loses anything. Every other prop here is an
        # ``AssetBaseCfg`` visual with no rigid body, and the catheter is not
        # simulated by this solver: the rod carries its own gravity in
        # ``XPBDRodSolverCfg``, so the wire still falls and still sags between
        # supports exactly as before.
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)

    def _make_embodiment(self) -> Any:
        """Build the flange-carried drive, which is what selects the coupled solver.

        Mirrors the armless gate: online RSL-RL takes the camera-free variant so
        the actor reads one flat vector, and every other mode keeps the named,
        image-bearing group. Both variants put an articulation in the model, so
        the choice does not change the integrator -- this scene is on coupled
        MJWarp + XPBD either way.
        """
        from i4h_arena.embodiments.franka_catheter import FrankaCatheterEmbodiment, FrankaCatheterRLEmbodiment

        embodiment = FrankaCatheterRLEmbodiment if self._wants_flat_rl_observations else FrankaCatheterEmbodiment
        return embodiment(patient_twin_manifest=self.args.patient_twin)

    def _joint_state_providers(self, env: Any, catheter: Any, carm_orbit: Any) -> dict[str, Any]:
        """Append the servo'd arm joints after the catheter and C-arm columns.

        The order is the contract with ``franka_catheter.yaml``: catheter and
        C-arm first so recordings stay column-compatible with the armless scene,
        arm appended.
        """
        from i4h_arena.embodiments.franka_catheter import ArmCatheterCArmJointStateProvider

        arm = env.unwrapped.action_manager.get_term("catheter")
        return {"robot": ArmCatheterCArmJointStateProvider(catheter, carm_orbit, arm)}
