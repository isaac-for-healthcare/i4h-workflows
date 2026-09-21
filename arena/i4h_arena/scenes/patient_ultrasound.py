# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared optional patient imaging behavior for Franka ultrasound scenes."""

from dataclasses import replace


class PatientUltrasoundMixin:
    def configure_args(self, args):
        if getattr(args, "patient_twin", None) and "ultrasound" not in self.spec.cameras:
            self.spec = replace(self.spec, cameras=(*self.spec.cameras, "ultrasound"))
        super().configure_args(args)

    def make_view(self, env):
        view = super().make_view(env)
        if getattr(self.args, "patient_twin", None):
            sensor = env.unwrapped.scene["ultrasound"]
            sensor.bind_probe(env.unwrapped.scene["ee_frame"])
        return view

    def default_sensor_views(self):
        return ("ultrasound",) if getattr(self.args, "patient_twin", None) else ()

    def sensor_view_titles(self):
        return {"ultrasound": "Ultrasound - Patient B-mode"}

    def configure_env_cfg(self, env_cfg):
        super().configure_env_cfg(env_cfg)
        if not getattr(self.args, "patient_twin", None):
            return
        # Source registration must remain fixed: phantom randomization would
        # separate the acoustic geometry from its rendered patient.
        for name in ("reset_object_position", "reset_joint_position", "reset_target", "reset_robot"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        # The patient imaging task checks actual probe/skin coupling and frames,
        # not the old phantom's fixed bounding-box success heuristic.
        env_cfg.terminations.success = None
        from isaaclab.envs.common import ViewerCfg

        env_cfg.viewer = ViewerCfg(eye=(1.8, -1.6, 1.5), lookat=(0.5, 0, 0.15))

    def patient_reset(self, env, view):
        import torch
        from isaaclab.utils.math import compute_pose_error
        from scipy.spatial.transform import Rotation

        from i4h_arena.tensor_utils import to_torch

        sensor = env.unwrapped.scene["ultrasound"]
        sensor.bind_probe(env.unwrapped.scene["ee_frame"])
        target, rotation = sensor.layout.tcp_target()
        device = env.unwrapped.device
        position = torch.tensor(target[None], dtype=torch.float32, device=device)
        quaternion = torch.tensor(Rotation.from_matrix(rotation).as_quat()[None], dtype=torch.float32, device=device)
        # Physics servo, without teleporting the robot or detaching its probe.
        for _ in range(100):
            frame = env.unwrapped.scene["ee_frame"].data
            dp, dr = compute_pose_error(
                to_torch(frame.target_pos_w)[:, 0],
                to_torch(frame.target_quat_w)[:, 0],
                position,
                quaternion,
                rot_error_type="axis_angle",
            )
            env.step(torch.cat((dp, dr), dim=-1))
        view.invalidate()
