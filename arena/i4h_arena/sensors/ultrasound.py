# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-driven B-mode sensor backed by i4h-sensor-simulation's CUDA ray tracer."""

from dataclasses import dataclass, field

import numpy as np
import torch
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.configclass import configclass
from scipy.spatial.transform import Rotation

from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_ultrasound import TCP_FROM_IMAGER, imager_in_patient, patient_layout, surface_height
from i4h_arena.tensor_utils import to_torch


@dataclass
class UltrasoundSensorData:
    output: dict = field(default_factory=dict)
    frame_id: object = None


class UltrasoundSensor(SensorBase):
    def __init__(self, cfg):
        self._data = UltrasoundSensorData()
        self._probe_frame = None
        self.layout = patient_layout(PatientTwin.load(cfg.patient_twin_manifest))
        super().__init__(cfg)

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    def bind_probe(self, frame):
        self._probe_frame = frame

    def _initialize_impl(self):
        super()._initialize_impl()
        if self._num_envs != 1:
            raise ValueError("Patient ultrasound currently supports --envs 1")
        from ultrasound_simulator import cuda as rs
        from ultrasound_simulator.usd import world_from_usd

        self._rs = rs
        twin = PatientTwin.load(self.cfg.patient_twin_manifest)
        self._world, self._materials, self.meshes = world_from_usd(twin.artifacts["anatomy_usd"])
        self._simulator = rs.RaytracingUltrasoundSimulator(self._world, self._materials)
        self._params = rs.SimParams()
        self._params.b_mode_size = (self.cfg.width, self.cfg.height)
        self._params.t_far = self.cfg.depth_mm
        self._params.buffer_size = 4096
        self._params.contact_epsilon = 5.0
        self._data.output = {
            "rgb": torch.zeros((1, self.cfg.height, self.cfg.width, 3), dtype=torch.uint8, device=self._device),
            "b_mode_db": torch.full((1, self.cfg.height, self.cfg.width, 1), -120.0, device=self._device),
            "contact_distance_m": torch.zeros((1, 1), device=self._device),
            "probe_pose": torch.zeros((1, 7), device=self._device),
            "target_pose": torch.zeros((1, 7), device=self._device),
        }
        self._data.frame_id = torch.zeros(1, dtype=torch.int64, device=self._device)

    def _update_buffers_impl(self, env_mask):
        if self._probe_frame is None:
            return
        frame = self._probe_frame.data
        pos = to_torch(frame.target_pos_w)[0, 0].detach().cpu().numpy()
        quat = to_torch(frame.target_quat_w)[0, 0].detach().cpu().numpy()
        position, angles = imager_in_patient(pos, quat, self.layout.world_from_patient_m)
        probe = self._rs.CurvilinearProbe(
            self._rs.Pose(position.astype(np.float32), angles.astype(np.float32)), num_elements_x=128
        )
        raw = self._simulator.simulate(probe, self._params)
        raw = np.clip(np.nan_to_num(raw, nan=-120, neginf=-120, posinf=0), -120, 0).astype(np.float32)
        image = (np.clip((raw + 60) / 60, 0, 1) * 255).astype(np.uint8)
        self._data.output["rgb"][0] = torch.as_tensor(np.repeat(image[..., None], 3, axis=-1), device=self._device)
        self._data.output["b_mode_db"][0, ..., 0] = torch.as_tensor(raw, device=self._device)
        world_imager = pos + Rotation.from_quat(quat).apply(TCP_FROM_IMAGER[:3, 3])
        try:
            gap = world_imager[2] - surface_height(
                self.layout.skin_vertices_m, self.layout.skin_faces, world_imager[:2]
            )
        except ValueError:
            gap = 1.0
        self._data.output["contact_distance_m"][0, 0] = gap
        self._data.output["probe_pose"][0] = torch.as_tensor(np.r_[pos, quat[[3, 0, 1, 2]]], device=self._device)
        target, rot = self.layout.tcp_target()
        target_quat = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        self._data.output["target_pose"][0] = torch.as_tensor(np.r_[target, target_quat], device=self._device)
        self._data.frame_id += 1


@configclass
class UltrasoundSensorCfg(SensorBaseCfg):
    class_type: type = UltrasoundSensor
    patient_twin_manifest: str = ""
    width: int = 384
    height: int = 384
    depth_mm: float = 180.0
