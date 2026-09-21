# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate real probe contact, lift-off and recovery with the native sensor."""

from dataclasses import dataclass
from i4h_common.types import Pose
from i4h_engine.status import Status
from i4h_engine.task import Task


class PatientUltrasoundScan(Task):
    requires = {"action_space": "ee_pose", "dof": 6}

    @dataclass
    class Outputs:
        contact_frames: int = 0
        lifted_frames: int = 0
        recovered_frames: int = 0
        peak_intensity: float = 0.0

    def on_enter(self, ctx, inputs):
        self.ticks = 0
        self.contact = self.lifted = self.recovered = 0
        self.peak = 0.0
        self.target = ctx.scene.sensor_signal("ultrasound", "target_pose")
        if self.target is None:
            raise ValueError("validate-ultrasound requires --patient-twin with SOMA and liver anatomy")
        self.target = self.target.copy()

    def tick(self, ctx):
        # Stay within the existing 250-step scene cap. Commands drive the real
        # robot through the standard actuation interface; no virtual probe moves.
        phase = 0 if self.ticks < 40 else (1 if self.ticks < 120 else 2)
        position = self.target[:3].copy()
        if phase == 1:
            position[2] += 0.04
        ctx.act.set_ee_target(Pose(pos=position[None], quat=self.target[3:][None]))
        gap = float(ctx.scene.sensor_signal("ultrasound", "contact_distance_m")[0])
        frame = ctx.scene.camera("ultrasound").to_array()
        mean = float(frame.mean())
        self.peak = max(self.peak, mean)
        if abs(gap) < 0.005 and mean > 1.0:
            if phase == 0:
                self.contact += 1
            elif phase == 2:
                self.recovered += 1
        if phase == 1 and gap > 0.02 and mean < 0.1:
            self.lifted += 1
        self.ticks += 1
        if self.ticks < 220:
            return Status.RUNNING
        return Status.SUCCESS if min(self.contact, self.lifted, self.recovered) >= 5 else Status.FAILURE

    def on_exit(self, ctx):
        return self.Outputs(self.contact, self.lifted, self.recovered, self.peak)
