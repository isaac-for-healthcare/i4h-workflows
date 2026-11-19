# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference commands from manual calibration; the Arena reducer owns task state."""

import numpy as np


class LiverReferenceController:
    def __init__(self, initial, workspace, example):
        self.action = np.array(initial, np.float32).copy()
        self.action[7:] = 0
        self.offset = np.array(example["capture_root"]) - np.array(example["capture_particle_positions"][0])
        self.action[3:7] = example["capture_quaternion_xyzw"]
        self.capture_grip = max(0.52, example["capture_grip"])
        self.arrival_ticks = 0

    def command(self, measurement, phase, destination):
        # Arrival is controller hysteresis, not an objective phase or hold timer.
        if phase == "approach" and self.arrival_ticks >= 3:
            phase = "capture"
        if phase == "approach":
            goal = measurement["geometry"][0] + self.offset
            delta = goal - self.action[:3]
            length = np.linalg.norm(delta)
            self.action[:3] += delta * min(1.0, (0.0001 if length < 0.002 else 0.001) / max(length, 1.0e-20))
            self.action[7] = min(0.45, self.action[7] + 0.015)
            self.arrival_ticks = self.arrival_ticks + 1 if length < 0.0003 else 0
        elif phase == "capture":
            self.action[7] = min(self.capture_grip, self.action[7] + 0.01)
        elif phase == "retract":
            delta = destination - measurement["geometry"][0]
            self.action[:3] += delta * min(1.0, 0.001 / max(np.linalg.norm(delta), 1.0e-20))
        elif phase in ("release", "relax"):
            self.action[7] = 0
        return self.action
