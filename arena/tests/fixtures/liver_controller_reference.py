# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Former hosted controller transfer path, retained as a numerical oracle."""

import numpy as np
import torch


def reference_teacher_action(self):
    self.component.sync_views()
    result = np.zeros((self.env.num_envs, 4), np.float32)
    measurements = self.probe.read()
    physical = self.physical.cpu().numpy()
    for e, teacher in enumerate(self.teachers):
        m = {key: value[e] for key, value in measurements.items()}
        teacher.action[:] = self.controller_action[e].cpu().numpy()
        teacher.arrival_ticks = int(self.controller_arrival[e, 0])
        s = self.objective.s[e].cpu().numpy()
        from i4h_arena.medical.liver import PHASES

        destination = None if s[4] < 0 else self.objective.f[e, :3].cpu().numpy().copy()
        command = teacher.command(m, PHASES[s[0]], destination)
        result[e, :3] = (command[:3] - physical[e, :3]) / 0.001
        result[e, 3] = command[7] * 2 - 1
        self.controller_action[e] = torch.as_tensor(command, device=self.env.device)
        self.controller_arrival[e, 0] = teacher.arrival_ticks
    return result
