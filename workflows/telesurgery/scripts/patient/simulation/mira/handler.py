# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import numpy as np
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class ApiHandlerOp(Operator):
    def __init__(self, fragment, callback, *args, **kwargs):
        self.callback = callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        lj = [0] * 6
        rj = [0] * 6
        if message["method"] == "set_mira_polar_delta" or message["method"] == "set_mira_cartesian_delta":
            lp = message["params"]["left"]
            rp = message["params"]["right"]
            if not np.all(lp == 0):
                lj = self.pos_to_joints_ik(lp[0], lp[1], lp[2], lp[3])
            if not np.all(rp == 0):
                rj = self.pos_to_joints_ik(rp[0], rp[1], rp[2], rp[3])

            message["pose_delta"] = {
                "left": lj,
                "right": rj,
            }

        if self.callback:
            self.callback(message)
        else:
            print(json.dumps(message))

    # TODO:: Not the best/accurate IK;  Get the IK from MIRA (refer: get_mira_polar_ik)
    def pos_to_joints_ik(self, delta_x, delta_y, delta_sweep_deg, delta_elbow_deg, L1=400, L2=300):
        delta_j1 = delta_sweep_deg

        delta_r = np.sqrt(delta_x**2 + delta_y**2)
        direction = np.arctan2(delta_y, delta_x)

        # Assume small motion approximation: distribute delta_r into j2 and j3
        delta_j2 = np.degrees((delta_r / L1) * np.cos(direction))  # Crude approx
        delta_j3 = delta_elbow_deg  # Directly specified

        # Wrist joints remain unchanged
        delta_j4 = 0
        delta_j5 = 0
        delta_j6 = 0
        return [delta_j1, delta_j2, delta_j3, delta_j4, delta_j5, delta_j6]
