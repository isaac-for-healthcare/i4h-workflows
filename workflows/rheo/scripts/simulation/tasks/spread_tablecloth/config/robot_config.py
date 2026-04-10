# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Robot presets for the `spread_tablecloth` task.

This task now reuses the Unitree G1 (29 DOF body + Dex3 hands) articulation
presets that are already maintained for `assemble_trocar`.
"""

from simulation.tasks.assemble_trocar.config.robot_config import G1RobotPresets

__all__ = ["G1RobotPresets"]
