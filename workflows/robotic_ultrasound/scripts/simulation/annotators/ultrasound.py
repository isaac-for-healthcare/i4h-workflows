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

from typing import Any

from dds.publisher import Publisher
from dds.schemas.usp_info import UltraSoundProbeInfo
from isaacsim.core.prims import XFormPrim
from simulation.configs.config import UltraSoundConfig


class UltraSoundPublisher(Publisher):
    """Publisher for ultrasound probe state information.

    This class handles publishing ultrasound probe position and orientation through
    DDS topics. It tracks the probe's transform in the simulation world.

    Args:
        prim_path: USD path to ultrasound probe
        topic: DDS topic name
        period: Publishing period in seconds
        domain_id: DDS domain identifier
    """

    def __init__(self, prim_path: str, topic: str, period: float, domain_id):
        """Initialize the ultrasound probe publisher."""
        super().__init__(topic, UltraSoundProbeInfo, period, domain_id)
        self.prim_path = prim_path

    def produce(self, dt: float, sim_time: float) -> Any:
        """Produce ultrasound probe state information for publishing.

        Gathers current probe position and orientation in world coordinates.

        Args:
            dt: Time delta since last physics step
            sim_time: Current simulation time

        Returns:
            UltraSoundProbeInfo: Probe state information including position and orientation
                refer to dds.schemas.usp_info.UltraSoundProbeInfo.
        """
        usb = XFormPrim(prim_path=self.prim_path)
        position, orientation = usb.get_world_pose()

        output = UltraSoundProbeInfo()
        output.position = position.tolist()
        output.orientation = orientation.tolist()
        return output

    @staticmethod
    def new_instance(config: UltraSoundConfig):
        """Create a new UltraSoundPublisher instance from configuration.

        Args:
            config: Ultrasound probe configuration object
        """
        if not config.topic_info or not config.topic_info.name:
            return None

        return UltraSoundPublisher(
            prim_path=config.prim_path,
            topic=config.topic_info.name,
            period=config.topic_info.period,
            domain_id=config.topic_info.domain_id,
        )
