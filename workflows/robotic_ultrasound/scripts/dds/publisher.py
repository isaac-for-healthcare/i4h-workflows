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

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import rti.connextdds as dds  # noqa: F401

from .base import DDSEntity


class Publisher(DDSEntity, ABC):
    """
    Base class for all publishers.

    Args:
        topic: The topic name to publish to.
        cls: The class type of the data to publish.
        period: Time period between successive publications in seconds.
        domain_id: The DDS domain ID to publish to.
        qos_provider_path: Path to XML file containing QoS profiles.
        transport_profile: Transport QoS profile name (format: "Library::Profile").
        writer_profile: Writer QoS profile name (format: "Library::Profile").
    """

    def __init__(
        self, 
        topic: str, 
        cls: Any, 
        period: float, 
        domain_id: int,
        qos_provider_path: str,
        transport_profile: str,
        writer_profile: str
    ):
        super().__init__(
            topic=topic,
            cls=cls,
            period=period,
            domain_id=domain_id,
            qos_provider_path=qos_provider_path,
            transport_profile=transport_profile,
            entity_profile=writer_profile
        )
        self.logger = logging.getLogger(__name__)

        # Initialize DDS entities with QoS
        participant = self._create_participant()
        dds_topic = self._create_topic(participant)
        _, _, writer_qos = self._get_cached_qos(writer_profile)
        self.dds_writer = dds.DataWriter(dds_topic, qos=writer_qos)

    def _get_entity_qos(self, provider: dds.QosProvider, profile_name: str) -> dds.DataWriterQos:
        """Get writer-specific QoS settings from provider."""
        return provider.datawriter_qos_from_profile(profile_name)

    def write(self, dt: float = 0.0, sim_time: float = 0.0) -> float:
        """
        Write data to the DDS writer.

        Args:
            dt: The time since the last publication.
            sim_time: The simulation time.

        Returns:
            The time taken to write the data to the DDS writer.
        """
        if not self.dds_writer:
            self.logger.info(f"{self.domain_id}:{self.topic} - DDS Writer is not ready.")
            return 0

        start_time = time.monotonic()
        topic = self.produce(dt, sim_time)
        self.dds_writer.write(topic)
        exec_time = time.monotonic() - start_time
        return exec_time

    @abstractmethod
    def produce(self, dt: float = 0.0, sim_time: float = 0.0) -> Any:
        """
        Produce data to be published.

        Args:
            dt: The time since the last publication.
            sim_time: The simulation time.
        """
        pass
