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

from typing import List

import omni.usd
from dds.publisher import Publisher
from dds.subscriber import Subscriber
from omni.isaac.core import World
from simulation.utils.ratelimit import add_physx_step_callback, remove_physx_callback

__all__ = ["Annotator"]


class Annotator:
    """Base class for simulation annotators that handle DDS communication.

    This class provides the basic infrastructure for creating annotators that can
    publish and subscribe to DDS topics in the simulation environment. It manages
    the lifecycle of publishers and subscribers, ensuring proper initialization
    and cleanup.

    Args:
        name: Unique identifier for the annotator
        prim_path: USD path to the sensor primitive in the stage
        publishers: List of DDS publishers to manage
        subscribers: List of DDS subscribers to manage
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        publishers: List[Publisher] | None = None,
        subscribers: List[Subscriber] | None = None,
    ):
        """Initialize the annotator with the given name, prim path, and DDS endpoints."""
        self.name = name
        self.prim_path = prim_path
        self.stage = omni.usd.get_context().get_stage()
        self.sensor_prim = self.stage.GetPrimAtPath(prim_path)
        self.publishers = [p for p in publishers if p is not None] if publishers else []
        self.subscribers = [s for s in subscribers if s is not None] if subscribers else []

    def start(self, world: World) -> None:
        """Start all publishers and subscribers with rate-limited callbacks.

        This method registers each publisher and subscriber with the physics
        simulation loop at their specified rates.

        Args:
            world: The simulation world object that provides the physics context
        """
        for publisher in self.publishers:
            add_physx_step_callback(f"{publisher.topic}", publisher.period, publisher.write, world)
        for subscriber in self.subscribers:
            add_physx_step_callback(f"{subscriber.topic}", subscriber.period, subscriber.read, world)
            subscriber.start()

    def stop(self, world: World) -> None:
        """Stop all publishers and subscribers and cleanup their callbacks.

        This method removes all registered callbacks from the physics simulation
        loop and stops the subscribers.

        Args:
            world: The simulation world object that provides the physics context
        """
        for publisher in self.publishers:
            remove_physx_callback(f"{publisher.topic}", world)
        for subscriber in self.subscribers:
            remove_physx_callback(f"{subscriber.topic}", world)
            subscriber.stop()
