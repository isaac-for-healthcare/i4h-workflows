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

import asyncio
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import rti.connextdds as dds  # noqa: F401
from dds.utils import get_default_entity_profile, get_default_qos_provider_path, get_default_transport_profile

from .base import DDSEntity


class Subscriber(DDSEntity, ABC):
    """
    Abstract base class for DDS subscribers.

    This class provides the base functionality for subscribing to DDS topics. It supports both
    synchronous and asynchronous reading modes, and can either store received data in a queue
    or process it immediately through a callback.

    Args:
        topic: The DDS topic name to subscribe to.
        cls: The class type of the data being received.
        period: Time period between successive reads in seconds (1/frequency).
        domain_id: The DDS domain ID to subscribe on.
        qos_provider_path: Path to XML file containing QoS profiles.
        transport_profile: Transport QoS profile name (format: "Library::Profile").
        reader_profile: Reader QoS profile name (format: "Library::Profile").
        add_to_queue: If True, stores received data in a queue.
            If False, processes data immediately. Defaults to True.
    """

    def __init__(
        self,
        topic: str,
        cls: Any,
        period: float,
        domain_id: int,
        qos_provider_path: str | None = None,
        transport_profile: str | None = None,
        reader_profile: str | None = None,
        add_to_queue: bool = True,
    ):
        qos_provider_path = qos_provider_path or get_default_qos_provider_path()
        transport_profile = transport_profile or get_default_transport_profile()
        reader_profile = reader_profile or get_default_entity_profile(topic)

        super().__init__(
            topic=topic,
            cls=cls,
            period=period,
            domain_id=domain_id,
            qos_provider_path=qos_provider_path,
            transport_profile=transport_profile,
            entity_profile=reader_profile,
        )
        self.dds_reader = None
        self.stop_event = None
        self.add_to_queue = add_to_queue
        self.data_q: Any = queue.Queue()

    def _get_entity_qos(self, provider: dds.QosProvider, profile_name: str) -> dds.DataReaderQos:
        """Get reader-specific QoS settings from provider."""
        return provider.datareader_qos_from_profile(profile_name)

    def _initialize_reader(self) -> None:
        """Initialize the DDS reader with QoS settings."""
        participant = self._create_participant()
        dds_topic = self._create_topic(participant)
        _, _, reader_qos = self._get_cached_qos(self.entity_profile)
        self.dds_reader = dds.DataReader(dds_topic, qos=reader_qos)

    async def read_async(self):
        """
        Asynchronously read data from the DDS topic.

        This method continuously reads data using async/await pattern and either
        stores it in the queue or processes it immediately based on add_to_queue setting.
        """
        if self.dds_reader is None:
            self._initialize_reader()
        print(f"{self.domain_id}:{self.topic} - Thread is reading data => {self.dds_reader.topic_name}")
        async for data in self.dds_reader.take_data_async():
            if self.add_to_queue:
                self.data_q.put(data)
            else:
                self.consume(data)
        print(f"{self.domain_id}:{self.topic} - Thread End")

    def read_sync(self):
        """
        Synchronously read data from the DDS topic.

        This method runs in a separate thread and continuously reads data at the specified
        period. Data is either stored in the queue or processed immediately based on
        add_to_queue setting.
        """
        print(f"{self.domain_id}:{self.topic} - Thread is reading data => {self.dds_reader.topic_name}")
        while self.stop_event and not self.stop_event.is_set():
            try:
                for data in self.dds_reader.take_data():
                    if self.add_to_queue:
                        self.data_q.put(data)
                    else:
                        self.consume(data)
                time.sleep(self.period if self.period > 0 else 1)
            except Exception as e:
                print(f"Error in {self.dds_reader.topic_name}: {e}")
                raise e

    def read_data(self) -> Any:
        """
        Retrieve a single data item from the queue.

        Returns:
            Any: The next data item from the queue, or None if the queue is empty.
        """
        if not self.data_q.empty():
            return self.data_q.get()
        return None

    def read(self, dt: float, sim_time: float) -> float:
        """
        Process all available data in the queue.

        Args:
            dt: Delta time since last update.
            sim_time: Current simulation time.

        Returns:
            float: Execution time in seconds.
        """
        start_time = time.monotonic()
        while not self.data_q.empty():
            data = self.data_q.get()
            print(f"{self.domain_id}:{self.topic} - Queue has data to run action: {data}")
            self.consume(data)
        exec_time = time.monotonic() - start_time
        return exec_time

    def start(self):
        """
        Start the subscriber.

        Initializes the DDS reader if not already initialized and starts the reading thread.
        If a previous reading thread exists, it is stopped before starting a new one.
        """
        self.stop()
        self.stop_event = threading.Event()

        if self.dds_reader is None:
            self._initialize_reader()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_in_executor(None, self.read_sync)

    def stop(self):
        """
        Stop the subscriber.

        Sets the stop event to terminate the reading thread and cleans up resources.
        """
        if self.stop_event:
            self.stop_event.set()
        self.stop_event = None

    @abstractmethod
    def consume(self, data) -> None:
        """
        Process received data.

        This abstract method must be implemented by subclasses to define how
        received data should be processed.

        Args:
            data: The received data item.
        """
        pass


class SubscriberWithQueue(Subscriber):
    """
    Subscriber implementation that stores received data in a queue.

    Args:
        domain_id: The DDS domain ID to subscribe on.
        topic: The DDS topic name to subscribe to.
        cls: The class type of the data being received.
        period: Time period between successive reads in seconds.
        qos_provider_path: Path to XML file containing QoS profiles.
        transport_profile: Transport QoS profile name (format: "Library::Profile").
        reader_profile: Reader QoS profile name (format: "Library::Profile").
    """

    def __init__(
        self,
        domain_id: int,
        topic: str,
        cls: Any,
        period: float,
        qos_provider_path: str | None = None,
        transport_profile: str | None = None,
        reader_profile: str | None = None,
    ):
        super().__init__(
            topic,
            cls,
            period,
            domain_id,
            add_to_queue=True,
            qos_provider_path=qos_provider_path,
            transport_profile=transport_profile,
            reader_profile=reader_profile,
        )

    def consume(self, data) -> None:
        """
        Not meant to be called directly.

        Args:
            data: The received data item.

        Raises:
            RuntimeError: If called directly. Use read_data() instead.
        """
        raise RuntimeError("This should not happen; Call read_data() explicitly.")


class SubscriberWithCallback(Subscriber):
    """
    Subscriber implementation that processes data immediately via callback.

    Args:
        cb: Callback function that takes (topic, data) as arguments.
        domain_id: The DDS domain ID to subscribe on.
        topic: The DDS topic name to subscribe to.
        cls: The class type of the data being received.
        period: Time period between successive reads in seconds.
        qos_provider_path: Path to XML file containing QoS profiles.
        transport_profile: Transport QoS profile name (format: "Library::Profile").
        reader_profile: Reader QoS profile name (format: "Library::Profile").
    """

    def __init__(
        self,
        cb: Callable,
        domain_id: int,
        topic: str,
        cls: Any,
        period: float,
        qos_provider_path: str | None = None,
        transport_profile: str | None = None,
        reader_profile: str | None = None,
    ):
        super().__init__(
            topic,
            cls,
            period,
            domain_id,
            add_to_queue=False,
            qos_provider_path=qos_provider_path,
            transport_profile=transport_profile,
            reader_profile=reader_profile,
        )
        self.cb = cb

    def consume(self, data) -> None:
        """
        Process received data using the callback function.

        Args:
            data: The received data item.
        """
        self.cb(self.topic, data)
