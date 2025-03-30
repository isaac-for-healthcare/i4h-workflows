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

import queue
import time
import unittest

import rti.connextdds as dds  # noqa: F401
import rti.idl as idl
from dds.subscriber import Subscriber, SubscriberWithCallback, SubscriberWithQueue

from helpers import requires_rti


@idl.struct
class _TestData:
    value: int = 0
    message: str = ""


class _TestSubscriber(Subscriber):
    """Concrete implementation of Subscriber for testing."""

    def consume(self, data) -> None:
        return data


@requires_rti
class TestDDSSubscriber(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.domain_id = 0
        self.topic_name = "test_topic"

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.topic = dds.Topic(self.participant, self.topic_name, _TestData)
        self.writer = dds.DataWriter(self.participant.implicit_publisher, self.topic)

        self.subscriber = _TestSubscriber(topic=self.topic_name, cls=_TestData, period=0.1, domain_id=self.domain_id)

        time.sleep(1.0)

    def tearDown(self):
        """Clean up after each test method"""
        if hasattr(self, "subscriber"):
            self.subscriber.stop()
        if hasattr(self, "writer"):
            self.writer.close()
        if hasattr(self, "topic"):
            self.topic.close()
        if hasattr(self, "participant"):
            self.participant.close()
        time.sleep(0.1)

    def test_init(self):
        """Test subscriber initialization"""
        self.assertEqual(self.subscriber.topic, self.topic_name)
        self.assertEqual(self.subscriber.cls, _TestData)
        self.assertEqual(self.subscriber.period, 0.1)
        self.assertEqual(self.subscriber.domain_id, self.domain_id)
        self.assertTrue(self.subscriber.add_to_queue)
        self.assertIsInstance(self.subscriber.data_q, queue.Queue)

    def test_start_stop(self):
        """Test start and stop functionality with real DDS"""
        self.subscriber.start()
        time.sleep(0.5)

        test_data = _TestData(value=42, message="Hello DDS")
        self.writer.write(test_data)

        for data in self.subscriber.dds_reader.take_data():
            self.assertEqual(data, test_data)
        self.subscriber.stop()
        self.assertIsNone(self.subscriber.stop_event)


@requires_rti
class TestSubscriberWithQueue(unittest.TestCase):
    def setUp(self):
        self.domain_id = 100
        self.topic_name = "test_topic"
        self.period = 0.1

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.topic_dds = dds.Topic(self.participant, self.topic_name, _TestData)
        self.writer = dds.DataWriter(self.participant.implicit_publisher, self.topic_dds)

        self.subscriber = SubscriberWithQueue(
            domain_id=self.domain_id, topic=self.topic_name, cls=_TestData, period=self.period
        )

    def tearDown(self):
        if hasattr(self, "subscriber"):
            self.subscriber.stop()
        if hasattr(self, "writer"):
            self.writer.close()
        if hasattr(self, "topic_dds"):
            self.topic_dds.close()
        if hasattr(self, "participant"):
            self.participant.close()

    def test_read_data(self):
        self.subscriber.start()
        time.sleep(1.0)

        self.assertIsNotNone(self.subscriber.dds_reader, "DDS reader not created")

        test_data = _TestData(value=42, message="Hello DDS")
        self.writer.write(test_data)

        max_retries = 5
        for _ in range(max_retries):
            data = self.subscriber.read_data()
            if data is not None:
                self.assertEqual(data.value, test_data.value)
                self.assertEqual(data.message, test_data.message)
                break
            time.sleep(0.2)
        else:
            self.fail("No data received after multiple retries")


@requires_rti
class TestSubscriberWithCallback(unittest.TestCase):
    def setUp(self):
        self.domain_id = 200  # Use a unique domain ID
        self.topic_name = "callback_test_topic"
        self.callback_called = False
        self.received_data = None

        # Define callback function
        def test_callback(topic: str, data: _TestData) -> None:
            self.callback_called = True
            self.received_data = data

        self.participant = dds.DomainParticipant(domain_id=self.domain_id)
        self.topic = dds.Topic(self.participant, self.topic_name, _TestData)
        self.writer = dds.DataWriter(self.participant.implicit_publisher, self.topic)

        self.subscriber = SubscriberWithCallback(
            cb=test_callback, domain_id=self.domain_id, topic=self.topic_name, cls=_TestData, period=0.1
        )

    def tearDown(self):
        if hasattr(self, "subscriber"):
            self.subscriber.stop()
        if hasattr(self, "writer"):
            self.writer.close()
        if hasattr(self, "topic"):
            self.topic.close()
        if hasattr(self, "participant"):
            self.participant.close()
        time.sleep(0.1)

    def test_callback_with_dds(self):
        # Start the subscriber
        self.subscriber.start()
        time.sleep(1.0)  # Allow time for discovery

        # Verify DDS entities are properly created
        self.assertIsNotNone(self.subscriber.dds_reader, "DDS reader not created")

        # Write test data
        test_data = _TestData(value=42, message="Callback Test")
        self.writer.write(test_data)

        # Wait for callback to be processed
        max_retries = 5
        for _ in range(max_retries):
            if self.callback_called:
                break
            time.sleep(0.2)
        else:
            self.fail("Callback was not called after multiple retries")

        # Verify callback data
        self.assertIsNotNone(self.received_data, "No data received in callback")
        self.assertEqual(self.received_data.value, test_data.value)
        self.assertEqual(self.received_data.message, test_data.message)


if __name__ == "__main__":
    unittest.main()
