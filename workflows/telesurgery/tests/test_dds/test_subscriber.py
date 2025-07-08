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

import random
import time
import unittest
from unittest.mock import MagicMock, patch

# Add the DDS module to path
from holohub.operators.dds.subscriber import DDSSubscriberOp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from .common import TestData


class AssertionOp(Operator):
    def __init__(self, fragment, assertion_callback, *args, **kwargs):
        self.assertion_callback = assertion_callback
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("input")
        self.assertion_callback(data)


class TestDDSSubscriberApplication(Application):
    def __init__(self, dds_domain_id, dds_topic, assertion_callback):
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.dds_topic_class = TestData
        self.assertion_op = None
        self.assertion_callback = assertion_callback
        super().__init__()

    def compose(self):
        subscriber_op = DDSSubscriberOp(
            self,
            name="subscriber_op",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=self.dds_topic_class,
            count=CountCondition(self, 1),
        )
        self.assertion_op = AssertionOp(self, name="assertion_op", assertion_callback=self.assertion_callback)

        self.add_flow(subscriber_op, self.assertion_op)


class TestDDSSubscriber(unittest.TestCase):
    def test_dds_subscriber(self):
        self.test_data = TestData(data=random.randint(0, 100))
        # mock the dds reader - patch the actual rti.connextdds module
        with (
            patch("rti.connextdds.DomainParticipant") as mock_domain_participant,
            patch("rti.connextdds.DataReader") as mock_dds_reader,
            patch("rti.connextdds.Topic") as mock_topic,
        ):
            mock_reader_instance = MagicMock()
            mock_reader_instance.take_data = MagicMock()
            mock_reader_instance.take_data.return_value = [self.test_data]
            mock_dds_reader.return_value = mock_reader_instance
            mock_topic.return_value = MagicMock()
            mock_domain_participant.return_value = MagicMock()

            app = TestDDSSubscriberApplication(
                dds_domain_id=1,
                dds_topic="test_topic",
                assertion_callback=self.assertion_callback,
            )
            future = app.run_async()

            while app.assertion_op is None:
                print("Waiting for assertion_op to be initialized")
                time.sleep(0.1)

            future.result(timeout=5)

            mock_reader_instance.take_data.assert_called_once()

    def assertion_callback(self, data):
        self.assertEqual(data, self.test_data)
