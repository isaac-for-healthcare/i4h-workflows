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
import random
import time
import unittest
from unittest.mock import MagicMock, patch

# Add the DDS module to path
from holohub.operators.dds.publisher import DDSPublisherOp
from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

from .common import TestData


class TestDataProviderOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.queue = queue.Queue()

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        data = self.queue.get()
        op_output.emit(data, "output")

    def enqueue(self, data: TestData):
        self.queue.put(data)


class TestDDSPublisherApplication(Application):
    def __init__(self, dds_domain_id, dds_topic):
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.dds_topic_class = TestData
        self.data_provider_op = None
        super().__init__()

    def compose(self):
        self.data_provider_op = TestDataProviderOp(self, name="data_provider_op", count=CountCondition(self, 1))
        publisher_op = DDSPublisherOp(
            self,
            name="publisher_op",
            dds_domain_id=self.dds_domain_id,
            dds_topic=self.dds_topic,
            dds_topic_class=self.dds_topic_class,
            count=CountCondition(self, 1),
        )

        self.add_flow(self.data_provider_op, publisher_op)

    def enqueue(self, data: TestData):
        self.data_provider_op.enqueue(data)


class TestDDSPublisher(unittest.TestCase):
    def test_dds_publisher(self):
        # mock the dds writer
        with (
            patch("rti.connextdds.DomainParticipant") as _,
            patch("rti.connextdds.DataWriter") as mock_dds_writer,
            patch("rti.connextdds.Topic") as mock_topic,
        ):
            mock_writer_instance = MagicMock()
            mock_writer_instance.write = MagicMock()
            mock_dds_writer.return_value = mock_writer_instance
            mock_topic.return_value = MagicMock()

            app = TestDDSPublisherApplication(
                dds_domain_id=1,
                dds_topic="test_topic",
            )
            future = app.run_async()

            while app.data_provider_op is None:
                print("Waiting for data_provider_op to be initialized")
                time.sleep(0.1)

            data = TestData(data=random.randint(0, 100))
            app.enqueue(data)
            future.result(timeout=5)

            mock_writer_instance.write.assert_called_once_with(data)
