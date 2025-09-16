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

import os

import rti.connextdds as dds
from holoscan.core import Operator
from holoscan.core._core import OperatorSpec


class DDSSubscriberOp(Operator):
    def __init__(self, fragment, dds_domain_id, dds_topic, dds_topic_class, *args, **kwargs):
        """
        Initialize the DDSPublisherOp operator.

        Parameters:
        - dds_domain_id (int): DDS domain ID.
        - dds_topic (str): DDS topic for frames.
        - dds_topic_class (class): DDS class that represents the topic.
        """
        self.dds_domain_id = dds_domain_id
        self.dds_topic = dds_topic
        self.dds_topic_class = dds_topic_class
        self.dds_reader = None
        self.selector = None

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def start(self):
        dp = dds.DomainParticipant(domain_id=self.dds_domain_id)

        # Configure QoS to only keep latest sample
        qos = dds.DataReaderQos()
        qos.history.kind = dds.HistoryKind.KEEP_LAST
        qos.history.depth = 1  # Only keep 1 sample

        self.dds_reader = dds.DataReader(dds.Topic(dp, self.dds_topic, self.dds_topic_class), qos)
        self.selector = self.dds_reader.select().max_samples(1)
        print(f"NDDS_DISCOVERY_PEERS: {os.environ.get('NDDS_DISCOVERY_PEERS')}")
        print(f"Reading data from DDS: {self.dds_topic}:{self.dds_domain_id} => {self.dds_topic_class.__name__}")

    def compute(self, op_input, op_output, context):
        for o in self.selector.take_data():
            op_output.emit(o, "output")
