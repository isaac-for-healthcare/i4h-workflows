# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import rti.connextdds as dds  # noqa: F401

from holoscan.core import Operator, OperatorSpec
from idl.CameraInfo.CameraInfo import CameraInfo


class CameraInfoPublisherOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logging.getLogger(__name__)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

        spec.param("qos_provider")
        spec.param("participant_qos")
        spec.param("domain_id")
        spec.param("writer_qos")
        spec.param("topic")

    def initialize(self):
        Operator.initialize(self)
        self._logger.info(
            f"Initializing: qos_provider={self.qos_provider}, participant_qos={self.participant_qos}, domain_id={self.domain_id}, writer_qos={self.writer_qos}, topic={self.topic}"
        )
        self._provider = dds.QosProvider(self.qos_provider)
        self._participant = dds.DomainParticipant(
            domain_id=self.domain_id,
            qos=self._provider.participant_qos_from_profile(self.participant_qos),
        )
        self._topic = dds.Topic(self._participant, self.topic, CameraInfo)
        self._publisher = dds.Publisher(self._participant)
        self._writer = dds.DataWriter(
            self._publisher,
            self._topic,
            qos=self._provider.datawriter_qos_from_profile(self.writer_qos),
        )

    def compute(self, op_input, op_output, context):
        camera_info = op_input.receive("input")
        if camera_info is None:
            self._logger.error("No camera info received")
            return
        self._writer.write(camera_info)
