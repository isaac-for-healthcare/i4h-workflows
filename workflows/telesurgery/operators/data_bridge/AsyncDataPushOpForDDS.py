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
import queue
import time

from dds_camera_info_publisher._dds_camera_info_publisher import CameraInfo
from holoscan.conditions import AsynchronousCondition, AsynchronousEventState
from holoscan.core import Operator, OperatorSpec, Tensor


class AsyncDataPushForDDS(Operator):
    """An asynchronous operator that allows pushing data from external sources into a Holoscan pipeline.

    This operator implements a producer-consumer pattern where data can be pushed from outside
    the pipeline using the push_data() method, and the operator will emit this data through
    its output port when available.

    The operator uses a queue for thread-safe data transfer between the external source and
    the pipeline.

    Attributes:
        _queue: A thread-safe queue for storing data to be processed.
    """

    def __init__(self, fragment, max_queue_size: int = 0, condition: AsynchronousCondition = None, *args, **kwargs):
        """Initialize the AsyncDataPushOp.

        Args:
            fragment: The Holoscan fragment this operator belongs to.
            max_queue_size: Maximum size of the internal queue. If 0, the queue size is unlimited.
            *args: Additional positional arguments passed to the parent Operator.
            **kwargs: Additional keyword arguments passed to the parent Operator.
        """
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._logger = logging.getLogger(__name__)
        self.condition = condition
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the operator's input and output ports.

        Args:
            spec: The OperatorSpec object used to define the operator's interface.
        """
        spec.output("image")
        spec.output("camera_info")

    def start(self):
        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def stop(self):
        self.condition.event_state = AsynchronousEventState.EVENT_NEVER

    def compute(self, op_input, op_output, context):
        """Process and emit data when available.

        This method waits for data to be pushed via push_data(), then emits it through
        the output port. It blocks until data becomes available in the queue.

        Args:
            op_input: The input data (not used in this operator).
            op_output: The output port to emit data through.
            context: The execution context.
        """
        data = self._queue.get()
        enter_timestamp = time.time_ns()

        camera_info = CameraInfo()
        camera_info.joint_names = list(data["joint_names"])
        camera_info.joint_positions = data["joint_positions"].tolist()
        camera_info.width = data["size"][1]
        camera_info.height = data["size"][0]
        camera_info.channels = data["size"][2]
        camera_info.frame_num = data["frame_num"]
        camera_info.frame_size = data["image"].size

        if data["last_hid_event"] is not None:
            camera_info.message_id = data["last_hid_event"].message_id
            camera_info.hid_capture_timestamp = data["last_hid_event"].hid_capture_timestamp
            camera_info.hid_publish_timestamp = data["last_hid_event"].hid_publish_timestamp
            camera_info.hid_receive_timestamp = data["last_hid_event"].hid_receive_timestamp
            camera_info.hid_to_sim_timestamp = data["last_hid_event"].hid_to_sim_timestamp
            camera_info.hid_process_timestamp = data["last_hid_event"].hid_process_timestamp

        camera_info.video_acquisition_timestamp = data["video_acquisition_timestamp"]
        camera_info.video_data_bridge_enter_timestamp = enter_timestamp
        camera_info.video_data_bridge_emit_timestamp = time.time_ns()

        op_output.emit({"": Tensor.as_tensor(data["image"])}, "image")
        op_output.emit(camera_info, "camera_info", "CameraInfo")

        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def push_data(self, data):
        """Push data into the operator for processing.

        This method is called from outside the pipeline to provide data for processing.
        It adds the data to the internal queue, which will be processed by the compute()
        method.

        Args:
            data: The data to be processed and emitted through the output port.

        Note:
            If the queue is full (when max_queue_size > 0), this method will block
            until space becomes available.
        """
        self._queue.put(data)
        self.condition.event_state = AsynchronousEventState.EVENT_DONE
