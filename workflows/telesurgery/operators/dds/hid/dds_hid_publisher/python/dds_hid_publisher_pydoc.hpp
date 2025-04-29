/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc::DDSHIDPublisherOp {

PYDOC(DDSHIDPublisherOp, R"doc(
DDS HID Publisher operator.
)doc")

// PyDDSHIDPublisherOp Constructor
PYDOC(DDSHIDPublisherOp_python, R"doc(
DDS HID Publisher operator.

Parameters
----------
qos_provider: str, optional
    URI for the QoS Provider
participant_qos: str, optional
    QoS profile for the domain participant
domain_id : int, optional
    DDS domain to use.
writer_qos: str, optional
    QoS profile for the data writer
name : str, optional
    The name of the operator.
hid_devices : list, optional
    List of HID devices to scan & publish. Each device is a dictionary with the following keys:
    - name: str, name of the device
    - path: str, path to the device
    - type: str, type of the device (joystick, keyboard, mouse)
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::DDSHIDPublisherOp
