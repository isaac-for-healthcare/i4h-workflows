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

namespace holoscan::doc::GenericHIDInterface {

PYDOC(GenericHIDInterface, R"doc(
Generic HID Interface operator.
)doc")

// PyGenericHIDInterface Constructor
PYDOC(GenericHIDInterface_python, R"doc(
Generic HID Interface operator.

Parameters
----------
human_interface_devices: holoscan.core.HumanInterfaceDevicesConfig, optional
    Human interface devices configuration.
name : str, optional
    The name of the operator.
name : str, optional
    The name of the operator.
)doc")

PYDOC(HIDDeviceType, R"doc(
Enum defining the type of HID device.

Values
------
JOYSTICK
KEYBOARD
MOUSE
)doc")

PYDOC(InputCommand, R"doc(
Structure representing a single HID event command.

Attributes
----------
device_name : str
    Name of the originating HID device.
device_type : HIDDeviceType
    Type of the originating HID device.
event_type : int
    Type identifier for the event (e.g., button press, axis movement).
number : int
    Identifier for the specific button or axis.
value : int
    Value associated with the event (e.g., axis position, button state).
hid_capture_timestamp : int
    Timestamp (ns) when the event was captured by the publisher.
hid_publish_timestamp : int
    Timestamp (ns) when the event was published via DDS.
hid_to_sim_timestamp: int
    Timestamp (ns) when the event was sent to the simulation.
hid_receive_timestamp : int
    Timestamp (ns) when the event was received by this subscriber.
hid_process_timestamp: int
    Timestamp (ns) when the event was processed by the Sim controller.
message_id : int
    Unique identifier for the message.
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

PYDOC(register_types, R"doc(
Register custom types used by the operator with the Holoscan emitter/receiver registry.
)doc")

}  // namespace holoscan::doc::DDSHIDSubscriberOp
