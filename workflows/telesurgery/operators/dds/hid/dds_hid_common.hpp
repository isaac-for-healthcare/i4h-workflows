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

#include <unordered_map>
#include "hid.hpp"          // For holoscan::ops::HIDDeviceType
#include "InputCommand.hpp" // For ::HIDDeviceType

namespace holoscan::ops {

// Map between Holoscan HIDDeviceType and the DDS InputCommand HIDDeviceType
static std::unordered_map<holoscan::ops::HIDDeviceType, ::HIDDeviceType> holoscan_hid_device_type_to_input_command_device_type = {
  {holoscan::ops::HIDDeviceType::JOYSTICK, ::HIDDeviceType::JOYSTICK},
  {holoscan::ops::HIDDeviceType::KEYBOARD, ::HIDDeviceType::KEYBOARD},
  {holoscan::ops::HIDDeviceType::MOUSE, ::HIDDeviceType::MOUSE},
};

// Create a function for reverse mapping
static holoscan::ops::HIDDeviceType input_command_device_type_to_holoscan_hid_device_type(::HIDDeviceType device_type) {
  for (const auto& [key, value] : holoscan_hid_device_type_to_input_command_device_type) {
    if (value == device_type) {
      return key;
    }
  }

  throw std::runtime_error("Invalid HIDDeviceType");
}

} // namespace holoscan::ops
