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

#include <yaml-cpp/yaml.h>
#include <sstream>
#include <string>
#include "InputCommand.hpp"

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

struct HIDDevice {
  std::string name;
  std::string path;
  HIDDeviceType type;

  mutable int file_descriptor;
};

struct HIDevicesConfig {
  std::vector<HIDDevice> devices;
};

}  // namespace holoscan::ops

template <>
struct YAML::convert<holoscan::ops::HIDevicesConfig> {
  static Node encode(const holoscan::ops::HIDevicesConfig& rhs) {
    Node node;
    for (const auto& device : rhs.devices) {
      Node device_node;
      device_node["name"] = device.name;
      device_node["path"] = device.path;
      device_node["type"] = encode_hid_device_type_as_string(device.type);
      node.push_back(device_node);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HIDevicesConfig& rhs) {
    if (!node.IsSequence()) return false;

    rhs.devices.clear();
    for (const auto& device_node : node) {
      holoscan::ops::HIDDevice device;
      if (!parse_hid_device_node(device_node, device)) { return false; }
      rhs.devices.push_back(device);
    }
    return true;
  }

  static bool parse_hid_device_node(const Node& node, holoscan::ops::HIDDevice& device) {
    try {
      device.name = node["name"].as<std::string>();
      device.path = node["path"].as<std::string>();
      if (!parse_hid_device_type(node["type"].as<std::string>(), device.type)) {
        return false;
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error parsing HIDevice node: {}", e.what());
      return false;
    }
    return true;
  }

  static std::string encode_hid_device_type_as_string(HIDDeviceType type) {
    switch (type) {
      case HIDDeviceType::JOYSTICK:
        return "joystick";
      case HIDDeviceType::KEYBOARD:
        return "keyboard";
      case HIDDeviceType::MOUSE:
        return "mouse";
      default:
        throw std::runtime_error("Unknown HIDevice type.");
    }
  }

  static bool parse_hid_device_type(const std::string& value, HIDDeviceType& type) {
    if (value == "joystick") {
      type = HIDDeviceType::JOYSTICK;
      return true;
    }
    if (value == "keyboard") {
      type = HIDDeviceType::KEYBOARD;
      return true;
    }
    if (value == "mouse") {
      type = HIDDeviceType::MOUSE;
      return true;
    }
    return false;
  }
};
