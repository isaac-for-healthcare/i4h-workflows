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

#include <yaml-cpp/yaml.h>
#include <sstream>
#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

enum class HIDDeviceType {
  JOYSTICK,
  KEYBOARD,
  MOUSE,
};

struct HumanInterfaceDevice {
  std::string name;
  std::string path;
  HIDDeviceType type;

  mutable int file_descriptor;
};

struct HumanInterfaceDevicesConfig {
  std::vector<HumanInterfaceDevice> devices;
};

}  // namespace holoscan::ops

template <>
struct YAML::convert<holoscan::ops::HumanInterfaceDevicesConfig> {
  static Node encode(const holoscan::ops::HumanInterfaceDevicesConfig& rhs) {
    Node node;
    for (const auto& device : rhs.devices) {
      Node device_node;
      device_node["name"] = device.name;
      device_node["path"] = device.path;
      device_node["type"] = encode_human_interface_device_type_as_string(device.type);
      node.push_back(device_node);
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::ops::HumanInterfaceDevicesConfig& rhs) {
    if (!node.IsSequence()) return false;

    rhs.devices.clear();
    for (const auto& device_node : node) {
      holoscan::ops::HumanInterfaceDevice device;
      if (!parse_human_interface_device_node(device_node, device)) { return false; }
      rhs.devices.push_back(device);
    }
    return true;
  }

  static bool parse_human_interface_device_node(const Node& node, holoscan::ops::HumanInterfaceDevice& device) {
    try {
      device.name = node["name"].as<std::string>();
      device.path = node["path"].as<std::string>();
      if (!parse_human_interface_device_type(node["type"].as<std::string>(), device.type)) {
        return false;
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error parsing HumanInterfaceDevice node: {}", e.what());
      return false;
    }
    return true;
  }

  static std::string encode_human_interface_device_type_as_string(holoscan::ops::HIDDeviceType type) {
    switch (type) {
      case holoscan::ops::HIDDeviceType::JOYSTICK:
        return "joystick";
      case holoscan::ops::HIDDeviceType::KEYBOARD:
        return "keyboard";
      case holoscan::ops::HIDDeviceType::MOUSE:
        return "mouse";
      default:
        throw std::runtime_error("Unknown HumanInterfaceDevice type.");
    }
  }

  static bool parse_human_interface_device_type(const std::string& value, holoscan::ops::HIDDeviceType& type) {
    if (value == "joystick") {
      type = holoscan::ops::HIDDeviceType::JOYSTICK;
      return true;
    }
    if (value == "keyboard") {
      type = holoscan::ops::HIDDeviceType::KEYBOARD;
      return true;
    }
    if (value == "mouse") {
      type = holoscan::ops::HIDDeviceType::MOUSE;
      return true;
    }
    return false;
  }
};
