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


#include <unistd.h>
#include <fcntl.h>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <linux/joystick.h>
#include <linux/input.h>
#include <tuple>
#include <variant>
#include <chrono>

#include "hid.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to interface with human interface devices.
 */
class GenericHIDInterface : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GenericHIDInterface)

  GenericHIDInterface() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  void start() override;
  void stop() override;

 private:
  Parameter<HumanInterfaceDevicesConfig> human_interface_devices_;
  Parameter<int> simulation_rate_ms_;

  std::atomic<uint64_t> total_events_{0};
  std::atomic<uint64_t> total_events_emitted_{0};
  std::chrono::time_point<std::chrono::steady_clock> last_stats_time_ = std::chrono::steady_clock::now();
  std::map<std::string, HumanInterfaceDevice>
      device_file_descriptors_;  // Sanitized device paths to file descriptors
  std::queue<std::tuple<HumanInterfaceDevice, std::variant<js_event, input_event>, uint64_t>>
      event_buffer_;                   // Buffer for storing events
  std::thread event_thread_;           // Thread for reading events
  std::atomic<bool> running_;          // Flag to control the running state of the thread
};

}  // namespace holoscan::ops
