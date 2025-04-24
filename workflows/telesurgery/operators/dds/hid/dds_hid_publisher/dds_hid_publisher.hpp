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

#include <dds/pub/ddspub.hpp>

#include "dds_operator_base.hpp"
#include "InputCommand.hpp"

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

#include "hid_device.cpp"

namespace holoscan::ops {

/**
 * @brief Operator class to publish a hid stream to DDS.
 */
class DDSHIDPublisherOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSHIDPublisherOp, DDSOperatorBase)

  DDSHIDPublisherOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  void start() override;
  void stop() override;

 private:
  Parameter<std::string> writer_qos_;
  Parameter<HIDevicesConfig> hid_devices_;

  dds::pub::DataWriter<InputCommand> writer_ = dds::core::null;

  uint32_t frame_num_ = 0;

  std::map<std::string, HIDDevice>
      device_file_descriptors_;  // Sanitized device paths to file descriptors
  std::queue<std::tuple<HIDDevice, std::variant<js_event, input_event>, uint64_t>>
      event_buffer_;                   // Buffer for storing events
  std::thread event_thread_;           // Thread for reading events
  std::atomic<bool> running_;          // Flag to control the running state of the thread
  std::mutex buffer_mutex_;            // Mutex for synchronizing access to the event buffer
  std::condition_variable buffer_cv_;  // Condition variable for buffer synchronization

  // Variables for throttling
  std::chrono::milliseconds publish_interval_{};
  std::chrono::steady_clock::time_point last_publish_time_;

  // Message tracking variables
  std::atomic<uint64_t> total_messages_sent_ = 0;
  std::atomic<uint64_t> next_message_id_{1};  // Atomic for thread-safe message ID generation
  std::chrono::time_point<std::chrono::steady_clock> last_stats_time_ = std::chrono::steady_clock::now();
  uint64_t stats_interval_ms_ = 5000; // Print stats every 5 seconds

};

}  // namespace holoscan::ops
