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

#include "hid.hpp"
#include "input_event.hpp"
#include "../dds_hid_common.hpp"

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
  Parameter<int> message_cap_;
  dds::pub::DataWriter<InputCommand> writer_ = dds::core::null;

  // Message tracking variables
  std::atomic<uint64_t> total_messages_sent_ = 0;
  std::atomic<uint64_t> next_message_id_{1};  // Atomic for thread-safe message ID generation

  // Statistics thread members
  std::thread stats_thread_;
  std::atomic<bool> stop_stats_thread_{false};

  // Statistics thread function
  void stats_printer_thread();
};

}  // namespace holoscan::ops
