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

#include "dds_hid_publisher.hpp"

#include <chrono>
#include <dds/topic/find.hpp>
#include <map>
#include <thread>

namespace holoscan::ops {

void DDSHIDPublisherOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.input<std::vector<InputEvent>>("input");

  spec.param(writer_qos_, "writer_qos", "Writer QoS", "Data Writer QoS Profile", std::string());
  spec.param(message_cap_, "message_cap", "Message Cap", "Maximum number of messages to publish", -1);
}

void DDSHIDPublisherOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the publisher
  dds::pub::Publisher publisher(participant_);

  // Create the InputCommand topic
  auto topic = dds::topic::find<dds::topic::Topic<InputCommand>>(participant_, INPUT_COMMAND_TOPIC);
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<InputCommand>(participant_, INPUT_COMMAND_TOPIC);
  }

  // Create the writer for the InputCommand
  writer_ = dds::pub::DataWriter<InputCommand>(
      publisher, topic, qos_provider_.datawriter_qos(writer_qos_.get()));
}

void DDSHIDPublisherOp::start() {
  stats_thread_ = std::thread(&DDSHIDPublisherOp::stats_printer_thread, this);
}

void DDSHIDPublisherOp::stop() {
  stop_stats_thread_.store(true);
  if (stats_thread_.joinable()) {
    stats_thread_.join();
  }
}

void DDSHIDPublisherOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  if (message_cap_.get() > 0 && total_messages_sent_.load() >= message_cap_.get()) {
    return;
  }

  auto now = std::chrono::system_clock::now();

  auto input_events = op_input.receive<std::vector<InputEvent>>("input");

  if (!input_events) {
    HOLOSCAN_LOG_WARN("No input events received");
    return;
  }

  auto events = input_events.value();

  if (!events.empty()) {
    for (auto const& input_event : events) {
      InputCommand input_command;
      input_command.device_type(holoscan::ops::holoscan_hid_device_type_to_input_command_device_type[input_event.device_type]);
      input_command.event_type(input_event.event_type);
      input_command.number(input_event.number);
      input_command.value(input_event.value);
      input_command.message_id(next_message_id_);
      input_command.hid_capture_timestamp(input_event.hid_capture_timestamp);

      uint64_t publish_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();
      input_command.hid_publish_timestamp(publish_time);
      writer_.write(input_command);
      total_messages_sent_++;
      next_message_id_++;
    }
  }
}

void DDSHIDPublisherOp::stats_printer_thread() {
  HOLOSCAN_LOG_INFO("Stats printer thread started.");
  while (!stop_stats_thread_.load()) {
    auto wake_up_time = std::chrono::system_clock::now() + std::chrono::seconds(3);
    while (std::chrono::system_clock::now() < wake_up_time) {
      if (stop_stats_thread_.load()) {
        HOLOSCAN_LOG_INFO("Stats printer thread stopping.");
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Check again after waking up, before printing
    if (stop_stats_thread_.load()) {
       HOLOSCAN_LOG_INFO("Stats printer thread stopping.");
       return;
    }

    HOLOSCAN_LOG_INFO("Total HID events published: {}", total_messages_sent_.load());
  }
}

}  // namespace holoscan::ops
