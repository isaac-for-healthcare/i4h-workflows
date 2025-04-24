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

namespace holoscan::ops {

void DDSHIDPublisherOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  spec.param(writer_qos_, "writer_qos", "Writer QoS", "Data Writer QoS Profile", std::string());
  spec.param(hid_devices_,
             "hid_devices",
             "HID Devices",
             "HID Devices for the DDS HID Stream",
             HIDevicesConfig());
}

void DDSHIDPublisherOp::initialize() {
  register_converter<holoscan::ops::HIDevicesConfig>();
  DDSOperatorBase::initialize();

  // Open the device
  for (const auto& device : hid_devices_.get().devices) {
    if (device.path == "/simulation") {
      device.file_descriptor = -1;
    }
    else {
      device.file_descriptor = open(device.path.c_str(), O_RDONLY | O_NONBLOCK);
      if (device.file_descriptor == -1) {
        throw std::runtime_error("Could not open device: " + device.path);
      }
    }
    device_file_descriptors_[device.name] = device;
  }

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
  running_ = true;
  event_thread_ = std::thread([this]() {
    std::variant<js_event, input_event> event;
    while (running_) {
      for (const auto& [device_name, device] : device_file_descriptors_) {
        size_t event_size = 0;
        void* event_ptr = nullptr;

        // Set up the appropriate event type and size based on device type
        switch (device.type) {
          case HIDDeviceType::JOYSTICK: {
            event = js_event{};
            event_size = sizeof(js_event);
            event_ptr = &std::get<js_event>(event);
            break;
          }
          case HIDDeviceType::KEYBOARD:
          case HIDDeviceType::MOUSE: {
            event = input_event{};
            event_size = sizeof(input_event);
            event_ptr = &std::get<input_event>(event);
            break;
          }
        }

        if (device.path == "/simulation") {
          // randomly generate events for testing
          // type is always joystick
          // for event type = 1, number can be 0, 1,2 ,3 ,5, 6, 8
          // for event type = 2, number can be 0, 1, 2, 3, 4, 5, 6, ,7
          // value can be between -32767 and 32767
          auto event = js_event{};
          event.type = rand() % 2 + 1;
          event.number = rand() % 8;
          event.value = rand() % 65535 - 32767;
          auto capture_time_epoch =
              std::chrono::high_resolution_clock::now().time_since_epoch().count();
          event_buffer_.push(std::make_tuple(device, event, capture_time_epoch));
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          HOLOSCAN_LOG_INFO("Simulated event: type={}, number={}, value={}", event.type, event.number,
                          event.value);
        } else {
          while (read(device.file_descriptor, event_ptr, event_size) > 0) {
            // std::lock_guard<std::mutex> lock(buffer_mutex_);
            // capture timestamp in unix timestamp
            auto capture_time_epoch =
                std::chrono::high_resolution_clock::now().time_since_epoch().count();
            event_buffer_.push(std::make_tuple(device, event, capture_time_epoch));
            // buffer_cv_.notify_one();
          }
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
}

void DDSHIDPublisherOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  auto now = std::chrono::steady_clock::now();

  std::map<std::pair<std::string, uint8_t>, InputCommand> latest_commands_in_batch;

  {
    while (!event_buffer_.empty()) {
      auto [device, current_event, capture_time_epoch] = event_buffer_.front();
      event_buffer_.pop();

      InputCommand command;
      command.device_type(device.type);
      command.device_name(device.name);
      command.hid_capture_timestamp(capture_time_epoch);

      switch (device.type) {
        case HIDDeviceType::JOYSTICK: {
          command.event_type(static_cast<uint8_t>(std::get<js_event>(current_event).type));
          command.number(static_cast<uint8_t>(std::get<js_event>(current_event).number));
          command.value(std::get<js_event>(current_event).value);
          break;
        }
        case HIDDeviceType::MOUSE:
        case HIDDeviceType::KEYBOARD: {
          const auto& input_ev = std::get<input_event>(current_event);
          command.event_type(static_cast<uint8_t>(input_ev.type));
          command.number(static_cast<uint8_t>(input_ev.code));
          command.value(input_ev.value);
          break;
        }
      }
      latest_commands_in_batch[{command.device_name(), command.number()}] = command;
    }
  }

  if (!latest_commands_in_batch.empty()) {
    auto publish_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::high_resolution_clock::now().time_since_epoch())
                                    .count();

    for (auto const& [key, command_to_send] : latest_commands_in_batch) {
      InputCommand final_command = command_to_send;
      final_command.hid_publish_timestamp(publish_timestamp_ns);
      final_command.message_id(next_message_id_);
      // HOLOSCAN_LOG_INFO("Publishing final command: Device={}, Number{}", key.first, key.second);
      writer_.write(final_command);
      total_messages_sent_++;
      next_message_id_++;
    }
    // HOLOSCAN_LOG_INFO("Current batch size: {}", latest_commands_in_batch.size());
    last_publish_time_ = now;
  }

  auto current_time = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stats_time_)
          .count();

  if (elapsed >= stats_interval_ms_) {
    HOLOSCAN_LOG_INFO("=== InputCommand Publisher Statistics ===");
    HOLOSCAN_LOG_INFO("Total InputCommand messages sent: {}", total_messages_sent_.load());
    HOLOSCAN_LOG_INFO("Next message ID: {}", next_message_id_.load());
    HOLOSCAN_LOG_INFO("=========================================");

    last_stats_time_ = current_time;
  }
}

void DDSHIDPublisherOp::stop() {
  running_ = false;
  buffer_cv_.notify_all();
  if (event_thread_.joinable()) { event_thread_.join(); }
  for (auto& [device_name, device] : device_file_descriptors_) { close(device.file_descriptor); }
}

}  // namespace holoscan::ops
