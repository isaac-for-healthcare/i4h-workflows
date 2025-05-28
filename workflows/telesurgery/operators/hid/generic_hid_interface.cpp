/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "generic_hid_interface.hpp"
#include "input_event.hpp"
#include "hid.hpp"

namespace holoscan::ops
{

    void GenericHIDInterface::setup(OperatorSpec &spec)
    {
        spec.output<std::vector<InputEvent>>("output");
        spec.param(human_interface_devices_, "human_interface_devices", "Human Interface Devices", "Human Interface Devices", HumanInterfaceDevicesConfig());
        spec.param(simulation_rate_ms_, "simulation_rate_ms", "Simulation Rate", "Simulation Rate", 100);
    }

    void GenericHIDInterface::initialize()
    {
        register_converter<holoscan::ops::HumanInterfaceDevicesConfig>();
        Operator::initialize();

        // Open the device
        for (const auto &device : human_interface_devices_.get().devices)
        {
            if (device.path == "/simulation")
            {
                device.file_descriptor = -1;
            }
            else
            {
                device.file_descriptor = open(device.path.c_str(), O_RDONLY | O_NONBLOCK);
                if (device.file_descriptor == -1)
                {
                    throw std::runtime_error("Could not open device: " + device.path);
                }
            }
            device_file_descriptors_[device.name] = device;
        }
    }

    void GenericHIDInterface::start()
    {
        running_ = true;
        event_thread_ = std::thread([this]()
                                    {
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
        auto capture_time_epoch =
              std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (device.path == "/simulation") {
          // randomly generate events for testing
          // type is always joystick
          // for event type = 1, number can be 0, 1, 2 ,3 ,5, 6, 8
          // for event type = 2, number can be 0, 1, 2, 3, 4, 5, 6, 7
          // value can be between -32767 and 32767
          auto event = js_event{};
          event.type = rand() % 2 + 1;
          event.number = rand() % 8;
          event.value = rand() % 65535 - 32767;

          event_buffer_.push(std::make_tuple(device, event, capture_time_epoch));
          std::this_thread::sleep_for(std::chrono::milliseconds(simulation_rate_ms_.get()));
        } else {
          while (read(device.file_descriptor, event_ptr, event_size) > 0) {
            event_buffer_.push(std::make_tuple(device, event, capture_time_epoch));
          }
        }
        total_events_++;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } });
    }

    void GenericHIDInterface::stop()
    {
        running_ = false;
        if (event_thread_.joinable())
        {
            event_thread_.join();
        }
        for (auto &[device_name, device] : device_file_descriptors_)
        {
            close(device.file_descriptor);
        }
    }

    void GenericHIDInterface::compute(InputContext &op_input, OutputContext &op_output,
                                      ExecutionContext &context)
    {
        std::map<std::pair<std::string, uint8_t>, InputEvent> latest_events_in_batch;
        while (!event_buffer_.empty())
        {
            auto [device, current_event, capture_time_epoch] = event_buffer_.front();
            event_buffer_.pop();

            InputEvent event;
            event.device_type = device.type;
            event.device_name = device.name;
            event.hid_capture_timestamp = capture_time_epoch;
            switch (device.type)
            {
            case HIDDeviceType::JOYSTICK:
            {
                event.event_type = static_cast<uint8_t>(std::get<js_event>(current_event).type);
                event.number = static_cast<uint8_t>(std::get<js_event>(current_event).number);
                event.value = std::get<js_event>(current_event).value;
                break;
            }
            case HIDDeviceType::MOUSE:
            case HIDDeviceType::KEYBOARD:
            {
                const auto &input_ev = std::get<input_event>(current_event);
                event.event_type = static_cast<uint8_t>(input_ev.type);
                event.number = static_cast<uint8_t>(input_ev.code);
                event.value = input_ev.value;
                break;
            }
            }
            latest_events_in_batch[{event.device_name, event.number}] = event;
        }

        if (!latest_events_in_batch.empty())
        {
            std::vector<InputEvent> events;
            events.reserve(latest_events_in_batch.size());
            for (const auto &[key, event] : latest_events_in_batch)
            {
                events.push_back(event);
            }
            op_output.emit(events, "output");
            total_events_emitted_ += events.size();
        }

        // Print stats every 3 seconds
        auto current_time = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stats_time_).count();
        if (elapsed >= 5000) {
          HOLOSCAN_LOG_INFO("Total events emitted: {}", total_events_emitted_.load());
          last_stats_time_ = current_time;
        }
    }
} // namespace holoscan::ops
