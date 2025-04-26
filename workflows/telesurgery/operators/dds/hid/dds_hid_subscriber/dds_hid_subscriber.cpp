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

#include "dds_hid_subscriber.hpp"
#include "dds/topic/find.hpp"

namespace holoscan::ops
{

  void DDSHIDSubscriberOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    // Change the output type to directly use std::vector<InputCommand>
    spec.output<std::vector<InputCommand>>("output");

    spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
    spec.param(hid_device_filters_,
               "hid_device_filters",
               "HID Device Filters",
               "HID Device Filters to capture HID events from",
               std::vector<std::string>());
  }

  void DDSHIDSubscriberOp::initialize()
  {
    HOLOSCAN_LOG_INFO("Initializing DDSHIDSubscriberOp");
    DDSOperatorBase::initialize();

    // Create the subscriber
    dds::sub::Subscriber subscriber(participant_);

    // Create the InputCommand topic
    auto topic = dds::topic::find<dds::topic::Topic<InputCommand>>(participant_, INPUT_COMMAND_TOPIC);
    if (topic == dds::core::null)
    {
      topic = dds::topic::Topic<InputCommand>(participant_, INPUT_COMMAND_TOPIC);
    }

    // Join the sanitized device filters with commas for the SQL-like IN clause
    if (hid_device_filters_.get().size() > 0)
    {
      std::string device_filter_string;
      for (const auto &device_filter : hid_device_filters_.get())
      {
        if (!device_filter_string.empty())
          device_filter_string += ",";
        device_filter_string += device_filter;
      }
      device_filter_string = "'" + device_filter_string + "'";
      HOLOSCAN_LOG_INFO("Device filters: {}", device_filter_string);

      dds::topic::ContentFilteredTopic<InputCommand> filtered_topic(
          topic,
          "FilteredInputCommand",
          dds::topic::Filter("device_name MATCH %0", {device_filter_string}));
      reader_ = dds::sub::DataReader<InputCommand>(
          subscriber, filtered_topic, qos_provider_.datareader_qos(reader_qos_.get()));
    }
    else
    {
      reader_ = dds::sub::DataReader<InputCommand>(
          subscriber, topic, qos_provider_.datareader_qos(reader_qos_.get()));
    }

  }

  void DDSHIDSubscriberOp::compute(InputContext &op_input, OutputContext &op_output,
                                   ExecutionContext &context)
  {
    dds::sub::LoanedSamples<InputCommand> commands = reader_.take();

    std::vector<InputCommand> valid_commands;
    for (const auto &command : commands)
    {
      if (command.info().valid())
      {
        valid_commands.push_back(command.data());
      }
    }

    // temporary convert InputCommand values to std::vector<string>
    std::vector<std::string> output_commands;
    for (const auto &command : valid_commands)
    {
      // Cast enums to int before converting to string
      auto value = command.device_name() + std::string(";") + std::to_string(static_cast<int>(command.device_type())) + std::string(";") + std::to_string(static_cast<int>(command.event_type())) + std::string(";") + std::to_string(command.number()) + std::string(";") + std::to_string(command.value());
      output_commands.push_back(value);
    }

    // join the vector of strings into a single string
    std::string output_commands_string;
    for (const auto &command : output_commands)
    {
      output_commands_string += command + std::string(":");
    }

    if (output_commands_string.length() > 0)
    {
      op_output.emit(output_commands_string, "output");
    }
  }

} // namespace holoscan::ops
