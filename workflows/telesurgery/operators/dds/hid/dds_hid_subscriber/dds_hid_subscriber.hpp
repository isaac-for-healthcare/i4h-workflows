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

#include <dds/sub/ddssub.hpp>

#include "dds_operator_base.hpp"
#include "InputCommand.hpp"

namespace holoscan::ops {

// The RTI DDS generated InputCommand type cannot be used directly and therefore we need to define our own.
struct InputEvent
{
    HIDDeviceType device_type;
    int event_type;
    int number;
    int value;
};

/**
 * @brief Operator class to subscribe to a DDS hid stream.
 */
class DDSHIDSubscriberOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSHIDSubscriberOp, DDSOperatorBase)

  DDSHIDSubscriberOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> reader_qos_;
  Parameter<std::vector<std::string>> hid_device_filters_;

  dds::sub::DataReader<InputCommand> reader_ = dds::core::null;
};

}  // namespace holoscan::ops
