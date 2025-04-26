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

#ifndef CAMERA_INFO_DDS_CAMERA_INFO_PUBLISHER_HPP
#define CAMERA_INFO_DDS_CAMERA_INFO_PUBLISHER_HPP


#include <dds/pub/ddspub.hpp>

#include "dds_operator_base.hpp"
#include "CameraInfo.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include <unordered_set>
#include <chrono>

namespace holoscan::ops {

/**
 * @brief Operator class to subscribe to a DDS hid stream.
 */
class DDSCameraInfoPublisherOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSCameraInfoPublisherOp, DDSOperatorBase)

  DDSCameraInfoPublisherOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> writer_qos_;
  Parameter<std::string> topic_;

  dds::pub::DataWriter<CameraInfo> writer_ = dds::core::null;

  uint32_t frame_num_ = 0;

};

}  // namespace holoscan::ops


#endif /* CAMERA_INFO_DDS_CAMERA_INFO_PUBLISHER_HPP */
