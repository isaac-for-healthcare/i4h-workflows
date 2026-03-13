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

#include <holoscan/holoscan.hpp>
#include <cuda.h>
#include <streamlift_def.h>
#include <streamlift_server.h>

namespace holoscan::ops {
class StreamLiftDownSamplerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamLiftDownSamplerOp)

  StreamLiftDownSamplerOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&, holoscan::ExecutionContext&) override;
  void stop() override;

 private:
  Parameter<uint32_t> cuda_device_ordinal_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<std::string> output_type_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;

  // Others
  Streamlift_CudaBuffer bufIn;
  Streamlift_CudaBuffer bufOut;
  Streamlift_Server_Config serverConfig;
  Streamlift_RuntimeConfig serverRuntimeConfig;
  Streamlift_Session serverSession = nullptr;

  uint32_t frame_idx = 0u;
};
}  // namespace holoscan::ops
