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
#include <streamlift_client_cuda.h>

namespace holoscan::ops {
class StreamLiftUpSamplerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamLiftUpSamplerOp)

  StreamLiftUpSamplerOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext&, holoscan::ExecutionContext&) override;
  void stop() override;

 private:
  Parameter<uint32_t> cuda_device_ordinal_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;

  // Others
  Streamlift_CudaBuffer bufIn;
  Streamlift_CudaBuffer bufOut;
  Streamlift_Client_Config clientConfig;
  Streamlift_RuntimeConfig clientRuntimeConfig;
  Streamlift_Session clientSession = nullptr;

  uint32_t frame_idx = 0u;
};
}  // namespace holoscan::ops
