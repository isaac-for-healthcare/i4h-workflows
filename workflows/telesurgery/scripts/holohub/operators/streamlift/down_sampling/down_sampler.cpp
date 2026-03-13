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

#include "down_sampler.hpp"
#include "common/cuda_helper.h"
#include "gxf/std/tensor.hpp"

namespace holoscan::ops {

void StreamLiftDownSamplerOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "Device to use for CUDA operations", 0u);
    spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
    spec.param(width_, "width", "Width", "Video Frame Width", 0u);
    spec.param(height_, "height", "Height", "Video Frame Height", 0u);
    spec.param(output_type_, "output_type", "OutputType", "Output type: 'videobuffer' or 'tensor'", std::string("tensor"));
}

void StreamLiftDownSamplerOp::initialize() {
    Operator::initialize();

    // Initialize CUDA
    CudaCheck(cuInit(0));

    // Get the CUDA device
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
    cu_device_ = cu_device;

    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
    CudaCheck(cuCtxPushCurrent(cu_context_));

    width_ = width_? width_ : 1920 * 2; // 3840
    height_ = height_? height_ : 1080 * 2; // 2160
    HOLOSCAN_LOG_INFO("Streamlift Downsampler Operator:: Using {} x {} (w x h)", width_, height_);

    serverConfig.inConfig.inFmt = STREAMLIFT_FORMAT_R8G8B8A8;
    serverConfig.inConfig.preset = STREAMLIFT_PRESET_GAME_STREAM;
    serverConfig.inConfig.inWidth = width_;
    serverConfig.inConfig.inHeight = height_;
    serverConfig.inConfig.outFmt = STREAMLIFT_FORMAT_R8G8B8A8;

    Streamlift_Error error = Streamlift_Server_Init(&serverSession, cu_context_, &serverConfig);
    if (error != STREAMLIFT_SUCCESS) {
        throw std::runtime_error("Failed to initialize Streamlift server due to error code: " + std::to_string(error));
    }

    serverRuntimeConfig.size = sizeof(unsigned int);
    serverRuntimeConfig.data = new unsigned int[1];
    std::memset(serverRuntimeConfig.data, 0, sizeof(unsigned int));

    bufIn.memoryType = STREAMLIFT_CUDAMEMORYTYPE_DEVICE;
    bufIn.pl.rowPitch = serverConfig.inConfig.inWidth * 4;
    bufIn.pl.pMemory = 0;

    bufOut.memoryType = STREAMLIFT_CUDAMEMORYTYPE_DEVICE;
    bufOut.pl.rowPitch = serverConfig.outConfig.outWidth * 4;
    bufOut.pl.pMemory = 0;
    HOLOSCAN_LOG_INFO("Streamlift Downsampler Operator:: Initialized!");
}

void StreamLiftDownSamplerOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) {
    auto start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    // Get input tensor
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input entity");
    }

    // Get the tensor from the input message
    bool isVideoBuffer = false;
    auto tensor = maybe_entity.value().get<Tensor>("");
    auto buffer = static_cast<nvidia::gxf::Entity>(maybe_entity.value()).get<nvidia::gxf::VideoBuffer>();
    if (!tensor) {
        if (!buffer) {
            throw std::runtime_error("Failed to get tensor/video buffer from input message");
        }
        const auto& info = buffer.value()->video_frame_info();
        if (info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
            HOLOSCAN_LOG_ERROR("color format: {} vs (RGBA) {}; width: {}; height: {}", int(info.color_format), int(nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA), info.width, info.height);
            throw std::runtime_error("Invalid video buffer format; Only RGBA is supported");
        }
        isVideoBuffer = true;
    }

    size_t data_size = isVideoBuffer ? buffer.value()->size() : tensor->size();
    int nFrameSize = width_ * height_ * 4;
    if (data_size != nFrameSize) {
        HOLOSCAN_LOG_ERROR("Frame-{}:: Frame size mismatch: {} (current) != {} (expected)", frame_idx, data_size, nFrameSize);
        if (isVideoBuffer) {
            const auto& info = buffer.value()->video_frame_info();
            HOLOSCAN_LOG_ERROR("width: {}; height: {}; channels: {}", info.width, info.height, info.color_planes.size());
        }
        throw std::runtime_error("Frame size mismatch");
    }
    //HOLOSCAN_LOG_INFO("Frame-{}:: Input Tensor size: {}; shape: {}", frame_idx, tensor->size(), tensor->shape());

    auto h2d_start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    bufIn.pl.pMemory = (CUdeviceptr) (isVideoBuffer ? buffer.value()->pointer() : tensor->data());
    auto h2d_end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto h2d_latency_ms = (h2d_end_ts - h2d_start_ts) / 1000000.0;

    auto d2h_start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto out_message = gxf::Entity::New(&context);
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
    if (output_type_.get() == "tensor") {
        auto output_tensor = static_cast<nvidia::gxf::Entity>(out_message).add<nvidia::gxf::Tensor>().value();
        nvidia::gxf::Shape shape = nvidia::gxf::Shape{static_cast<int>(serverConfig.outConfig.outWidth), static_cast<int>(serverConfig.outConfig.outHeight), static_cast<int>(4)};
        output_tensor->reshape<uint8_t>(shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        bufOut.pl.pMemory = (CUdeviceptr) output_tensor->pointer();
    } else {
        nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> output_buffer = static_cast<nvidia::gxf::Entity>(out_message).add<nvidia::gxf::VideoBuffer>().value();
        output_buffer->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(serverConfig.outConfig.outWidth, serverConfig.outConfig.outHeight, nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());
        bufOut.pl.pMemory = (CUdeviceptr) output_buffer->pointer();
    }
    auto d2h_end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto d2h_latency_ms = (d2h_end_ts - d2h_start_ts) / 1000000.0;

    auto s_start_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    ((unsigned int*) serverRuntimeConfig.data)[0] = frame_idx;
    if (Streamlift_Server_ProcessFrame(serverSession, &bufIn, 1, &bufOut, 1, &serverRuntimeConfig, 0) != STREAMLIFT_SUCCESS) {
        throw std::runtime_error("Failed to process frame");
    }
    auto s_end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto s_latency_ms = (s_end_ts - s_start_ts) / 1000000.0;

    auto meta = metadata();
    auto end_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto total_latency_ms = (end_ts - start_ts) / 1000000.0;

    meta->set("streamlift_down_h2d_latency_ms", h2d_latency_ms);
    meta->set("streamlift_down_sampling_latency_ms", s_latency_ms);
    meta->set("streamlift_down_d2h_latency_ms", d2h_latency_ms);
    meta->set("streamlift_down_total_latency_ms", total_latency_ms);

    //HOLOSCAN_LOG_INFO("Frame-{}:: HostToDevice Copy Latency (ms): {}", frame_idx, h2d_latency_ms);
    //HOLOSCAN_LOG_INFO("Frame-{}:: Down Sampling Latency (ms): {}", frame_idx, s_latency_ms);
    //HOLOSCAN_LOG_INFO("Frame-{}:: DeviceToHost Copy Latency (ms): {}", frame_idx, d2h_latency_ms);
    //HOLOSCAN_LOG_INFO("Frame-{}:: Total Latency: {}", frame_idx, total_latency_ms);

    // Transmit the output message
    op_output.emit(out_message, "output");
    frame_idx++;
}

void StreamLiftDownSamplerOp::stop() {
  if (serverSession) {
    Streamlift_Server_DeInit(&serverSession);
    serverSession = nullptr;
  }

  if (cu_context_) {
    CUcontext current_ctx;
    CUresult result = cuCtxGetCurrent(&current_ctx);
    if (result == CUDA_SUCCESS && current_ctx == cu_context_) {
      CudaCheck(cuCtxPopCurrent(nullptr));
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device_));
    cu_context_ = nullptr;
  }
}
}
