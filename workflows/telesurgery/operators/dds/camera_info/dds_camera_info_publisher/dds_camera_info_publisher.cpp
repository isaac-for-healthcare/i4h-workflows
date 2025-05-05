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

#include "dds_camera_info_publisher.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "dds/topic/find.hpp"
#include "gxf/multimedia/video.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <inttypes.h>

// If GXF has gxf/std/dlpack_utils.hpp it has DLPack support
#if __has_include("gxf/std/dlpack_utils.hpp")
  #define GXF_HAS_DLPACK_SUPPORT 1
  #include "gxf/std/tensor.hpp"
#else
  #define GXF_HAS_DLPACK_SUPPORT 0
  #include "holoscan/core/gxf/gxf_tensor.hpp"
#endif

namespace holoscan::ops
{

  // Simple 64-bit FNV-1a hash for quick data integrity check.
  static inline uint64_t fnv1a_hash(const uint8_t* data, size_t len) {
    const uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
      hash ^= static_cast<uint64_t>(data[i]);
      hash *= kPrime;
    }
    return hash;
  }

  void DDSCameraInfoPublisherOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    spec.input<gxf::Entity>("image");
    spec.input<CameraInfo>("metadata");

    spec.param(writer_qos_, "writer_qos", "Writer QoS", "Data Writer QoS Profile", std::string());
    spec.param(topic_, "topic", "Topic", "Topic name", std::string("topic_wrist_camera_data_rgb"));
    spec.param(encoded_channels_, "encoded_channels", "Encoded Channels", "Number of channels in the encoded image", 3u);
  }

  void DDSCameraInfoPublisherOp::initialize()
  {
    DDSOperatorBase::initialize();

    // Create the publisher
    dds::pub::Publisher publisher(participant_);

    // Create the VideoFrame topic
    auto topic = dds::topic::find<dds::topic::Topic<CameraInfo>>(participant_, topic_.get());
    if (topic == dds::core::null)
    {
      topic = dds::topic::Topic<CameraInfo>(participant_, topic_.get());
    }

    // Create the writer for the CameraInfo
    writer_ = dds::pub::DataWriter<CameraInfo>(publisher, topic,
                                               qos_provider_.datawriter_qos(writer_qos_.get()));
  }

  void DDSCameraInfoPublisherOp::compute(InputContext &op_input,
                                         OutputContext &op_output,
                                         ExecutionContext &context)
  {
    HOLOSCAN_LOG_INFO("DDSCameraInfoPublisherOp::compute");
    // Get current timestamp
    auto time_now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();

    auto image = op_input.receive<gxf::Entity>("image").value();
    if (!image)
    {
      HOLOSCAN_LOG_WARN("No image received");
      return;
    }
    // Get CUDA stream
    cudaStream_t stream = op_input.receive_cuda_stream("image", true);
    HOLOSCAN_LOG_INFO("{} received {}default CUDA stream from port '{}'",
                      name(),
                      stream == cudaStreamDefault ? "" : "non-",
                      fmt::ptr(stream));
    auto maybe_device = context.device_from_stream(stream);
    if (maybe_device) {
      HOLOSCAN_LOG_INFO("CUDA stream from port '{}' corresponds to device {}",
                        "image",
                        maybe_device.value());
    }

    auto maybe_tensor = image.get<Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error("Tensor not found in message");
    }

#if GXF_HAS_DLPACK_SUPPORT
    // Convert to GXF Tensor to access storage_type(), data(), shape(), etc.
    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>(maybe_tensor->dl_ctx());
#else
    auto gxf_tensor = gxf::GXFTensor::from_tensor(maybe_tensor);
#endif

    // The tensor coming from the encoder contains a 1-D byte array (encoded bit-stream).
    const int32_t num_bytes = static_cast<int32_t>(gxf_tensor->shape().dimension(0));
    HOLOSCAN_LOG_INFO("Input bit-stream size: {} bytes", num_bytes);

    // Copy the data to host memory (DDS owns its own copy).
    std::vector<uint8_t> host_data(num_bytes);

    auto tensor_ptr_expected = gxf_tensor->data<uint8_t>();
    if (!tensor_ptr_expected) {
      throw std::runtime_error("Failed to get tensor data pointer");
    }
    auto tensor_ptr = tensor_ptr_expected.value();

    if (gxf_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice) {
      cudaError_t cuda_result = cudaMemcpy(host_data.data(), tensor_ptr, num_bytes, cudaMemcpyDeviceToHost);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(fmt::format("cudaMemcpy failed: {}", cudaGetErrorString(cuda_result)));
      }
    } else {
      std::memcpy(host_data.data(), tensor_ptr, num_bytes);
    }

    // Compute and log hash of the encoded bit-stream for debugging.
    uint64_t hash = fnv1a_hash(host_data.data(), host_data.size());
    HOLOSCAN_LOG_INFO("Publish bitstream hash (FNV-1a 64-bit): 0x{0:016x}", hash);

    auto camera_info = op_input.receive<CameraInfo>("metadata").value();
    camera_info.data(std::move(host_data));
    camera_info.channels(encoded_channels_.get());
    camera_info.video_publisher_timestamp(time_now);
    writer_.write(camera_info);
    auto time_emitted = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(time_emitted - last_emitted_time_).count();
    last_emitted_time_ = time_emitted;
    total_emitted_times += time_diff;

    HOLOSCAN_LOG_INFO("Published camera info with frame number: {}, time diff: {} ns", camera_info.frame_num(), time_diff);
    total_frames_published_++;

    // Record max, min and average data size in camera_info.data()
    auto size = camera_info.data().size();
    if (size > max_data_size_)
    {
      max_data_size_ = size;
    }
    if (size < min_data_size_)
    {
      min_data_size_ = size;
    }
    total_data_size_ += size;
    num_data_samples_++;

    // print stats every 3 seconds
    if (std::chrono::steady_clock::now() - last_stats_time_ > std::chrono::seconds(3))
    {
      HOLOSCAN_LOG_INFO("Published {} frames", total_frames_published_);
      HOLOSCAN_LOG_INFO("Max data size: {} bytes", max_data_size_);
      HOLOSCAN_LOG_INFO("Min data size: {} bytes", min_data_size_);
      HOLOSCAN_LOG_INFO("Average data size: {} bytes", total_data_size_ / num_data_samples_);
      HOLOSCAN_LOG_INFO("Average emit time between frames: {} ms", total_emitted_times / total_frames_published_ / 1000000);
      HOLOSCAN_LOG_INFO("Average FPS: {}", total_frames_published_ / (total_emitted_times / 1000000000));
      last_stats_time_ = std::chrono::steady_clock::now();
    }
  }
} // namespace holoscan::ops
