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
#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.

#include "holoscan/core/gxf/entity.hpp"

#include "dds/topic/find.hpp"

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


  void DDSCameraInfoPublisherOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    spec.input<nvidia::gxf::Entity>("image");
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
    // Get current timestamp
    auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();

    auto maybe_image = op_input.receive<nvidia::gxf::Entity>("image");
    if (!maybe_image)
    {
      HOLOSCAN_LOG_WARN("No image received");
      return;
    }
    auto& image = static_cast<nvidia::gxf::Entity&>(maybe_image.value());
    auto tensors = image.findAll<nvidia::gxf::Tensor, 4>();
    if (!tensors)
    {
      throw std::runtime_error("Tensor not found");
    }

    auto maybe_tensor = image.get<nvidia::gxf::Tensor>("");
    if (!maybe_tensor)
    {
      throw std::runtime_error("Tensor not found in message");
    }
    auto frame_size = maybe_tensor.value()->size();

    // Copy the data to host memory (DDS owns its own copy).
    std::vector<uint8_t> host_data(frame_size);

    auto tensor_ptr = maybe_tensor.value()->pointer();

    if (maybe_tensor.value()->storage_type() == nvidia::gxf::MemoryStorageType::kDevice)
    {
      cudaError_t cuda_result = cudaMemcpy(host_data.data(), tensor_ptr, frame_size, cudaMemcpyDeviceToHost);
      if (cuda_result != cudaSuccess)
      {
        throw std::runtime_error(fmt::format("cudaMemcpy failed: {}", cudaGetErrorString(cuda_result)));
      }
    }
    else
    {
      std::memcpy(host_data.data(), tensor_ptr, frame_size);
    }

    // Compute and log hash of the encoded bit-stream for debugging.
    auto meta = metadata();

    auto camera_info = op_input.receive<CameraInfo>("metadata").value();
    camera_info.data(std::move(host_data));
    camera_info.channels(encoded_channels_.get());
    camera_info.video_encoder_enter_timestamp(meta->get<int64_t>("video_encoder_enter_timestamp"));
    camera_info.video_encoder_emit_timestamp(meta->get<int64_t>("video_encoder_emit_timestamp"));
    camera_info.video_publisher_enter_timestamp(enter_timestamp);
    camera_info.video_publisher_emit_timestamp(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count());
    camera_info.encoded_frame_size(frame_size);
    writer_.write(camera_info);
    auto time_emitted = std::chrono::system_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(time_emitted - last_emitted_time_).count();
    last_emitted_time_ = time_emitted;
    total_emitted_times += time_diff;

    total_frames_published_++;

    // Record max, min and average data size in camera_info.data()
    if (frame_size > max_data_size_)
    {
      max_data_size_ = frame_size;
    }
    if (frame_size < min_data_size_)
    {
      min_data_size_ = frame_size;
    }
    total_data_size_ += frame_size;
    num_data_samples_++;

    // print stats every 3 seconds
    if (std::chrono::system_clock::now() - last_stats_time_ > std::chrono::seconds(3))
    {
      HOLOSCAN_LOG_INFO("Published {} frames", total_frames_published_);
      HOLOSCAN_LOG_INFO("Max data size: {} bytes", max_data_size_);
      HOLOSCAN_LOG_INFO("Min data size: {} bytes", min_data_size_);
      HOLOSCAN_LOG_INFO("Average data size: {} bytes", total_data_size_ / num_data_samples_);
      HOLOSCAN_LOG_INFO("Average emit time between frames: {} ms", total_emitted_times / total_frames_published_ / 1e6);
      HOLOSCAN_LOG_INFO("Average FPS: {}", total_frames_published_ / (total_emitted_times / 1e9));
      last_stats_time_ = std::chrono::system_clock::now();
    }
  }

} // namespace holoscan::ops
