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

#include "dds_camera_info_subscriber.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "dds/topic/find.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/multimedia/video.hpp"
#include "holoscan/core/gxf/entity.hpp"

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS)                                                             \
    {                                                                                       \
      const char *error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char *error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }
#define CUDA_TRY(stmt)                                                                       \
  {                                                                                          \
    cudaError_t cuda_status = stmt;                                                          \
    if (cudaSuccess != cuda_status)                                                          \
    {                                                                                        \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                              \
                         __LINE__,                                                           \
                         __FILE__,                                                           \
                         cudaGetErrorString(cuda_status),                                    \
                         int(cuda_status));                                                  \
      throw std::runtime_error("CUDA runtime call failed");                                  \
    }                                                                                        \
  }

namespace holoscan::ops
{
  void DDSCameraInfoSubscriberOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    // Change output type to Tensor
    spec.output<nvidia::gxf::Entity>("video");
    spec.output<nvidia::gxf::Entity>("overlay");
    spec.output<std::vector<ops::HolovizOp::InputSpec>>("overlay_specs");
    spec.output<std::map<std::string, uint64_t>>("stats");

    spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
    spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
    spec.param(topic_, "topic", "Topic", "Topic name", std::string("camera_info"));
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
               "Device to use for CUDA operations", 0);
  }

  void DDSCameraInfoSubscriberOp::initialize()
  {
    // Requires Holoscan SDK 3.1.0 or later
    register_converter<std::vector<ops::HolovizOp::InputSpec>>();
    DDSOperatorBase::initialize();

    // Create the subscriber
    dds::sub::Subscriber subscriber(participant_);

    // Create the CameraInfo topic
    auto topic = dds::topic::find<dds::topic::Topic<CameraInfo>>(participant_, topic_.get());
    if (topic == dds::core::null)
    {
      topic = dds::topic::Topic<CameraInfo>(participant_, topic_.get());
    }

    // Create the reader for the CameraInfo
    reader_ = dds::sub::DataReader<CameraInfo>(
        subscriber, topic, qos_provider_.datareader_qos(reader_qos_.get()));

    // Obtain the reader's status condition
    status_condition_ = dds::core::cond::StatusCondition(reader_);

    // Enable the 'data available' status
    status_condition_.enabled_statuses(dds::core::status::StatusMask::data_available());

    // Attach the status condition to the waitset
    waitset_ += status_condition_;

  }

  void DDSCameraInfoSubscriberOp::compute(InputContext &op_input, OutputContext &op_output,
                                          ExecutionContext &context)
  {
    auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();

    dds::sub::LoanedSamples<CameraInfo> frames = reader_.take();

    auto camera_info = find_valid_frame(frames);
    if(camera_info)
    {
      // Pass allocator handle and op_output to emit_frame
      emit_frame(camera_info,op_output, context, enter_timestamp, false);

      last_valid_frame_data_ = camera_info;
    }
    else if (last_valid_frame_data_)
    {
      // prevent Holoviz freezing when no new frame is received
      HOLOSCAN_LOG_WARN("Received {} frames from DDS: emitting cached frame", frames.length());
      // Pass allocator handle and op_output to emit_frame
      emit_frame(last_valid_frame_data_,op_output, context, enter_timestamp, true);
    }
  }

  nvidia::gxf::Entity DDSCameraInfoSubscriberOp::create_video_output(
      const std::shared_ptr<CameraInfo> camera_info,
      ExecutionContext &context)
  {
    auto allocator =
          nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

      auto output = nvidia::gxf::Entity::New(context.context());
      if (!output)
      {
        throw std::runtime_error("Failed to allocate message for output");
      }

      auto maybe_tensor = output.value().add<nvidia::gxf::Tensor>();
      if (!maybe_tensor)
      {
        throw std::runtime_error("Failed to allocate tensor");
      }

      int frame_size = camera_info->data().size();
      auto tensor = maybe_tensor.value();
      tensor->reshape<uint8_t>({frame_size}, nvidia::gxf::MemoryStorageType::kHost, allocator.value());
      std::memcpy(tensor->pointer(), camera_info->data().data(), frame_size);

      return output.value();
  }

  std::tuple<nvidia::gxf::Entity, std::vector<HolovizOp::InputSpec>> DDSCameraInfoSubscriberOp::create_overlay_output(
      const std::shared_ptr<CameraInfo> camera_info,
      ExecutionContext &context,
      bool cached)
  {
      // generate overlay specs

      auto overlay_entity = gxf::Entity::New(&context);
      if (!overlay_entity)
      {
        throw std::runtime_error("Failed to allocate overlay entity");
      }

      auto overlay_specs = std::vector<HolovizOp::InputSpec>();

      std::stringstream ss;
      ss.clear();
      for (auto index = 0; index < camera_info->joint_names().size(); index++)
      {
        ss << camera_info->joint_names()[index] << ": " << camera_info->joint_positions()[index]
           << "\n";
      }
      HolovizOp::InputSpec joint_positions_spec;
      joint_positions_spec.tensor_name_ = "joint_positions";
      joint_positions_spec.type_ = HolovizOp::InputType::TEXT;
      joint_positions_spec.color_ = {1.0f, 0.0f, 0.0f};
      joint_positions_spec.text_.clear();
      joint_positions_spec.text_.push_back(ss.str());
      joint_positions_spec.priority_ = 1;
      overlay_specs.push_back(joint_positions_spec);
      add_data<1, 2>(overlay_entity, "joint_positions", {{{0.01f, 0.01f}}}, context);

      if(cached)
      {
        HolovizOp::InputSpec cached_frame_spec;
        cached_frame_spec.tensor_name_ = "cached_frame";
        cached_frame_spec.type_ = HolovizOp::InputType::TEXT;
        cached_frame_spec.color_ = {1.0f, 0.0f, 0.0f};
        cached_frame_spec.text_.clear();
        cached_frame_spec.text_.push_back("CACHED");
        cached_frame_spec.priority_ = 2;
        overlay_specs.push_back(cached_frame_spec);
        add_data<1, 2>(overlay_entity, "cached_frame", {{{0.90f, 0.95f}}}, context);

        HolovizOp::InputSpec cached_frame_box_spec;
        cached_frame_box_spec.tensor_name_ = "cached_frame_box";
        cached_frame_box_spec.type_ = HolovizOp::InputType::RECTANGLES;
        cached_frame_box_spec.line_width_ = 20.F;
        cached_frame_box_spec.color_ =  {1.0f, 0.0f, 0.0f};
        cached_frame_box_spec.priority_ = 3;
        overlay_specs.push_back(cached_frame_box_spec);
        add_data<2, 2>(overlay_entity,
                   "cached_frame_box",
                   {{{0.0F, 0.0F},
                     {1.0F, 1.0F}}},
                   context);
      }

      return { overlay_entity, overlay_specs };
  }

  std::map<std::string, uint64_t> DDSCameraInfoSubscriberOp::create_stats_output(
      const std::shared_ptr<CameraInfo> camera_info,
      uint64_t enter_timestamp,
      bool cached)
  {
    return {
      {"hid_capture_timestamp", camera_info->hid_capture_timestamp()},
      {"hid_publish_timestamp", camera_info->hid_publish_timestamp()},
      {"hid_receive_timestamp", camera_info->hid_receive_timestamp()},
      {"hid_to_sim_timestamp", camera_info->hid_to_sim_timestamp()},
      {"hid_process_timestamp", camera_info->hid_process_timestamp()},
      {"video_acquisition_timestamp", camera_info->video_acquisition_timestamp()},
      {"video_data_bridge_enter_timestamp", camera_info->video_data_bridge_enter_timestamp()},
      {"video_data_bridge_emit_timestamp", camera_info->video_data_bridge_emit_timestamp()},
      {"video_encoder_enter_timestamp", camera_info->video_encoder_enter_timestamp()},
      {"video_encoder_emit_timestamp", camera_info->video_encoder_emit_timestamp()},
      {"video_publisher_enter_timestamp", camera_info->video_publisher_enter_timestamp()},
      {"video_publisher_emit_timestamp", camera_info->video_publisher_emit_timestamp()},
      {"video_subscriber_enter_timestamp", enter_timestamp},
      {"total_frames_received", total_frames_received_},
      {"valid_frames_received", valid_frames_received_},
      {"invalid_frames_received", invalid_frames_received_},
      {"loss_frame_count", loss_frame_count_},
      {"total_messages_received", total_messages_received_},
      {"loss_message_count", loss_message_count_},
      {"skipped_frame_count", skipped_frame_count_},
      {"frame_size", camera_info->frame_size()},
      {"encoded_frame_size", camera_info->encoded_frame_size()},
      {"cached", cached}
     };
  }

  uint64_t DDSCameraInfoSubscriberOp::emit_frame(
      const std::shared_ptr<CameraInfo> camera_info,
      OutputContext &op_output,
      ExecutionContext &context,
      uint64_t enter_timestamp,
      bool cached)
  {
      if(camera_info == nullptr)
      {
        throw std::runtime_error("Received nullptr frame");
      }

      auto video_output = create_video_output(camera_info, context);
      auto [overlay_entity, overlay_specs] = create_overlay_output(camera_info, context, cached);
      auto stats = create_stats_output(camera_info, enter_timestamp, cached);


      auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();

      stats["video_subscriber_emit_timestamp"] = timestamp;

      auto video_entity = gxf::Entity(std::move(video_output));
      op_output.emit(video_entity, "video");
      op_output.emit(overlay_entity, "overlay");
      op_output.emit(overlay_specs, "overlay_specs");
      op_output.emit(stats, "stats");

      return timestamp;
  }

  std::shared_ptr<CameraInfo>  DDSCameraInfoSubscriberOp::find_valid_frame(dds::sub::LoanedSamples<CameraInfo> &frames) {
    std::shared_ptr<CameraInfo> last_valid_frame_data = nullptr;

    for (const auto &frame : frames)
    {
      total_frames_received_++;
      if (!frame.info().valid())
      {
        invalid_frames_received_++;
      }
      else
      {
        valid_frames_received_++;
        uint64_t message_id = frame.data().message_id();
        uint64_t frame_num = frame.data().frame_num();

        if (next_frame_id_ != 0 && frame_num != next_frame_id_)
        {
          loss_frame_count_ += (frame_num > next_frame_id_) ? (frame_num - next_frame_id_) : 1; // Account for jumps or single losses
        }
        next_frame_id_ = frame_num + 1;

        if (message_id != 0)
        {
          total_messages_received_++;
          if (next_message_id_ != 0 && message_id != next_message_id_)
          {
            loss_message_count_ += (message_id > next_message_id_) ? (message_id - next_message_id_) : 1;
          }
          next_message_id_ = message_id + 1;
        }

        if (!last_valid_frame_data || frame.data().frame_num() > last_valid_frame_data->frame_num())
        {
          last_valid_frame_data = std::make_shared<CameraInfo>(frame.data());
        }
        else
        {
          skipped_frame_count_++;
        }
      }
    }
    return last_valid_frame_data;
  }

  template <std::size_t N, std::size_t C>
  void DDSCameraInfoSubscriberOp::add_data(gxf::Entity &entity, const char *name,
                                           const std::array<std::array<float, C>, N> &data,
                                           ExecutionContext &context)
  {
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
    // Add a tensor
    auto tensor = static_cast<nvidia::gxf::Entity &>(entity).add<nvidia::gxf::Tensor>(name).value();
    // Reshape the tensor to the size of the data
    tensor->reshape<float>(
        nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
    // Copy the data to the tensor
    std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
  }


} // namespace holoscan::ops
