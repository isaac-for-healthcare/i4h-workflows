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

  struct LatencyStats
  {
    double min = std::numeric_limits<double>::max();
    double max = 0.0;
    double sum = 0.0;
    int count = 0;

    void update(double value)
    {
      min = std::min(min, value);
      max = std::max(max, value);
      sum += value;
      count++;
    }

    double average() const { return count > 0 ? sum / count : 0.0; }

    void reset()
    {
      min = std::numeric_limits<double>::max();
      max = 0.0;
      sum = 0.0;
      count = 0;
    }
  };

  void DDSCameraInfoSubscriberOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    // Change output type to Tensor
    spec.output<nvidia::gxf::Entity>("video");
    spec.output<nvidia::gxf::Entity>("overlay");
    spec.output<std::vector<ops::HolovizOp::InputSpec>>("overlay_specs");

    spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
    spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
    spec.param(topic_, "topic", "Topic", "Topic name", std::string("camera_info"));
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
    dds::sub::LoanedSamples<CameraInfo> frames = reader_.take();
    auto received_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();

    // Always take the very last valid frame based on the frame_num
    CameraInfo last_valid_frame_data;
    bool has_last_valid_frame = false;

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

        if (!has_last_valid_frame || frame.data().frame_num() > last_valid_frame_data.frame_num())
        {
          last_valid_frame_data = frame.data();
          has_last_valid_frame = true;
        }
      }
    }

    if (has_last_valid_frame)
    {
      // Pass allocator handle and op_output to emit_frame
      auto emit_timestamp = emit_frame(last_valid_frame_data,
                                       op_output,
                                       context);

      if (emit_timestamp > 0)
      { // Only record stats if frame was successfully emitted
        record_stats(last_valid_frame_data, received_time_ns, emit_timestamp);
      }
    }

    print_stats();
  }

  uint64_t DDSCameraInfoSubscriberOp::emit_frame(
      const CameraInfo &frame,
      OutputContext &op_output,
      ExecutionContext &context)
  {
    if (frame.frame_num() > 0)
    {
      auto allocator =
          nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

      auto output = nvidia::gxf::Entity::New(context.context());
      if (!output)
      {
        throw std::runtime_error("Failed to allocate message for output");
      }

      auto overlay_entity = gxf::Entity::New(&context);
      if (!overlay_entity)
      {
        throw std::runtime_error("Failed to allocate overlay entity");
      }

      auto overlay_specs = std::vector<HolovizOp::InputSpec>();

      auto shape = nvidia::gxf::Shape{
          static_cast<int>(frame.height()), static_cast<int>(frame.width()), frame.channels()};

      auto tensor = output.value().add<nvidia::gxf::Tensor>("");
      if (!tensor)
      {
        throw std::runtime_error("Failed to allocate tensor");
      }

      // Allocate memory and reshape the tensor
      tensor.value()->reshape<uint8_t>(
          shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

      // Copy the data instead of wrapping it
      auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
      auto bytes_per_element = nvidia::gxf::PrimitiveTypeSize(type);
      size_t data_size = frame.width() * frame.height() * frame.channels() * bytes_per_element;

      // Copy data from frame to tensor
      CUDA_TRY(cudaMemcpy(tensor.value()->pointer(),
                          frame.data().data(),
                          data_size,
                          cudaMemcpyHostToDevice));

      // generate overlay specs
      std::stringstream ss;
      ss.clear();
      for (auto index = 0; index < frame.joint_names().size(); index++)
      {
        ss << frame.joint_names()[index] << ": " << frame.joint_positions()[index]
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
      add_data<1, 2>(overlay_entity, "joint_positions", {{{0.01f, 0.06f}}}, context);

      auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
      auto result = gxf::Entity(std::move(output.value()));
      op_output.emit(result, "video");
      op_output.emit(overlay_entity, "overlay");
      op_output.emit(overlay_specs, "overlay_specs");

      return timestamp;
    }
    else
    {
      HOLOSCAN_LOG_WARN("Received CameraInfo with frame_num 0, skipping emit.");
    }
    return 0;
  }

  void DDSCameraInfoSubscriberOp::record_stats(const CameraInfo &frame, uint64_t received_time_ns, uint64_t emit_timestamp)
  {
    uint64_t message_id = frame.message_id();

    // Calculate latencies - only for messages with non-zero message_id and valid timestamps
    if (message_id != 0)
    {
      // 1. Capture to HID publish latency (ms)
      double hid_capture_to_hid_publish_ms =
          (frame.hid_publish_timestamp() - frame.hid_capture_timestamp()) /
          1000000.0;

      // 2. HID publish to receive latency (ms)
      double hid_publish_to_hid_receive_ms = // Renamed variable for clarity
          (frame.hid_receive_timestamp() - frame.hid_publish_timestamp()) /
          1000000.0;

      // 3. HID receive to HID to Sim latency (ms)
      double hid_receive_to_hid_to_sim_ms = // Renamed variable for clarity
          (frame.hid_to_sim_timestamp() - frame.hid_receive_timestamp()) /
          1000000.0;

      // 4. HID to Sim to HID Process latency (ms)
      double hid_to_sim_to_hid_process_ms =
          (frame.hid_process_timestamp() - frame.hid_to_sim_timestamp()) /
          1000000.0;

      // 5. HID Process to Video Acquisition latency (ms)
      double hid_process_to_video_acquisition_ms =
          (frame.video_acquisition_timestamp() - frame.hid_process_timestamp()) /
          1000000.0;

      // 6. Video Acquisition to Video Data Bridge Enter latency (ms)
      double video_acquisition_to_video_data_bridge_enter_ms =
          (frame.video_data_bridge_enter_timestamp() - frame.video_acquisition_timestamp()) /
          1000000.0;

      // 7. Video Data Bridge Enter to Video Data Bridge Emit latency (ms)
      double video_data_bridge_enter_to_video_data_bridge_emit_ms =
          (frame.video_data_bridge_emit_timestamp() - frame.video_data_bridge_enter_timestamp()) /
          1000000.0;

      // 8. Video Data Bridge Emit to Video Publish latency (ms)
      double video_data_bridge_emit_to_video_publish_ms =
          (frame.video_publisher_timestamp() - frame.video_data_bridge_emit_timestamp()) / // Corrected typo
          1000000.0;

      // 9. Video Publish to Subscriber Receive latency (ms)
      double video_publish_to_subscriber_receive_ms =
          (received_time_ns - frame.video_publisher_timestamp()) /
          1000000.0;

      // 10. Subscriber Receive to Subscriber Emit latency (ms)
      double subscriber_receive_to_subscriber_emit_ms =
          (emit_timestamp - received_time_ns) /
          1000000.0;

      // 11. Network latency (ms) - Sum of publish-to-receive latencies
      double network_latency_ms = hid_publish_to_hid_receive_ms + video_publish_to_subscriber_receive_ms;

      // 12. End-to-end latency (ms)
      double end_to_end_latency_ms = (emit_timestamp - frame.hid_capture_timestamp()) / 1000000.0; // Convert ns to ms

      // Update latency statistics
      hid_capture_to_hid_publish_stats_.update(hid_capture_to_hid_publish_ms);
      hid_publish_to_hid_receive_stats_.update(hid_publish_to_hid_receive_ms); // Corrected typo
      hid_receive_to_hid_to_sim_stats_.update(hid_receive_to_hid_to_sim_ms);   // Corrected typo
      hid_to_sim_to_hid_process_stats_.update(hid_to_sim_to_hid_process_ms);
      hid_process_to_video_acquisition_stats_.update(hid_process_to_video_acquisition_ms);
      video_acquisition_to_video_data_bridge_enter_stats_.update(video_acquisition_to_video_data_bridge_enter_ms);
      video_data_bridge_enter_to_video_data_bridge_emit_stats_.update(video_data_bridge_enter_to_video_data_bridge_emit_ms);
      video_data_bridge_emit_to_video_publish_stats_.update(video_data_bridge_emit_to_video_publish_ms);
      video_publish_to_subscriber_receive_stats_.update(video_publish_to_subscriber_receive_ms);
      subscriber_receive_to_subscriber_emit_stats_.update(subscriber_receive_to_subscriber_emit_ms);
      network_latency_stats_.update(network_latency_ms);
      end_to_end_latency_stats_.update(end_to_end_latency_ms);
    }
  }

  void DDSCameraInfoSubscriberOp::print_stats()
  {

    // Check if it's time to print stats
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stats_time_)
            .count();

    if (elapsed >= stats_interval_ms_ && hid_capture_to_hid_publish_stats_.count > 0)
    {
      HOLOSCAN_LOG_INFO("=== CameraInfo Message Statistics ===");
      HOLOSCAN_LOG_INFO("Total frames received: {}", total_frames_received_);
      HOLOSCAN_LOG_INFO("Valid frames (rate): {} ({:.2f}%)", valid_frames_received_, (static_cast<double>(valid_frames_received_) / (total_frames_received_ + loss_frame_count_)) * 100.0);
      HOLOSCAN_LOG_INFO("Invalid frames (rate): {} ({:.2f}%)", invalid_frames_received_, (static_cast<double>(invalid_frames_received_) / (total_frames_received_ + loss_frame_count_)) * 100.0);
      double frame_loss_rate = (total_frames_received_ > 0) ? (static_cast<double>(loss_frame_count_) / (total_frames_received_ + loss_frame_count_)) * 100.0 : 0.0;
      HOLOSCAN_LOG_INFO("Lost frames (rate): {} ({:.2f}%)", loss_frame_count_, frame_loss_rate);
      HOLOSCAN_LOG_INFO("Total messages received: {}", total_messages_received_);
      double message_loss_rate = (total_messages_received_ > 0) ? (static_cast<double>(loss_message_count_) / (total_messages_received_ + loss_message_count_)) * 100.0 : 0.0; // Base message rate on stats count
      HOLOSCAN_LOG_INFO("Lost messages (rate): {} ({:.2f}%)", loss_message_count_, message_loss_rate);
      HOLOSCAN_LOG_INFO("++=== Latency Statistics (ms) ===++");

      HOLOSCAN_LOG_INFO("avg ({}): {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.unit_str(),
                        hid_capture_to_hid_publish_stats_.average_auto(),
                        hid_publish_to_hid_receive_stats_.average_auto(),
                        hid_receive_to_hid_to_sim_stats_.average_auto(),
                        hid_to_sim_to_hid_process_stats_.average_auto(),
                        hid_process_to_video_acquisition_stats_.average_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.average_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.average_auto(),
                        video_data_bridge_emit_to_video_publish_stats_.average_auto(),
                        video_publish_to_subscriber_receive_stats_.average_auto(),
                        subscriber_receive_to_subscriber_emit_stats_.average_auto(),
                        network_latency_stats_.average_auto(),
                        end_to_end_latency_stats_.average_auto());
      HOLOSCAN_LOG_INFO("min ({}): {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.unit_str(),
                        hid_capture_to_hid_publish_stats_.min_auto(),
                        hid_publish_to_hid_receive_stats_.min_auto(),
                        hid_receive_to_hid_to_sim_stats_.min_auto(),
                        hid_to_sim_to_hid_process_stats_.min_auto(),
                        hid_process_to_video_acquisition_stats_.min_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.min_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.min_auto(),
                        video_data_bridge_emit_to_video_publish_stats_.min_auto(),
                        video_publish_to_subscriber_receive_stats_.min_auto(),
                        subscriber_receive_to_subscriber_emit_stats_.min_auto(),
                        network_latency_stats_.min_auto(),
                        end_to_end_latency_stats_.min_auto());
      HOLOSCAN_LOG_INFO("max ({}): {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.unit_str(),
                        hid_capture_to_hid_publish_stats_.max_auto(),
                        hid_publish_to_hid_receive_stats_.max_auto(),
                        hid_receive_to_hid_to_sim_stats_.max_auto(),
                        hid_to_sim_to_hid_process_stats_.max_auto(),
                        hid_process_to_video_acquisition_stats_.max_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.max_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.max_auto(),
                        video_data_bridge_emit_to_video_publish_stats_.max_auto(),
                        video_publish_to_subscriber_receive_stats_.max_auto(),
                        subscriber_receive_to_subscriber_emit_stats_.max_auto(),
                        network_latency_stats_.max_auto(),
                        end_to_end_latency_stats_.max_auto());

      last_stats_time_ = current_time;

      // Reset stats after printing
      hid_capture_to_hid_publish_stats_.reset();
      hid_publish_to_hid_receive_stats_.reset();
      hid_receive_to_hid_to_sim_stats_.reset();
      hid_to_sim_to_hid_process_stats_.reset();
      hid_process_to_video_acquisition_stats_.reset();
      video_acquisition_to_video_data_bridge_enter_stats_.reset();
      video_data_bridge_enter_to_video_data_bridge_emit_stats_.reset();
      video_data_bridge_emit_to_video_publish_stats_.reset();
      video_publish_to_subscriber_receive_stats_.reset();
      subscriber_receive_to_subscriber_emit_stats_.reset();
      network_latency_stats_.reset();
      end_to_end_latency_stats_.reset();
    }
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
