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
#include "dds/topic/find.hpp"
#include "gxf/multimedia/video.hpp"
#include "holoscan/core/gxf/entity.hpp"

#define CUDA_TRY(stmt)                                                                       \
  {                                                                                          \
    cudaError_t cuda_status = stmt;                                                          \
    if (cudaSuccess != cuda_status) {                                                        \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                              \
                         __LINE__,                                                           \
                         __FILE__,                                                           \
                         cudaGetErrorString(cuda_status),                                    \
                         int(cuda_status));                                                  \
      throw std::runtime_error("CUDA runtime call failed");                                  \
    }                                                                                        \
  }

namespace holoscan::ops {

struct LatencyStats {
  double min = std::numeric_limits<double>::max();
  double max = 0.0;
  double sum = 0.0;
  int count = 0;

  void update(double value) {
    min = std::min(min, value);
    max = std::max(max, value);
    sum += value;
    count++;
  }

  double average() const { return count > 0 ? sum / count : 0.0; }

  void reset() {
    min = std::numeric_limits<double>::max();
    max = 0.0;
    sum = 0.0;
    count = 0;
  }
};

void DDSCameraInfoSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  // Change output type to Tensor
  spec.output<nvidia::gxf::Entity>("video");
  spec.output<nvidia::gxf::Entity>("overlay");
  spec.output<std::vector<ops::HolovizOp::InputSpec>>("overlay_specs");

  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
  spec.param(topic_, "topic", "Topic", "Topic name", std::string("topic_wrist_camera_data_rgb"));
}

void DDSCameraInfoSubscriberOp::initialize() {
  register_converter<std::vector<ops::HolovizOp::InputSpec>>();
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the CameraInfo topic
  auto topic = dds::topic::find<dds::topic::Topic<CameraInfo>>(participant_, topic_.get());
  if (topic == dds::core::null) {
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

void DDSCameraInfoSubscriberOp::compute(InputContext& op_input, OutputContext& op_output,
                                        ExecutionContext& context) {
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) { throw std::runtime_error("Failed to allocate message for output"); }

  auto overlay_entity = gxf::Entity::New(&context);
  if (!overlay_entity) { throw std::runtime_error("Failed to allocate overlay entity"); }
  auto overlay_specs = std::vector<HolovizOp::InputSpec>();

  bool output_written = false;
  dds::sub::LoanedSamples<CameraInfo> frames = reader_.take();

  // Update total message count
  total_camera_info_messages_received_ += frames.length();

  // record time between here and first line of loop
  for (const auto& frame : frames) {
    if (frame.info().valid()) {
      // Get current time for latency calculation
      auto current_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::high_resolution_clock::now().time_since_epoch())
                                 .count();
      // Track message ID
      uint64_t message_id = frame.data().message_id();
      message_ids_received_.insert(message_id);

      if (frame.data().frame_num() != expected_frame_id_) {
        HOLOSCAN_LOG_WARN("Frame ID out of order: {} (expected {})",
                          frame.data().frame_num(),
                          expected_frame_id_);
        loss_frame_count_++;
      }

      expected_frame_id_ = frame.data().frame_num() + 1;

      if (message_id != 0) {
        if (last_message_id_ == 0) { HOLOSCAN_LOG_INFO("First message ID: {}", message_id); }
        if (message_id != last_message_id_ + 1) {
          HOLOSCAN_LOG_WARN(
              "Message ID out of order: {} (expected {})", message_id, last_message_id_ + 1);
          loss_message_count_++;
        }
        last_message_id_ = message_id;
      }

      // Calculate latencies - only for messages with non-zero message_id
      if (message_id != 0 && frame.data().hid_capture_timestamp() > 0 &&
          frame.data().hid_publish_timestamp() > 0 && frame.data().hid_receive_timestamp() > 0 &&
          frame.data().camera_update_time() > 0 && frame.data().camera_publish_timestamp() > 0) {
        // 1. Capture to HID publish latency (ms)
        double capture_to_publish_ms =
            (frame.data().hid_publish_timestamp() - frame.data().hid_capture_timestamp()) /
            1000000.0;

        // 2. HID publish to receive latency (ms)
        double publish_to_receive_ms =
            (frame.data().hid_receive_timestamp() - frame.data().hid_publish_timestamp()) /
            1000000.0;

        // 3. Receive to Sim Process Start latency (ms)
        double receive_to_camera_update_ms =
            (frame.data().camera_update_time() - frame.data().hid_receive_timestamp()) /
            1000000.0;

        // 4. Sim Process Start to Camera Publish latency (ms)
        double camera_update_to_publish_ms =
            (frame.data().camera_publish_timestamp() - frame.data().camera_update_time()) /
            1000000.0;
        // --- End Refined Breakdown ---

        // 5. Camera publish to compute latency (ms)
        double camera_publish_to_compute_ms =
            (current_time_ns - frame.data().camera_publish_timestamp()) / 1000000.0;

        // 6. Combined App Processing (1, 3)
        double combined_app_processing_ms = capture_to_publish_ms + receive_to_camera_update_ms;

        // 7. Total end-to-end latency (ms)
        double capture_to_compute_ms =
            (current_time_ns - frame.data().hid_capture_timestamp()) / 1000000.0;

        // 8. Network latency (ms)
        double network_latency_ms = publish_to_receive_ms + camera_publish_to_compute_ms;

        // Update latency statistics
        capture_to_publish_stats_.update(capture_to_publish_ms);
        publish_to_receive_stats_.update(publish_to_receive_ms);
        receive_to_camera_update_stats_.update(receive_to_camera_update_ms);
        camera_update_to_publish_stats_.update(camera_update_to_publish_ms);
        camera_publish_to_compute_stats_.update(camera_publish_to_compute_ms);
        in_app_processing_latency_stats_.update(combined_app_processing_ms);
        end_to_end_latency_stats_.update(capture_to_compute_ms);
        network_latency_stats_.update(network_latency_ms);
      }

      auto shape = nvidia::gxf::Shape{
          static_cast<int>(frame.data().height()), static_cast<int>(frame.data().width()), 3};

      auto tensor = output.value().add<nvidia::gxf::Tensor>("");
      if (!tensor) { throw std::runtime_error("Failed to allocate tensor"); }

      // Allocate memory and reshape the tensor
      tensor.value()->reshape<uint8_t>(
          shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

      // Copy the data instead of wrapping it
      auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
      auto bytes_per_element = nvidia::gxf::PrimitiveTypeSize(type);
      size_t data_size = frame.data().width() * frame.data().height() * 3 * bytes_per_element;

      // Copy data from frame to tensor
      CUDA_TRY(cudaMemcpy(tensor.value()->pointer(),
                          frame.data().data().data(),
                          data_size,
                          cudaMemcpyHostToDevice));

      // generate overlay specs
      std::stringstream ss;
      ss << "Robot Index: " << frame.data().robot_index();
      HolovizOp::InputSpec robot_index_spec;
      robot_index_spec.tensor_name_ = "robot_index";
      robot_index_spec.type_ = HolovizOp::InputType::TEXT;
      robot_index_spec.color_ = {0.0f, 1.0f, 0.0f};
      robot_index_spec.text_.clear();
      robot_index_spec.text_.push_back(ss.str());
      robot_index_spec.priority_ = 1;
      overlay_specs.push_back(robot_index_spec);
      add_data<1, 2>(overlay_entity, "robot_index", {{{0.01f, 0.01f}}}, context);

      ss.str("");
      ss.clear();
      for (auto index = 0; index < frame.data().joint_names().size(); index++) {
        ss << frame.data().joint_names()[index] << ": " << frame.data().joint_positions()[index]
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

      output_written = true;
    } else {
      HOLOSCAN_LOG_INFO("Invalid CameraInfo");
    }
  }

  // Check if it's time to print stats
  auto current_time = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stats_time_)
          .count();

  if (elapsed >= stats_interval_ms_) {
    HOLOSCAN_LOG_INFO("=== CameraInfo Message Statistics ===");
    HOLOSCAN_LOG_INFO("Total CameraInfo messages received: {}",
                      total_camera_info_messages_received_);
    HOLOSCAN_LOG_INFO("Unique message IDs received: {}", message_ids_received_.size());

    // Print latency statistics
    if (end_to_end_latency_stats_.count > 0) {
      HOLOSCAN_LOG_INFO("====== Latency Statistics (ms) ======");

      // Print latency stats in the following format:
      // avg: capture_to_publish_stats, 	publish_to_receive_stats,
      // receive_to_camera_update_stats, camera_update_to_publish_stats,
      // camera_publish_to_compute_stats, in_app_processing_latency_stats, network_latency_stats,
      // end_to_end_latency_stats min: capture_to_publish_stats, 	publish_to_receive_stats,
      // receive_to_camera_update_stats, camera_update_to_publish_stats,
      // camera_publish_to_compute_stats, in_app_processing_latency_stats, network_latency_stats,
      // end_to_end_latency_stats max: capture_to_publish_stats, 	publish_to_receive_stats,
      // receive_to_camera_update_stats, camera_update_to_publish_stats,
      // camera_publish_to_compute_stats, in_app_processing_latency_stats, network_latency_stats,
      // end_to_end_latency_stats count: capture_to_publish_stats, 	publish_to_receive_stats,
      // receive_to_camera_update_stats, camera_update_to_publish_stats,
      // camera_publish_to_compute_stats, in_app_processing_latency_stats, network_latency_stats,
      // end_to_end_latency_stats

      HOLOSCAN_LOG_INFO("avg: {}, {}, {}, {}, {}, {}, {}, {}",
                        capture_to_publish_stats_.average(),
                        publish_to_receive_stats_.average(),
                        receive_to_camera_update_stats_.average(),
                        camera_update_to_publish_stats_.average(),
                        camera_publish_to_compute_stats_.average(),
                        in_app_processing_latency_stats_.average(),
                        network_latency_stats_.average(),
                        end_to_end_latency_stats_.average());
      HOLOSCAN_LOG_INFO("min: {}, {}, {}, {}, {}, {}, {}, {}",
                        capture_to_publish_stats_.min,
                        publish_to_receive_stats_.min,
                        receive_to_camera_update_stats_.min,
                        camera_update_to_publish_stats_.min,
                        camera_publish_to_compute_stats_.min,
                        in_app_processing_latency_stats_.min,
                        network_latency_stats_.min,
                        end_to_end_latency_stats_.min);
      HOLOSCAN_LOG_INFO("max: {}, {}, {}, {}, {}, {}, {}, {}",
                        capture_to_publish_stats_.max,
                        publish_to_receive_stats_.max,
                        receive_to_camera_update_stats_.max,
                        camera_update_to_publish_stats_.max,
                        camera_publish_to_compute_stats_.max,
                        in_app_processing_latency_stats_.max,
                        network_latency_stats_.max,
                        end_to_end_latency_stats_.max);
      HOLOSCAN_LOG_INFO("count: {}, {}, {}, {}, {}, {}, {}, {}",
                        capture_to_publish_stats_.count,
                        publish_to_receive_stats_.count,
                        receive_to_camera_update_stats_.count,
                        camera_update_to_publish_stats_.count,
                        camera_publish_to_compute_stats_.count,
                        in_app_processing_latency_stats_.count,
                        network_latency_stats_.count,
                        end_to_end_latency_stats_.count);

      HOLOSCAN_LOG_INFO("Message loss count: {}", loss_message_count_);
      HOLOSCAN_LOG_INFO("Frame loss count: {}", loss_frame_count_);

      if (total_camera_info_messages_received_ > 0) {
        HOLOSCAN_LOG_INFO("Message loss rate: {:.2f}%",
                          (static_cast<double>(loss_message_count_) /
                           static_cast<double>(total_camera_info_messages_received_)) *
                              100.0);
        HOLOSCAN_LOG_INFO("Frame loss rate: {:.2f}%",
                          (static_cast<double>(loss_frame_count_) /
                           static_cast<double>(total_camera_info_messages_received_)) *
                              100.0);
      }
      HOLOSCAN_LOG_INFO("=====================================");
    }

    last_stats_time_ = current_time;
  }

  if (output_written) {
    // Output the buffer
    auto result = gxf::Entity(std::move(output.value()));
    op_output.emit(result, "video");
    op_output.emit(overlay_entity, "overlay");
    op_output.emit(overlay_specs, "overlay_specs");
  }
}

template <std::size_t N, std::size_t C>
void DDSCameraInfoSubscriberOp::add_data(gxf::Entity& entity, const char* name,
                                         const std::array<std::array<float, C>, N>& data,
                                         ExecutionContext& context) {
  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  // Add a tensor
  auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
  // Reshape the tensor to the size of the data
  tensor->reshape<float>(
      nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
  // Copy the data to the tensor
  std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
}

}  // namespace holoscan::ops
