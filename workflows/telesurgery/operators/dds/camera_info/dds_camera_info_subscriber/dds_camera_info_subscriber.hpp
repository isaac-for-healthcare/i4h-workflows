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

#ifndef CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP
#define CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP


#include <dds/sub/ddssub.hpp>

#include "dds_operator_base.hpp"
#include "CameraInfo.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <unordered_set>
#include <chrono>

namespace holoscan::ops {

/**
 * @brief Operator class to subscribe to a DDS hid stream.
 */
class DDSCameraInfoSubscriberOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSCameraInfoSubscriberOp, DDSOperatorBase)

  DDSCameraInfoSubscriberOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context);

  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<std::string> reader_qos_;
  Parameter<std::string> topic_;

  dds::sub::DataReader<CameraInfo> reader_ = dds::core::null;
  dds::core::cond::StatusCondition status_condition_ = dds::core::null;
  dds::core::cond::WaitSet waitset_;

  // Message tracking variables
  uint64_t total_camera_info_messages_received_ = 0;
  std::unordered_set<uint64_t> message_ids_received_;
  uint64_t last_message_id_ = 0;
  uint64_t loss_message_count_ = 0;
  uint64_t expected_frame_id_ = 0;
  uint64_t loss_frame_count_ = 0;
  std::chrono::time_point<std::chrono::steady_clock> last_stats_time_ = std::chrono::steady_clock::now();
  uint64_t stats_interval_ms_ = 3000; // Print stats every 3 seconds

  // Latency statistics
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

    double average() const {
      return count > 0 ? sum / count : 0.0;
    }

    void reset() {
      min = std::numeric_limits<double>::max();
      max = 0.0;
      sum = 0.0;
      count = 0;
    }
  };

  LatencyStats capture_to_publish_stats_;
  LatencyStats publish_to_receive_stats_;
  LatencyStats receive_to_camera_update_stats_;
  LatencyStats camera_update_to_publish_stats_;
  LatencyStats camera_publish_to_compute_stats_;
  LatencyStats end_to_end_latency_stats_;
  LatencyStats in_app_processing_latency_stats_;
  LatencyStats network_latency_stats_;
};

}  // namespace holoscan::ops


#endif /* CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP */
