/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan::ops {

/**
 * @brief Base class for a DDS operator.
 */
class StatsOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StatsOp)

  StatsOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;


 private:
    void record_stats(const std::map<std::string, uint64_t> &stats, const std::map<std::string, uint64_t> &decoder_stats);
    void print_stats(const std::map<std::string, uint64_t> &stats);

     // Latency statistics
    struct LatencyStats
    {
      explicit LatencyStats(const std::string &name) : name(name) {}

      std::string name;
      double min = std::numeric_limits<double>::max();
      double max = 0.0;
      double sum = 0.0;
      int count = 0;

      std::string unit_str() const
      {
        return "ms";
      }

      double sum_auto() const
      {
        // Always convert nanoseconds to milliseconds
        return sum / 1000000.0;
      }

      double min_auto() const
      {
        // Always convert nanoseconds to milliseconds
        return min / 1000000.0;
      }

      double max_auto() const
      {
        // Always convert nanoseconds to milliseconds
        return max / 1000000.0;
      }

      double average_auto() const
      {
        // Always convert nanoseconds to milliseconds
        return average() / 1000000.0;
      }

      void update(double value)
      {
        if (value < 0)
        {
          HOLOSCAN_LOG_ERROR("[{}] Latency value is negative: {}", name, value);
        }

        min = std::min(min, value);
        max = std::max(max, value);
        sum += value;
        count++;
      }

      double average() const
      {
        return count > 0 ? sum / count : 0.0;
      }

      void reset()
      {
        min = std::numeric_limits<double>::max();
        max = 0.0;
        sum = 0.0;
        count = 0;
      }
    };

    std::chrono::time_point<std::chrono::steady_clock> last_stats_time_ = std::chrono::steady_clock::now();
    uint64_t stats_interval_ms_ = 1000; // Print stats every 1 second

    LatencyStats hid_capture_to_hid_publish_stats_ = LatencyStats("hid_capture_to_hid_publish"); // 1
    LatencyStats hid_publish_to_hid_receive_stats_ = LatencyStats("hid_publish_to_hid_receive"); // 2
    LatencyStats hid_receive_to_hid_to_sim_stats_ = LatencyStats("hid_receive_to_hid_to_sim"); // 3
    LatencyStats hid_to_sim_to_hid_process_stats_ = LatencyStats("hid_to_sim_to_hid_process"); // 4
    LatencyStats hid_process_to_video_acquisition_stats_ = LatencyStats("hid_process_to_video_acquisition"); // 5
    LatencyStats video_acquisition_to_video_data_bridge_enter_stats_ = LatencyStats("video_acquisition_to_video_data_bridge_enter"); // 6
    LatencyStats video_data_bridge_enter_to_video_data_bridge_emit_stats_ = LatencyStats("video_data_bridge_enter_to_video_data_bridge_emit"); // 7
    LatencyStats video_data_bridge_emit_to_video_encoder_enter_stats_ = LatencyStats("video_data_bridge_emit_to_video_encoder_enter"); // 8
    LatencyStats video_encoder_enter_to_video_encoder_emit_stats_ = LatencyStats("video_encoder_enter_to_video_encoder_emit"); // 9
    LatencyStats video_encoder_emit_to_video_publisher_enter_stats_ = LatencyStats("video_encoder_emit_to_video_publisher_enter"); // 10
    LatencyStats video_publisher_enter_to_video_publisher_emit_stats_ = LatencyStats("video_publisher_enter_to_video_publisher_emit"); // 11
    LatencyStats video_publisher_emit_to_video_subscriber_enter_stats_ = LatencyStats("video_publisher_emit_to_video_subscriber_enter"); // 12
    LatencyStats video_subscriber_enter_to_video_subscriber_emit_stats_ = LatencyStats("video_subscriber_enter_to_video_subscriber_emit"); // 13
    LatencyStats video_subscriber_emit_to_video_decoder_enter_stats_ = LatencyStats("video_subscriber_emit_to_video_decoder_enter"); // 14
    LatencyStats video_decoder_enter_to_video_decoder_emit_stats_ = LatencyStats("video_decoder_enter_to_video_decoder_emit"); // 15

    LatencyStats jitter_stats_ = LatencyStats("jitter"); // 16
    LatencyStats network_latency_stats_ = LatencyStats("network_latency"); // 17
    LatencyStats end_to_end_latency_stats_ = LatencyStats("end_to_end_latency"); // 18

    LatencyStats frame_size_stats_ = LatencyStats("frame_size"); // 19
    LatencyStats encoded_frame_size_stats_ = LatencyStats("encoded_frame_size"); // 20
    LatencyStats fps_stats_ = LatencyStats("fps"); // 21

};

}  // namespace holoscan::ops
