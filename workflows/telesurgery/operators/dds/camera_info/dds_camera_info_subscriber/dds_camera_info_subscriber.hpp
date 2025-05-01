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

namespace holoscan::ops
{

  /**
   * @brief Operator class to subscribe to a DDS hid stream.
   */
  class DDSCameraInfoSubscriberOp : public DDSOperatorBase
  {
  public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSCameraInfoSubscriberOp, DDSOperatorBase)

    DDSCameraInfoSubscriberOp() = default;

    void setup(OperatorSpec &spec) override;
    void initialize() override;
    void compute(InputContext &op_input, OutputContext &op_output,
                 ExecutionContext &context) override;

  private:
    template <std::size_t N, std::size_t C>
    void add_data(gxf::Entity &entity, const char *name,
                  const std::array<std::array<float, C>, N> &data, ExecutionContext &context);

    uint64_t emit_frame(const CameraInfo &frame,
                        OutputContext &op_output,
                        ExecutionContext &context);
    void record_stats(const CameraInfo &frame, uint64_t received_time_ns, uint64_t emit_timestamp);
    void print_stats();

    Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    Parameter<std::string> reader_qos_;
    Parameter<std::string> topic_;

    dds::sub::DataReader<CameraInfo> reader_ = dds::core::null;
    dds::core::cond::StatusCondition status_condition_ = dds::core::null;
    dds::core::cond::WaitSet waitset_;

    // Message tracking variables
    uint64_t total_frames_received_ = 0;
    uint64_t invalid_frames_received_ = 0;
    uint64_t valid_frames_received_ = 0;
    uint64_t next_frame_id_ = 0;
    uint64_t loss_frame_count_ = 0;

    uint64_t total_messages_received_ = 0;
    uint64_t next_message_id_ = 0;
    uint64_t loss_message_count_ = 0;

    std::chrono::time_point<std::chrono::steady_clock> last_emit_time = std::chrono::steady_clock::now();
    std::chrono::time_point<std::chrono::steady_clock> last_stats_time_ = std::chrono::steady_clock::now();
    uint64_t stats_interval_ms_ = 3000; // Print stats every 3 seconds

    // Latency statistics
    struct LatencyStats
    {
      explicit LatencyStats(const std::string &name) : name(name) {}
      enum class Unit
      {
        ns,
        us,
        ms,
        s
      };

      std::string name;
      Unit unit = Unit::ns;
      double min = std::numeric_limits<double>::max();
      double max = 0.0;
      double sum = 0.0;
      int count = 0;

      std::string unit_str() const
      {
        switch (unit)
        {
        case Unit::ns:
          return "ns";
        case Unit::us:
          return "us";
        case Unit::ms:
          return "ms";
        case Unit::s:
          return "s";
        default:
          return "unknown";
        }
      }

      double sum_auto() const
      {
        if (unit == Unit::ns)
        {
          return sum;
        }
        else if (unit == Unit::us)
        {
          return sum / 1000.0;
        }
        else if (unit == Unit::ms)
        {
          return sum / 1000000.0;
        }
        else
        {
          return sum / 1000000000.0;
        }
      }

      double min_auto() const
      {
        if (unit == Unit::ns)
        {
          return min;
        }
        else if (unit == Unit::us)
        {
          return min / 1000.0;
        }
        else if (unit == Unit::ms)
        {
          return min / 1000000.0;
        }
        else
        {
          return min / 1000000000.0;
        }
      }

      double max_auto() const
      {
        if (unit == Unit::ns)
        {
          return max;
        }
        else if (unit == Unit::us)
        {
          return max / 1000.0;
        }
        else if (unit == Unit::ms)
        {
          return max / 1000000.0;
        }
        else
        {
          return max / 1000000000.0;
        }
      }

      double average_auto() const
      {
        if (unit == Unit::ns)
        {
          return average();
        }
        else if (unit == Unit::us)
        {
          return average() / 1000.0;
        }
        else if (unit == Unit::ms)
        {
          return average() / 1000000.0;
        }
        else
        {
          return average() / 1000000000.0;
        }
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

        if (value < 1000)
        {
          unit = Unit::ns;
        }
        else if (value < 1000000)
        {
          unit = Unit::us;
        }
        else if (value < 1000000000)
        {
          unit = Unit::ms;
        }
        else
        {
          unit = Unit::s;
        }
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

    LatencyStats hid_capture_to_hid_publish_stats_ = LatencyStats("hid_capture_to_hid_publish");
    LatencyStats hid_publish_to_hid_receive_stats_ = LatencyStats("hid_publish_to_hid_receive");
    LatencyStats hid_receive_to_hid_to_sim_stats_ = LatencyStats("hid_receive_to_hid_to_sim");
    LatencyStats hid_to_sim_to_hid_process_stats_ = LatencyStats("hid_to_sim_to_hid_process");
    LatencyStats hid_process_to_video_acquisition_stats_ = LatencyStats("hid_process_to_video_acquisition");
    LatencyStats video_acquisition_to_video_data_bridge_enter_stats_ = LatencyStats("video_acquisition_to_video_data_bridge_enter");
    LatencyStats video_data_bridge_enter_to_video_data_bridge_emit_stats_ = LatencyStats("video_data_bridge_enter_to_video_data_bridge_emit");
    LatencyStats video_data_bridge_emit_to_video_publish_stats_ = LatencyStats("video_data_bridge_emit_to_video_publish");
    LatencyStats video_publish_to_subscriber_receive_stats_ = LatencyStats("video_publish_to_subscriber_receive");
    LatencyStats subscriber_receive_to_subscriber_emit_stats_ = LatencyStats("subscriber_receive_to_subscriber_emit");
    LatencyStats network_latency_stats_ = LatencyStats("network_latency");
    LatencyStats end_to_end_latency_stats_ = LatencyStats("end_to_end_latency");
  };

} // namespace holoscan::ops

#endif /* CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP */
