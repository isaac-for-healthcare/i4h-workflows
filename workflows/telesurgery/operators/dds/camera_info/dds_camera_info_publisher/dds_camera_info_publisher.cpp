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

namespace holoscan::ops
{

  void DDSCameraInfoPublisherOp::setup(OperatorSpec &spec)
  {
    DDSOperatorBase::setup(spec);

    spec.input<CameraInfo>("input");

    spec.param(writer_qos_, "writer_qos", "Writer QoS", "Data Writer QoS Profile", std::string());
    spec.param(topic_, "topic", "Topic", "Topic name", std::string("topic_wrist_camera_data_rgb"));
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
    auto time_now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
    auto input = op_input.receive<CameraInfo>("input");
    if (!input)
    {
      HOLOSCAN_LOG_WARN("No input received");
      return;
    }

    auto camera_info = input.value();
    camera_info.video_publisher_timestamp(time_now);
    writer_.write(camera_info);
    total_frames_published_++;

    // print stats every 3 seconds
    if (std::chrono::steady_clock::now() - last_stats_time_ > std::chrono::seconds(3))
    {
      HOLOSCAN_LOG_INFO("Published {} frames", total_frames_published_);
      last_stats_time_ = std::chrono::steady_clock::now();
    }
  }

} // namespace holoscan::ops
