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

#include "stats_operator.hpp"

namespace holoscan::ops {

void StatsOp::setup(OperatorSpec& spec) {
  spec.input<std::map<std::string, uint64_t>>("in");
  spec.input<std::map<std::string, uint64_t>>("decoder_stats");
}

void StatsOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {

    auto maybe_stats = op_input.receive<std::map<std::string, uint64_t>>("in");
    auto maybe_decoder_stats = op_input.receive<std::map<std::string, uint64_t>>("decoder_stats");
    if (maybe_stats && maybe_decoder_stats) {
      std::map<std::string, uint64_t> stats = maybe_stats.value();
      std::map<std::string, uint64_t> decoder_stats = maybe_decoder_stats.value();

      // Not all camera info messages coming back from the Patient side have timestamps
      if(stats.find("hid_capture_timestamp") != stats.end() &&
        stats.at("hid_capture_timestamp") != 0) {
        if(stats.find("cached") != stats.end() && stats.at("cached") == false) {
          record_stats(stats, decoder_stats);
          print_stats(stats);
        }
      }
    }
}

void StatsOp::record_stats(const std::map<std::string, uint64_t> &stats, const std::map<std::string, uint64_t> &decoder_stats)
{
  // 1. Capture to HID publish latency (ns)
  double hid_capture_to_hid_publish_ns =
      (stats.at("hid_publish_timestamp") - stats.at("hid_capture_timestamp"));

  // 2. HID publish to receive latency (ns)
  double hid_publish_to_hid_receive_ns =
      (stats.at("hid_receive_timestamp") - stats.at("hid_publish_timestamp"));

  // 3. HID receive to HID to Sim latency (ns)
  double hid_receive_to_hid_to_sim_ns =
      (stats.at("hid_to_sim_timestamp") - stats.at("hid_receive_timestamp"));

  // 4. HID to Sim to HID Process latency (ns)
  double hid_to_sim_to_hid_process_ns =
      (stats.at("hid_process_timestamp") - stats.at("hid_to_sim_timestamp"));

  // 5. HID Process to Video Acquisition latency (ns)
  double hid_process_to_video_acquisition_ns =
      (stats.at("video_acquisition_timestamp") - stats.at("hid_process_timestamp"));

  // 6. Video Acquisition to Video Data Bridge Enter latency (ns)
  double video_acquisition_to_video_data_bridge_enter_ns =
      (stats.at("video_data_bridge_enter_timestamp") - stats.at("video_acquisition_timestamp"));

  // 7. Video Data Bridge Enter to Video Data Bridge Emit latency (ns)
  double video_data_bridge_enter_to_video_data_bridge_emit_ns =
      (stats.at("video_data_bridge_emit_timestamp") - stats.at("video_data_bridge_enter_timestamp"));

  // 8. Video Data Bridge Emit to Video Encoder Enter latency (ns)
  double video_data_bridge_emit_to_video_encoder_enter_ns =
      (stats.at("video_encoder_enter_timestamp") - stats.at("video_data_bridge_emit_timestamp"));

  // 9. Video Encoder Enter to Video Encoder Emit latency (ns)
  double video_encoder_enter_to_video_encoder_emit_ns =
      (stats.at("video_encoder_emit_timestamp") - stats.at("video_encoder_enter_timestamp"));

  // 10. Video Encoder Emit to Video Publisher Enter latency (ns)
  double video_encoder_emit_to_video_publisher_enter_ns =
      (stats.at("video_publisher_enter_timestamp") - stats.at("video_encoder_emit_timestamp"));

  // 11. Video Publisher Enter to Video Publisher Emit latency (ns)
  double video_publisher_enter_to_video_publisher_emit_ns =
      (stats.at("video_publisher_emit_timestamp") - stats.at("video_publisher_enter_timestamp"));

  // 12. Video Publisher Emit to Video Subscriber Enter latency (ns)
  double video_publisher_emit_to_video_subscriber_enter_ns =
      (stats.at("video_subscriber_enter_timestamp") - stats.at("video_publisher_emit_timestamp"));

  // 13. Video Subscriber Enter to Video Subscriber Emit latency (ns)
  double video_subscriber_enter_to_video_subscriber_emit_ns =
      (stats.at("video_subscriber_emit_timestamp") - stats.at("video_subscriber_enter_timestamp"));

  // 14. Video Subscriber Emit to Video Decoder Enter latency (ns)
  double video_subscriber_emit_to_video_decoder_enter_ns =
      (decoder_stats.at("video_decoder_enter_timestamp") - stats.at("video_subscriber_emit_timestamp"));

  // 15. Video Decoder Enter to Video Decoder Emit latency (ns)
  double video_decoder_enter_to_video_decoder_emit_ns =
      (decoder_stats.at("video_decoder_emit_timestamp") - decoder_stats.at("video_decoder_enter_timestamp"));

  // 16. Jitter latency (ns)
  double jitter_ns = decoder_stats.at("jitter_time");

  // 17. Network latency (ns) - Sum of publish-to-receive latencies
  double network_latency_ns = hid_publish_to_hid_receive_ns + video_publisher_emit_to_video_subscriber_enter_ns;

  // 18. End-to-end latency (ns)
  double end_to_end_latency_ns = decoder_stats.at("video_decoder_emit_timestamp") - stats.at("hid_capture_timestamp");

  // 19. Frame size (ns)
  double frame_size = stats.at("frame_size");

  // 20. Encoded frame size (ns)
  double encoded_frame_size = stats.at("encoded_frame_size");

  // 21. FPS
  double fps = decoder_stats.at("fps");

  // Update latency statistics
  hid_capture_to_hid_publish_stats_.update(hid_capture_to_hid_publish_ns);
  hid_publish_to_hid_receive_stats_.update(hid_publish_to_hid_receive_ns);
  hid_receive_to_hid_to_sim_stats_.update(hid_receive_to_hid_to_sim_ns);
  hid_to_sim_to_hid_process_stats_.update(hid_to_sim_to_hid_process_ns);
  hid_process_to_video_acquisition_stats_.update(hid_process_to_video_acquisition_ns);
  video_acquisition_to_video_data_bridge_enter_stats_.update(video_acquisition_to_video_data_bridge_enter_ns);
  video_data_bridge_enter_to_video_data_bridge_emit_stats_.update(video_data_bridge_enter_to_video_data_bridge_emit_ns);
  video_data_bridge_emit_to_video_encoder_enter_stats_.update(video_data_bridge_emit_to_video_encoder_enter_ns);
  video_encoder_enter_to_video_encoder_emit_stats_.update(video_encoder_enter_to_video_encoder_emit_ns);
  video_encoder_emit_to_video_publisher_enter_stats_.update(video_encoder_emit_to_video_publisher_enter_ns);
  video_publisher_enter_to_video_publisher_emit_stats_.update(video_publisher_enter_to_video_publisher_emit_ns);
  video_publisher_emit_to_video_subscriber_enter_stats_.update(video_publisher_emit_to_video_subscriber_enter_ns);
  video_subscriber_enter_to_video_subscriber_emit_stats_.update(video_subscriber_enter_to_video_subscriber_emit_ns);
  video_subscriber_emit_to_video_decoder_enter_stats_.update(video_subscriber_emit_to_video_decoder_enter_ns);
  video_decoder_enter_to_video_decoder_emit_stats_.update(video_decoder_enter_to_video_decoder_emit_ns);
  jitter_stats_.update(jitter_ns);
  network_latency_stats_.update(network_latency_ns);
  end_to_end_latency_stats_.update(end_to_end_latency_ns);
  frame_size_stats_.update(frame_size);
  encoded_frame_size_stats_.update(encoded_frame_size);
  fps_stats_.update(fps);
}

void StatsOp::print_stats(const std::map<std::string, uint64_t> &stats) {
    // Check if it's time to print stats
    auto current_time = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_stats_time_)
            .count();
    if (elapsed >= stats_interval_ms_ && hid_capture_to_hid_publish_stats_.count > 0)
    {
      uint64_t total_frames_received = stats.at("total_frames_received");
      uint64_t valid_frames_received = stats.at("valid_frames_received");
      uint64_t invalid_frames_received = stats.at("invalid_frames_received");
      uint64_t loss_frame_count = stats.at("loss_frame_count");
      uint64_t skipped_frame_count = stats.at("skipped_frame_count");
      uint64_t total_messages_received = stats.at("total_messages_received");
      uint64_t loss_message_count = stats.at("loss_message_count");
      HOLOSCAN_LOG_INFO("=== Statistics ===");
      HOLOSCAN_LOG_INFO("Total frames received: {}", total_frames_received);
      HOLOSCAN_LOG_INFO("Valid frames (rate): {} ({:.2f}%)", valid_frames_received, (static_cast<double>(valid_frames_received) / (total_frames_received + loss_frame_count)) * 100.0);
      HOLOSCAN_LOG_INFO("Invalid frames (rate): {} ({:.2f}%)", invalid_frames_received, (static_cast<double>(invalid_frames_received) / (total_frames_received + loss_frame_count)) * 100.0);
      double frame_loss_rate = (total_frames_received > 0) ? (static_cast<double>(loss_frame_count) / (total_frames_received + loss_frame_count)) * 100.0 : 0.0;
      HOLOSCAN_LOG_INFO("Lost frames (rate): {} ({:.2f}%)", loss_frame_count, frame_loss_rate);
      HOLOSCAN_LOG_INFO("Skipped frames (rate): {} ({:.2f}%)", skipped_frame_count, (static_cast<double>(skipped_frame_count) / (total_frames_received + loss_frame_count)) * 100.0);
      HOLOSCAN_LOG_INFO("Total messages received: {}", total_messages_received);
      double message_loss_rate = (total_messages_received > 0) ? (static_cast<double>(loss_message_count) / (total_messages_received + loss_message_count)) * 100.0 : 0.0;
      HOLOSCAN_LOG_INFO("Lost messages (rate): {} ({:.2f}%)", loss_message_count, message_loss_rate);
      HOLOSCAN_LOG_INFO("++=== Latency (milliseconds) ===++");

      HOLOSCAN_LOG_INFO("avg: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.average_auto(),
                        hid_publish_to_hid_receive_stats_.average_auto(),
                        hid_receive_to_hid_to_sim_stats_.average_auto(),
                        hid_to_sim_to_hid_process_stats_.average_auto(),
                        hid_process_to_video_acquisition_stats_.average_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.average_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.average_auto(),
                        video_data_bridge_emit_to_video_encoder_enter_stats_.average_auto(),
                        video_encoder_enter_to_video_encoder_emit_stats_.average_auto(),
                        video_encoder_emit_to_video_publisher_enter_stats_.average_auto(),
                        video_publisher_enter_to_video_publisher_emit_stats_.average_auto(),
                        video_publisher_emit_to_video_subscriber_enter_stats_.average_auto(),
                        video_subscriber_enter_to_video_subscriber_emit_stats_.average_auto(),
                        video_subscriber_emit_to_video_decoder_enter_stats_.average_auto(),
                        video_decoder_enter_to_video_decoder_emit_stats_.average_auto(),
                        jitter_stats_.average_auto(),
                        network_latency_stats_.average_auto(),
                        end_to_end_latency_stats_.average_auto(),
                        frame_size_stats_.average(),
                        encoded_frame_size_stats_.average(),
                        fps_stats_.average());
      HOLOSCAN_LOG_INFO("min: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.min_auto(),
                        hid_publish_to_hid_receive_stats_.min_auto(),
                        hid_receive_to_hid_to_sim_stats_.min_auto(),
                        hid_to_sim_to_hid_process_stats_.min_auto(),
                        hid_process_to_video_acquisition_stats_.min_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.min_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.min_auto(),
                        video_data_bridge_emit_to_video_encoder_enter_stats_.min_auto(),
                        video_encoder_enter_to_video_encoder_emit_stats_.min_auto(),
                        video_encoder_emit_to_video_publisher_enter_stats_.min_auto(),
                        video_publisher_enter_to_video_publisher_emit_stats_.min_auto(),
                        video_publisher_emit_to_video_subscriber_enter_stats_.min_auto(),
                        video_subscriber_enter_to_video_subscriber_emit_stats_.min_auto(),
                        video_subscriber_emit_to_video_decoder_enter_stats_.min_auto(),
                        video_decoder_enter_to_video_decoder_emit_stats_.min_auto(),
                        jitter_stats_.min_auto(),
                        network_latency_stats_.min_auto(),
                        end_to_end_latency_stats_.min_auto(),
                        frame_size_stats_.min,
                        encoded_frame_size_stats_.min,
                        fps_stats_.min);
      HOLOSCAN_LOG_INFO("max: {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
                        hid_capture_to_hid_publish_stats_.max_auto(),
                        hid_publish_to_hid_receive_stats_.max_auto(),
                        hid_receive_to_hid_to_sim_stats_.max_auto(),
                        hid_to_sim_to_hid_process_stats_.max_auto(),
                        hid_process_to_video_acquisition_stats_.max_auto(),
                        video_acquisition_to_video_data_bridge_enter_stats_.max_auto(),
                        video_data_bridge_enter_to_video_data_bridge_emit_stats_.max_auto(),
                        video_data_bridge_emit_to_video_encoder_enter_stats_.max_auto(),
                        video_encoder_enter_to_video_encoder_emit_stats_.max_auto(),
                        video_encoder_emit_to_video_publisher_enter_stats_.max_auto(),
                        video_publisher_enter_to_video_publisher_emit_stats_.max_auto(),
                        video_publisher_emit_to_video_subscriber_enter_stats_.max_auto(),
                        video_subscriber_enter_to_video_subscriber_emit_stats_.max_auto(),
                        video_subscriber_emit_to_video_decoder_enter_stats_.max_auto(),
                        video_decoder_enter_to_video_decoder_emit_stats_.max_auto(),
                        jitter_stats_.max_auto(),
                        network_latency_stats_.max_auto(),
                        end_to_end_latency_stats_.max_auto(),
                        frame_size_stats_.max,
                        encoded_frame_size_stats_.max,
                        fps_stats_.max);

        last_stats_time_ = std::chrono::system_clock::now();
    }
}
}  // namespace holoscan::ops
