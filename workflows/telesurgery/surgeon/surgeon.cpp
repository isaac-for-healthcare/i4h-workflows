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

#include <holoscan/holoscan.hpp>
#include <holoscan/core/resources/gxf/gxf_component_resource.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>

#include <generic_hid_interface.hpp>
#include <dds_hid_publisher.hpp>
#include <dds_camera_info_subscriber.hpp>

#include <getopt.h>

// The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
// of the decoded H264 bit stream.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder context
//   Handle.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderResponseOp, "nvidia::gxf::VideoDecoderResponse")

// The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
// for the H264 bit stream decode.
// Parameters:
// - inbuf_storage_type (uint32_t): Input Buffer storage type, 0:kHost, 1:kDevice.
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder
//   context Handle.
// - codec (uint32_t): Video codec to use, 0:H264, only H264 supported. Default:0.
// - disableDPB (uint32_t): Enable low latency decode, works only for IPPP case.
// - output_format (std::string): VidOutput frame video format, nv12pl and yuv420planar are
//   supported.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderRequestOp, "nvidia::gxf::VideoDecoderRequest")

// The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoDecoderContext, "nvidia::gxf::VideoDecoderContext")

class ForwardOp : public holoscan::Operator
{
public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardOp)

  ForwardOp() = default;

  void setup(holoscan::OperatorSpec &spec) override
  {
    spec.input<std::any>("in");
    spec.output<std::any>("out");
    spec.param(description_, "description", "Description", "Description of the operator");
  }

  void compute(holoscan::InputContext &op_input, holoscan::OutputContext &op_output,
               holoscan::ExecutionContext &context) override
  {
    auto in_message = op_input.receive<std::any>("in");
    if (in_message)
    {
      auto value = in_message.value();
      if (value.type() == typeid(holoscan::gxf::Entity))
      {
        HOLOSCAN_LOG_INFO("{}: Forwarding entity", description_);
        // emit as entity
        auto entity = std::any_cast<holoscan::gxf::Entity>(value);
        op_output.emit(entity, "out");
      }
      else
      {
        HOLOSCAN_LOG_INFO("{}: Forwarding std::any", description_);
        // emit as std::any
        op_output.emit(value, "out");
      }
    }
  }

private:
  holoscan::Parameter<std::string> description_;
};

class SurgeonApp : public holoscan::Application
{
public:
  void compose() override
  {
    HOLOSCAN_LOG_INFO("Composing SurgeonApp");
    using namespace holoscan;

    // Capture HID events and publish them to DDS
    auto hid_interface = make_operator<ops::GenericHIDInterface>(
        "hid_interface",
        from_config("hid"));

    std::shared_ptr<holoscan::Operator> hid_publisher;
    auto hid_protocol = from_config("protocol.hid").as<std::string>();
    if (hid_protocol == "dds")
    {
      hid_publisher = make_operator<ops::DDSHIDPublisherOp>(
          "hid_publisher",
          make_condition<PeriodicCondition>("periodic-condition",
                                            Arg("recess_period") = std::string("60hz")),
          from_config("hid_publisher"));
    }
    else if (hid_protocol == "streamsdk")
    {
      // TODO: Implement StreamSDK HID publisher
      throw std::runtime_error("StreamSDK HID publisher is not implemented");
    }
    else
    {
      throw std::runtime_error("Invalid HID protocol: " + hid_protocol);
    }
    add_flow(hid_interface, hid_publisher, {{"output", "input"}});

    auto video_protocol = from_config("protocol.video").as<std::string>();
    if (video_protocol == "dds")
    {
      uint32_t width = from_config("holoviz.width").as<uint32_t>();
      uint32_t height = from_config("holoviz.height").as<uint32_t>();
      int64_t source_block_size = width * height * 3 * 4;
      int64_t source_num_blocks = 2;

      auto camera_subscriber = make_operator<ops::DDSCameraInfoSubscriberOp>(
          "camera_subscriber",
          make_condition<PeriodicCondition>("periodic-condition",
                                            Arg("recess_period") = std::string("10hz")),
          Arg("allocator") = make_resource<UnboundedAllocator>("allocator"),
          Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
          from_config("camera"));
      auto forward_op_camera = make_operator<ForwardOp>("forward_op_camera", Arg("description") = std::string("Forwarding camera info"));
      auto response_condition =
          make_condition<AsynchronousCondition>("response_condition");
      auto video_decoder_context = make_resource<VideoDecoderContext>(
          "decoder-context", Arg("async_scheduling_term") = response_condition);

      auto request_condition =
          make_condition<AsynchronousCondition>("request_condition");
      auto video_decoder_request = make_operator<VideoDecoderRequestOp>(
          "video_decoder_request",
          from_config("video_decoder_request"),
          Arg("async_scheduling_term") = request_condition,
          Arg("videodecoder_context") = video_decoder_context);

      auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
          "video_decoder_response",
          from_config("video_decoder_response"),
          Arg("pool") =
              make_resource<BlockMemoryPool>(
                  "pool", 1, source_block_size, source_num_blocks),
          Arg("videodecoder_context") = video_decoder_context);
      auto forward_response = make_operator<ForwardOp>("forward_response", Arg("description") = std::string("Forwarding video decoder response"));

      auto decoder_output_format_converter =
          make_operator<ops::FormatConverterOp>("decoder_output_format_converter",
                                                from_config("decoder_output_format_converter"),
                                                Arg("pool") = make_resource<BlockMemoryPool>(
                                                    "pool", 1, source_block_size, source_num_blocks));
      auto forward_formatter = make_operator<ForwardOp>("forward_formatter", Arg("description") = std::string("Forwarding holoviz"));
      auto holoviz =
          make_operator<ops::HolovizOp>("holoviz",
                                        Arg("allocator") = make_resource<BlockMemoryPool>(
                                            "allocator", 1, source_block_size, source_num_blocks),
                                        from_config("holoviz"));

      add_flow(camera_subscriber, video_decoder_request, {{"video", "input_frame"}});
      add_flow(camera_subscriber, holoviz, {{"overlay", "receivers"}, {"overlay_specs", "input_specs"}});
      add_flow(video_decoder_response, decoder_output_format_converter, {{"output_transmitter", "source_video"}});
      add_flow(decoder_output_format_converter, holoviz, {{"tensor", "receivers"}});

      // add_flow(camera_subscriber, forward_op_camera, {{"video", "in"}});
      // add_flow(forward_op_camera, video_decoder_request, {{"out", "input_frame"}});
      // add_flow(camera_subscriber, holoviz, {{"overlay", "receivers"}, {"overlay_specs", "input_specs"}});
      // add_flow(video_decoder_response,
      //          forward_response,
      //          {{"output_transmitter", "in"}});
      // add_flow(forward_response,
      //          decoder_output_format_converter,
      //          {{"out", "source_video"}});
      // add_flow(decoder_output_format_converter, forward_formatter, {{"tensor", "in"}});
      // add_flow(forward_formatter, holoviz, {{"out", "receivers"}});
    }
    else if (video_protocol == "streamsdk")
    {
      // TODO: Implement StreamSDK video subscriber
      throw std::runtime_error("StreamSDK video subscriber is not implemented");
    }
    else
    {
      throw std::runtime_error("Invalid video protocol: " + video_protocol);
    }
  }
};

void usage()
{
  std::cout << "Usage: surgeon [options]" << std::endl
            << std::endl
            << "Options" << std::endl
            << "  -c PATH,  --config=PATH    Path to the config file" << std::endl;
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char **argv, std::string &config_path)
{
  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"config", required_argument, 0, 'c'},
                                  {0, 0, 0, 0}};

  int c;
  while (optind < argc)
  {
    if ((c = getopt_long(argc, argv, "hsrc:", long_options, NULL)) != -1)
    {
      switch (c)
      {
      case 'h':
        usage();
        return false;
      case 'c':
        config_path = optarg;
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unhandled option '{}'", static_cast<char>(c));
      }
    }
  }

  return true;
}

int main(int argc, char **argv)
{
  std::string config_path = "";

  if (!parse_arguments(argc, argv, config_path))
  {
    return 1;
  }

  if (config_path.empty())
  {
    // Get the input data environment variable
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0')
    {
      auto config_file = std::filesystem::canonical(argv[0]).parent_path();
      config_path = config_file / std::filesystem::path("surgeon.yaml");
    }
  }

  if (!std::filesystem::exists(config_path))
  {
    HOLOSCAN_LOG_ERROR("Config file {} does not exist", config_path);
    return -1;
  }

  HOLOSCAN_LOG_INFO("Starting surgeon app with config {}", config_path);
  auto app = holoscan::make_application<SurgeonApp>();
  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "scheduler", app->from_config("scheduler")));
  app->run();

  return 0;
}
