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
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <generic_hid_interface.hpp>
#include <dds_hid_publisher.hpp>
#include <dds_camera_info_subscriber.hpp>

#include <getopt.h>

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
    add_operator(hid_interface);

    auto hid_publisher = make_operator<ops::DDSHIDPublisherOp>(
        "hid_publisher",
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = std::string("60hz")),
        from_config("hid_publisher"));

    add_flow(hid_interface, hid_publisher, {{"output", "input"}});

    auto camera_subscriber = make_operator<ops::DDSCameraInfoSubscriberOp>(
        "camera_subscriber",
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = std::string("120hz")),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
        from_config("camera"));

    auto holoviz =
        make_operator<ops::HolovizOp>("holoviz",
                                      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
                                      from_config("holoviz"));

    add_flow(camera_subscriber,
             holoviz,
             {{"video", "receivers"}, {"overlay", "receivers"}, {"overlay_specs", "input_specs"}});
    HOLOSCAN_LOG_INFO("Composed SurgeonApp");
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
  app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
      "scheduler", app->from_config("scheduler")));
  app->run();

  return 0;
}
