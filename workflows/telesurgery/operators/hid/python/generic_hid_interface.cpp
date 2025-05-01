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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "./generic_hid_interface.hpp"
#include "./generic_hid_interface_pydoc.hpp"
#include "../input_event.hpp"

#include "../../operator_util.hpp"

#include "holoscan/python/core/emitter_receiver_registry.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// Automatically export enum values (see https://github.com/pybind/pybind11/issues/1759)
template <typename E, typename... Extra>
py::enum_<E> export_enum(const py::handle &scope, Extra &&...extra)
{
  py::enum_<E> enum_type(
      scope, magic_enum::enum_type_name<E>().data(), std::forward<Extra>(extra)...);
  for (const auto &[value, name] : magic_enum::enum_entries<E>())
  {
    enum_type.value(name.data(), value);
  }
  return enum_type;
}

namespace holoscan::ops
{
  /* Trampoline class for handling Python kwargs
   *
   * These add a constructor that takes a Fragment for which to initialize the operator.
   * The explicit parameter list and default arguments take care of providing a Pythonic
   * kwarg-based interface with appropriate default values matching the operator's
   * default parameters in the C++ API `setup` method.
   *
   * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
   */

  class PyGenericHIDInterface : public GenericHIDInterface
  {
  public:
    /* Inherit the constructors */
    using GenericHIDInterface::GenericHIDInterface;

    // Define a constructor that fully initializes the object.
    PyGenericHIDInterface(Fragment *fragment, const py::args &args,
                          const HumanInterfaceDevicesConfig &human_interface_devices,
                          const std::string &name = "generic_hid_interface")
        : GenericHIDInterface(ArgList{
              Arg{"name", name},
              Arg{"human_interface_devices", human_interface_devices}})
    {
      add_positional_condition_and_resource_args(this, args);
      name_ = name;
      fragment_ = fragment;
      spec_ = std::make_shared<OperatorSpec>(fragment);
      setup(*spec_.get());
    }
  };

  /* The python module */

  PYBIND11_MODULE(_generic_hid_interface, m)
  {
    m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _generic_hid_interface
        .. autosummary::
           :toctree: _generate
           GenericHIDInterface
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    export_enum<HIDDeviceType>(m, "HIDDeviceType");

    py::class_<HumanInterfaceDevice> human_interface_device(
        m, "HumanInterfaceDevice");
    human_interface_device.def(py::init<>())
        .def_readwrite("name", &HumanInterfaceDevice::name)
        .def_readwrite("path", &HumanInterfaceDevice::path)
        .def_readwrite("type", &HumanInterfaceDevice::type);

    py::class_<HumanInterfaceDevicesConfig> human_interface_devices_config(
        m, "HumanInterfaceDevicesConfig");
    human_interface_devices_config.def(py::init<>())
        .def_readwrite("devices", &HumanInterfaceDevicesConfig::devices);

    py::class_<InputEvent> input_event(
        m, "InputEvent");
    input_event.def(py::init<>())
        .def_readwrite("device_type", &InputEvent::device_type)
        .def_readwrite("device_name", &InputEvent::device_name)
        .def_readwrite("event_type", &InputEvent::event_type)
        .def_readwrite("number", &InputEvent::number)
        .def_readwrite("value", &InputEvent::value)
        .def_readwrite("hid_capture_timestamp", &InputEvent::hid_capture_timestamp)
        .def_readwrite("hid_publish_timestamp", &InputEvent::hid_publish_timestamp)
        .def_readwrite("hid_to_sim_timestamp", &InputEvent::hid_to_sim_timestamp)
        .def_readwrite("hid_receive_timestamp", &InputEvent::hid_receive_timestamp)
        .def_readwrite("hid_process_timestamp", &InputEvent::hid_process_timestamp)
        .def_readwrite("message_id", &InputEvent::message_id);

    py::class_<GenericHIDInterface,
               PyGenericHIDInterface,
               Operator,
               std::shared_ptr<GenericHIDInterface>>(
        m, "GenericHIDInterface", doc::GenericHIDInterface::doc_GenericHIDInterface)
        .def(py::init<Fragment *,
                      const py::args &,
                      const HumanInterfaceDevicesConfig &,
                      const std::string &>(),
             "fragment"_a,
             "human_interface_devices"_a = HumanInterfaceDevicesConfig(),
             "name"_a = "generic_hid_interface"s,
             doc::GenericHIDInterface::doc_GenericHIDInterface)
        .def("initialize", &GenericHIDInterface::initialize,
             doc::GenericHIDInterface::doc_initialize)
        .def("setup", &GenericHIDInterface::setup, "spec"_a,
             doc::GenericHIDInterface::doc_setup);

    m.def("register_types", [](EmitterReceiverRegistry &registry)
          {
            registry.add_emitter_receiver<HumanInterfaceDevice>(
                "holoscan::ops::HumanInterfaceDevice"s);
            registry.add_emitter_receiver<HumanInterfaceDevicesConfig>(
                "holoscan::ops::HumanInterfaceDevicesConfig"s);
            registry.add_emitter_receiver<std::vector<HumanInterfaceDevice>>(
                "std::vector<holoscan::ops::HumanInterfaceDevice>"s);

            registry.add_emitter_receiver<std::vector<InputEvent>>(
                "std::vector<InputEvent>"s);
            registry.add_emitter_receiver<HIDDeviceType>("HIDDeviceType"s);
          });
  } // PYBIND11_MODULE NOLINT
} // namespace holoscan::ops
