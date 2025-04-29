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

#include "../dds_hid_subscriber.hpp"
#include "./dds_hid_subscriber_pydoc.hpp"
#include "InputCommand.hpp"

#include "../../../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

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

  class PyDDSHIDSubscriberOp : public DDSHIDSubscriberOp
  {
  public:
    /* Inherit the constructors */
    using DDSHIDSubscriberOp::DDSHIDSubscriberOp;

    // Define a constructor that fully initializes the object.
    PyDDSHIDSubscriberOp(Fragment *fragment, const py::args &args,
                         const std::string &qos_provider = "",
                         const std::string &participant_qos = "",
                         uint32_t domain_id = 0,
                         const std::string &reader_qos = "",
                         const std::vector<std::string> &hid_device_filters = {},
                         const std::string &name = "dds_hid_subscriber")
        : DDSHIDSubscriberOp(ArgList{
              Arg{"qos_provider", qos_provider},
              Arg{"participant_qos", participant_qos},
              Arg{"domain_id", domain_id},
              Arg{"reader_qos", reader_qos},
              Arg{"hid_device_filters", hid_device_filters}})
    {
      add_positional_condition_and_resource_args(this, args);
      name_ = name;
      fragment_ = fragment;
      spec_ = std::make_shared<OperatorSpec>(fragment);
      setup(*spec_.get());
    }
  };

  /* The python module */

  PYBIND11_MODULE(_dds_hid_subscriber, m)
  {
    m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _dds_hid_subscriber
        .. autosummary::
           :toctree: _generate
           DDSHIDSubscriberOp
    )pbdoc";

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::class_<DDSHIDSubscriberOp,
               PyDDSHIDSubscriberOp,
               Operator,
               std::shared_ptr<DDSHIDSubscriberOp>>(
        m, "DDSHIDSubscriberOp", doc::DDSHIDSubscriberOp::doc_DDSHIDSubscriberOp)
        .def(py::init<Fragment *,
                      const py::args &,
                      const std::string &,
                      const std::string &,
                      uint32_t,
                      const std::string &,
                      const std::vector<std::string> &,
                      const std::string &>(),
             "fragment"_a,
             "qos_provider"_a = ""s,
             "participant_qos"_a = ""s,
             "domain_id"_a = 0,
             "reader_qos"_a = ""s,
             "hid_device_filters"_a = std::vector<std::string>{},
             "name"_a = "dds_hid_subscriber"s,
             doc::DDSHIDSubscriberOp::doc_DDSHIDSubscriberOp)
        .def("initialize", &DDSHIDSubscriberOp::initialize,
             doc::DDSHIDSubscriberOp::doc_initialize)
        .def("setup", &DDSHIDSubscriberOp::setup, "spec"_a,
             doc::DDSHIDSubscriberOp::doc_setup);

    export_enum<HIDDeviceType>(m, "HIDDeviceType");

    py::class_<InputEvent> input_event(
        m, "InputEvent");
    input_event.def(py::init<>())
        .def_readwrite("device_type", &InputEvent::device_type)
        .def_readwrite("event_type", &InputEvent::event_type)
        .def_readwrite("number", &InputEvent::number)
        .def_readwrite("value", &InputEvent::value);

    m.def("register_types", [](EmitterReceiverRegistry &registry)
          {
            registry.add_emitter_receiver<std::vector<InputEvent>>(
                "std::vector<InputEvent>"s);
            registry.add_emitter_receiver<HIDDeviceType>("HIDDeviceType"s);
          });
  } // PYBIND11_MODULE NOLINT
} // namespace holoscan::ops
