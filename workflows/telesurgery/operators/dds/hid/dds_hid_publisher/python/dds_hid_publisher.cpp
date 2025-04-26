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

#include <cstdint>
#include <memory>
#include <string>

#include "../dds_hid_publisher.hpp"
#include "./dds_hid_publisher_pydoc.hpp"

#include "../../../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyDDSHIDPublisherOp : public DDSHIDPublisherOp {
 public:
  /* Inherit the constructors */
  using DDSHIDPublisherOp::DDSHIDPublisherOp;

  // Define a constructor that fully initializes the object.
  PyDDSHIDPublisherOp(Fragment* fragment, const py::args& args,
                      const std::string& qos_provider = "", const std::string& participant_qos = "",
                      uint32_t domain_id = 0, const std::string& writer_qos = "",
                      const HIDevicesConfig& hid_devices = {},
                      const std::string& name = "dds_hid_publisher")
      : DDSHIDPublisherOp(ArgList{Arg{"qos_provider", qos_provider},
                                  Arg{"participant_qos", participant_qos},
                                  Arg{"domain_id", domain_id},
                                  Arg{"writer_qos", writer_qos},
                                  Arg{"hid_devices", hid_devices}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_dds_hid_publisher, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _dds_hid_publisher
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<DDSHIDPublisherOp, PyDDSHIDPublisherOp, Operator, std::shared_ptr<DDSHIDPublisherOp>>(
      m, "DDSHIDPublisherOp", doc::DDSHIDPublisherOp::doc_DDSHIDPublisherOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    const std::string&,
                    const HIDevicesConfig&,
                    const std::string&>(),
           "fragment"_a,
           "qos_provider"_a = ""s,
           "participant_qos"_a = ""s,
           "domain_id"_a = 0,
           "writer_qos"_a = ""s,
           "hid_devices"_a = HIDevicesConfig{},
           "name"_a = "dds_hid_publisher"s,
           doc::DDSHIDPublisherOp::doc_DDSHIDPublisherOp)
      .def("initialize", &DDSHIDPublisherOp::initialize, doc::DDSHIDPublisherOp::doc_initialize)
      .def("setup", &DDSHIDPublisherOp::setup, "spec"_a, doc::DDSHIDPublisherOp::doc_setup)
      .def("start",
           &DDSHIDPublisherOp::start,
           doc::DDSHIDPublisherOp::doc_start,
           py::call_guard<py::gil_scoped_release>())
      .def("stop",
           &DDSHIDPublisherOp::stop,
           doc::DDSHIDPublisherOp::doc_stop,
           py::call_guard<py::gil_scoped_release>());
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
