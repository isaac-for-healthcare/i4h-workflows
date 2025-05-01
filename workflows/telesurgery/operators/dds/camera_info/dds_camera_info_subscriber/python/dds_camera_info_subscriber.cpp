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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../dds_camera_info_subscriber.hpp"
#include "./dds_camera_info_subscriber_pydoc.hpp"
#include "../../../../operator_util.hpp"
// #include "../python/camera_info_bindings.hpp" // No longer needed
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

class PyDDSCameraInfoSubscriberOp : public DDSCameraInfoSubscriberOp {
 public:
  /* Inherit the constructors */
  using DDSCameraInfoSubscriberOp::DDSCameraInfoSubscriberOp;

  // Define a constructor that fully initializes the object.
  PyDDSCameraInfoSubscriberOp(Fragment* fragment, const py::args& args,
                        const std::string& qos_provider = "",
                        const std::string& participant_qos = "",
                        uint32_t domain_id = 0,
                        const std::string& reader_qos = "",
                        const std::string& topic = "camera_info",
                        const std::string& name = "dds_camera_info_subscriber")
      : DDSCameraInfoSubscriberOp(ArgList{Arg{"qos_provider", qos_provider},
                                    Arg{"participant_qos", participant_qos},
                                    Arg{"domain_id", domain_id},
                                    Arg{"reader_qos", reader_qos},
                                    Arg{"topic", topic}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_dds_camera_info_subscriber, m) {
  // Import the dedicated bindings module first
  try {
      py::module_::import("_camera_info_bindings"); // Import the dedicated module
  } catch (py::error_already_set &e) {
      if (!e.matches(PyExc_ImportError)) { throw; } // Re-throw if it's not an ImportError
      // Otherwise, swallow the ImportError, maybe the module is already loaded.
  }

  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _dds_camera_info_subscriber
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<DDSCameraInfoSubscriberOp,
             PyDDSCameraInfoSubscriberOp,
             Operator,
             std::shared_ptr<DDSCameraInfoSubscriberOp>>(
      m, "DDSCameraInfoSubscriberOp", doc::DDSCameraInfoSubscriberOp::doc_DDSCameraInfoSubscriberOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::string&,
                    uint32_t,
                    const std::string&,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "qos_provider"_a = ""s,
           "participant_qos"_a = ""s,
           "domain_id"_a = 0,
           "reader_qos"_a = ""s,
           "topic"_a = "camera_info"s,
           "name"_a = "dds_camera_info_subscriber"s,
           doc::DDSCameraInfoSubscriberOp::doc_DDSCameraInfoSubscriberOp)
      .def("initialize", &DDSCameraInfoSubscriberOp::initialize, doc::DDSCameraInfoSubscriberOp::doc_initialize)
      .def("setup", &DDSCameraInfoSubscriberOp::setup, "spec"_a, doc::DDSCameraInfoSubscriberOp::doc_setup);

}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
