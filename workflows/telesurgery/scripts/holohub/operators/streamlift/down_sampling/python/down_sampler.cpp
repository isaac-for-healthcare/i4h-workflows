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

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "../down_sampler.hpp"
#include "./down_sampler_pydoc.hpp"
#include "operator_util.hpp"


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

class PyStreamLiftDownSamplerOp : public StreamLiftDownSamplerOp {
 public:
  /* Inherit the constructors */
  using StreamLiftDownSamplerOp::StreamLiftDownSamplerOp;

  // Define a constructor that fully initializes the object.
  PyStreamLiftDownSamplerOp(Fragment* fragment, const py::args& args, std::shared_ptr<::holoscan::Allocator> allocator,
                     uint32_t cuda_device_ordinal, uint32_t width, uint32_t height,
                     const std::string& output_type,
                     const std::string& name = "streamlift_downsampler")
      : StreamLiftDownSamplerOp(ArgList{
                                 Arg{"allocator", allocator},
                                 Arg{"cuda_device_ordinal", cuda_device_ordinal},
                                 Arg{"width", width},
                                 Arg{"height", height},
                                 Arg{"output_type", output_type}
                                 }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_streamlift_downsampler, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _streamlift_downsampler
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<StreamLiftDownSamplerOp, PyStreamLiftDownSamplerOp, Operator, std::shared_ptr<StreamLiftDownSamplerOp>>(
      m, "StreamLiftDownSamplerOp", doc::StreamLiftDownSamplerOp::doc_StreamLiftDownSamplerOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<::holoscan::Allocator>,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    const std::string&,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "cuda_device_ordinal"_a = 0,
           "width"_a = 0,
           "height"_a = 0,
           "output_type"_a = "tensor"s,
           "name"_a = "streamlift_downsampler"s,
           doc::StreamLiftDownSamplerOp::doc_StreamLiftDownSamplerOp)
      .def("initialize", &StreamLiftDownSamplerOp::initialize, doc::StreamLiftDownSamplerOp::doc_initialize)
      .def("setup", &StreamLiftDownSamplerOp::setup, "spec"_a, doc::StreamLiftDownSamplerOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
