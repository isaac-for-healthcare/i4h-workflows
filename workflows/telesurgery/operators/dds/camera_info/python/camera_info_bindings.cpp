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
#include <pybind11/complex.h>
#include <rti/core/BoundedSequence.hpp>

#include "CameraInfo.hpp"
#include "holoscan/python/core/emitter_receiver_registry.hpp"
#include "camera_info_bindings.hpp"

namespace py = pybind11;

// Custom type caster for rti::core::bounded_sequence
// This allows pybind11 to automatically convert between Python lists
// and rti::core::bounded_sequence<T, N>.
namespace pybind11
{
    namespace detail
    {

        template <typename T, size_t N>
        struct type_caster<rti::core::bounded_sequence<T, N>>
        {
        public:
            using SequenceType = rti::core::bounded_sequence<T, N>;

            // Macro to define the name displayed in error messages
            PYBIND11_TYPE_CASTER(SequenceType, const_name("rti::core::bounded_sequence"));

            // Conversion from Python object (list) to C++ bounded_sequence
            bool load(handle src, bool convert)
            {
                // Check if the source Python object is a sequence (like list or tuple)
                // but not a string or bytes object, which are also sequences.
                if (!py::isinstance<py::sequence>(src) || py::isinstance<py::str>(src) || py::isinstance<py::bytes>(src))
                {
                    return false;
                }

                auto seq = py::reinterpret_borrow<py::sequence>(src);
                // Check if the Python sequence size exceeds the bound N
                if (seq.size() > N)
                {
                    // Throw an error if the sequence is too large
                    py::pybind11_fail("Sequence is too large for bounded_sequence<" + std::string(typeid(T).name()) + ", bound=" + std::to_string(N) + ">");
                    // Although fail throws, return false for logical completeness
                    return false;
                }

                // Resize the C++ sequence to match the Python sequence size
                value.resize(static_cast<typename SequenceType::size_type>(seq.size()));

                size_t index = 0;
                // Iterate through the Python sequence elements
                for (auto it : seq)
                {
                    // Use the existing pybind11 caster for the element type T
                    make_caster<T> elem_caster;
                    // Attempt to load/convert the Python element to C++ type T
                    if (!elem_caster.load(it, convert))
                    {
                        // Failed to convert an element
                        return false;
                    }
                    // Move or copy the converted element into the C++ sequence
                    // cast_op<T> handles potential moves for efficiency
                    value[static_cast<typename SequenceType::size_type>(index++)] = cast_op<T>(std::move(elem_caster));
                }
                // Successfully loaded all elements
                return true;
            }

            // Conversion from C++ bounded_sequence to Python object (list)
            static handle cast(const SequenceType &src, return_value_policy policy, handle parent)
            {
                // Create a new Python list with the same size as the C++ sequence
                py::list list(src.size());
                size_t index = 0;
                // Iterate through the C++ sequence elements
                for (const auto &element : src)
                {
                    // Cast the C++ element to a Python object using pybind11's cast
                    // and insert it into the Python list. Respect the return value policy.
                    list[index++] = py::cast(element, policy, parent);
                }
                // Release ownership of the Python list to pybind11
                return list.release();
            }
        };

    }
} // namespace pybind11::detail

// Custom type caster for std::vector<uint8_t>
// Handles conversion between Python bytes/list[int] and std::vector<uint8_t>
namespace pybind11
{
    namespace detail
    {

        template <>
        struct type_caster<std::vector<uint8_t>>
        {
        public:
            using VectorType = std::vector<uint8_t>;

            PYBIND11_TYPE_CASTER(VectorType, const_name("bytes | list[int]"));

            // Conversion from Python bytes/list object to C++ std::vector<uint8_t>
            bool load(handle src, bool convert)
            {
                if (py::isinstance<py::bytes>(src))
                {
                    // Input is Python bytes object
                    py::buffer_info info(py::buffer(py::reinterpret_borrow<py::object>(src)).request());
                    const uint8_t *buffer_ptr = static_cast<const uint8_t *>(info.ptr);
                    value.assign(buffer_ptr, buffer_ptr + info.size);
                    return true;
                }
                if (py::isinstance<py::sequence>(src) && !py::isinstance<py::str>(src))
                {
                    // Input is a sequence (list/tuple) but not str
                    auto seq = py::reinterpret_borrow<py::sequence>(src);
                    value.reserve(seq.size());
                    for (auto it : seq)
                    {
                        // Use make_caster to handle potential Python int -> uint8_t conversion
                        make_caster<uint8_t> elem_caster;
                        if (!elem_caster.load(it, convert))
                        {
                            return false; // Element conversion failed
                        }
                        value.push_back(cast_op<uint8_t>(std::move(elem_caster)));
                    }
                    return true;
                }
                // Unsupported type
                return false;
            }

            // Conversion from C++ std::vector<uint8_t> to Python bytes object
            static handle cast(const VectorType &src, return_value_policy /* policy */, handle /* parent */)
            {
                // Create Python bytes object directly from the vector's data
                return py::bytes(reinterpret_cast<const char *>(src.data()), src.size()).release();
            }
        };

    }
} // namespace pybind11::detail

namespace holoscan::ops
{

    // Function to perform the actual binding registration
    void register_camera_info_bindings(py::module_ &m)
    {
        py::class_<::CameraInfo> camera_info(
            m, "CameraInfo", "Data structure for camera information transmitted over DDS.");
        camera_info.def(py::init<>())
            .def_property("frame_num",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::frame_num),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::frame_num))
            .def_property("width",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::width),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::width))
            .def_property("height",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::height),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::height))
            .def_property("channels",
                          static_cast<const int8_t &(::CameraInfo::*)() const>(&::CameraInfo::channels),
                          static_cast<void (::CameraInfo::*)(int8_t)>(&::CameraInfo::channels))
            .def_property("data",
                          static_cast<const std::vector<uint8_t> &(::CameraInfo::*)() const>(&::CameraInfo::data),
                          static_cast<void (::CameraInfo::*)(const std::vector<uint8_t> &)>(&::CameraInfo::data))
            .def_property("joint_names",
                          static_cast<const ::rti::core::bounded_sequence<std::string, 6L> &(::CameraInfo::*)() const>(&::CameraInfo::joint_names),
                          static_cast<void (::CameraInfo::*)(const ::rti::core::bounded_sequence<std::string, 6L> &)>(&::CameraInfo::joint_names))
            .def_property("joint_positions",
                          static_cast<const ::rti::core::bounded_sequence<double, 6L> &(::CameraInfo::*)() const>(&::CameraInfo::joint_positions),
                          static_cast<void (::CameraInfo::*)(const ::rti::core::bounded_sequence<double, 6L> &)>(&::CameraInfo::joint_positions))
            .def_property("hid_capture_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::hid_capture_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::hid_capture_timestamp))
            .def_property("hid_publish_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::hid_publish_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::hid_publish_timestamp))
            .def_property("hid_receive_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::hid_receive_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::hid_receive_timestamp))
            .def_property("hid_to_sim_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::hid_to_sim_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::hid_to_sim_timestamp))
            .def_property("hid_process_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::hid_process_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::hid_process_timestamp))
            .def_property("video_acquisition_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_acquisition_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_acquisition_timestamp))
            .def_property("video_data_bridge_enter_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_data_bridge_enter_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_data_bridge_enter_timestamp))
            .def_property("video_data_bridge_emit_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_data_bridge_emit_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_data_bridge_emit_timestamp))
            .def_property("video_encoder_enter_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_encoder_enter_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_encoder_enter_timestamp))
            .def_property("video_encoder_emit_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_encoder_emit_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_encoder_emit_timestamp))
            .def_property("video_publisher_enter_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_publisher_enter_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_publisher_enter_timestamp))
            .def_property("video_publisher_emit_timestamp",
                          static_cast<const int64_t &(::CameraInfo::*)() const>(&::CameraInfo::video_publisher_emit_timestamp),
                          static_cast<void (::CameraInfo::*)(int64_t)>(&::CameraInfo::video_publisher_emit_timestamp))
            .def_property("message_id",
                          static_cast<const uint64_t &(::CameraInfo::*)() const>(&::CameraInfo::message_id),
                          static_cast<void (::CameraInfo::*)(uint64_t)>(&::CameraInfo::message_id))
            .def_property("frame_size",
                          static_cast<const uint64_t &(::CameraInfo::*)() const>(&::CameraInfo::frame_size),
                          static_cast<void (::CameraInfo::*)(uint64_t)>(&::CameraInfo::frame_size))
            .def_property("encoded_frame_size",
                          static_cast<const uint64_t &(::CameraInfo::*)() const>(&::CameraInfo::encoded_frame_size),
                          static_cast<void (::CameraInfo::*)(uint64_t)>(&::CameraInfo::encoded_frame_size));

        m.def("register_types", [](EmitterReceiverRegistry &registry)
              { registry.add_emitter_receiver<CameraInfo>("CameraInfo"s); });
    }

} // namespace holoscan::ops
