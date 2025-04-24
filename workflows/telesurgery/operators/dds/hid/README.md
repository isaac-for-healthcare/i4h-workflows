### DDS Human Interface Device (HID) Operators

The DDS Human Interface Device (HID) Operators allow applications to read or write
Human Interface Device (HID) events to a DDS databus, enabling communication with
other applications via the [InputCommand](InputCommand.idl) DDS topic.

This operator requires an installation of [RTI Connext](https://content.rti.com/l/983311/2024-04-30/pz1wms)
to provide access to the DDS domain, as specified by the [OMG Data-Distribution Service](https://www.omg.org/omg-dds-portal/)

#### `holoscan::ops::DDSHIDPublisher`

Operator class for the DDS HID publisher. This operator opens and reads from the specified input device and publishes each event to DDS as a [InputCommand](InputCommand.idl).

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`writer_qos`**: The name of the QoS profile to use for the DDS DataWriter
  - type: `std::string`
- **`hid_devices`**: A list of HID devices to read from.
  - type: `holoscan::ops::HIDevicesConfig`


#### `holoscan::ops::DDSHIDSubscriberOp`

Operator class for the DDS HID subscriber. This operator reads from the
[InputCommand](InputCommand.idl) DDS topic and outputs received events as a vector of [InputCommand](InputCommand.idl) objects.

This operator also inherits the parameters from [DDSOperatorBase](../base/README.md).

##### Parameters

- **`reader_qos`**: The name of the QoS profile to use for the DDS DataReader
  - type: `std::string`
- **`hid_device_filters`**: A list of HID devices to subscribe to.
  - type: `std::vector<std::string>`

##### Outputs

- **`output`**: Output vector of [InputCommand](InputCommand.idl) objects
  - type: `std::vector<InputCommand>`

#### `holoscan::ops::HIDRendererOp`

Operator class for processing HID events, and create tensors with specs for the Holoviz to render.

##### Parameters

- **`allocator`**: The allocator to use for the output tensors.
  - type: `std::shared_ptr<holoscan::Allocator>`
- **`tensors`**: A list of tensors to create. This can be the same configuration as the Holoviz operator.
  - type: `std::vector<holoscan::ops::HolovizOp::InputSpec>`
- **`width`**: The width of the Holoviz window.
  - type: `int32_t`
- **`height`**: The height of the Holoviz window.
  - type: `int32_t`

##### Inputs

- **`input`**: Input vector of [InputCommand](InputCommand.idl) objects
  - type: `std::vector<InputCommand>`

##### Outputs

- **`outputs`**: GXF Entity with tensors for Holoviz
  - type: `gxf::Entity`
- **`output_specs`**: A list of tensor specs for Holoviz
  - type: `std::vector<holoscan::ops::HolovizOp::InputSpec>`
