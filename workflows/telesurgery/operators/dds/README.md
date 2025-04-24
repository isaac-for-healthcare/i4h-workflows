# DDS Operators for Holoscan

This module provides DDS (Data Distribution Service) operators for the Holoscan platform, enabling communication between different parts of a distributed system using the DDS middleware.

## Features


## Dependencies

- Holoscan SDK
- RTI Connext DDS

## Usage

### In CMake

To use these operators in your application, add the following to your CMakeLists.txt:

```cmake
find_package(holoscan REQUIRED)

# Link against the DDS operators
target_link_libraries(your_application
  PRIVATE
  holoscan::ops::dds
)
```

### In C++ Code

#### Publisher

```cpp
#include <holoscan/holoscan.hpp>
#include <dds_publisher.hpp>

// Inside your application
auto dds_publisher = make_operator<holoscan::ops::DDSPublisher>(
    "dds_publisher",
    Config{
        {"domain_id", 0},
        {"topic_name", "your_topic"},
        {"input_types", std::vector<std::string>{"input"}}
    });

// Add the operator to your fragment
fragment.add_operator(dds_publisher);
```

#### Subscriber

```cpp
#include <holoscan/holoscan.hpp>
#include <dds_subscriber.hpp>

// Inside your application
auto dds_subscriber = make_operator<holoscan::ops::DDSSubscriber>(
    "dds_subscriber",
    Config{
        {"domain_id", 0},
        {"topic_name", "your_topic"},
        {"output_name", "output"},
        {"timeout_ms", 100},
        {"allocator", allocator} // Pass your allocator resource
    });

// Add the operator to your fragment
fragment.add_operator(dds_subscriber);
```

## Configuration Parameters

### DDSPublisher

- `domain_id`: DDS domain ID (default: 0)
- `topic_name`: Name of the DDS topic to publish to (default: "default_topic")
- `input_types`: List of input types to publish (default: ["input"])

### DDSSubscriber

- `domain_id`: DDS domain ID (default: 0)
- `topic_name`: Name of the DDS topic to subscribe to (default: "default_topic")
- `output_name`: Name for the output port (default: "output")
- `timeout_ms`: Timeout in milliseconds for receiving data (default: 100)
- `allocator`: Allocator resource for memory management (required)
