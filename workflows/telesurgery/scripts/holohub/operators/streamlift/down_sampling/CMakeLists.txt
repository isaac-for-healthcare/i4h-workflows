# Operator Lib
add_library(streamlift_downsampler SHARED down_sampler.cpp)
target_link_libraries(streamlift_downsampler PRIVATE holoscan::core cuda streamlift_server)

# Python Lib/Module
add_subdirectory(python)
