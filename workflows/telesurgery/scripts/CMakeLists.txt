cmake_minimum_required(VERSION 3.20)
project(telesurgery)

# Find the Holoscan package
find_package(holoscan REQUIRED)

set(FETCH_HOLOHUB_OPERATOR_URL "https://raw.githubusercontent.com/nvidia-holoscan/holohub/refs/heads/main/cmake/FetchHolohubOperator.cmake")
set(FETCH_HOLOHUB_OPERATOR_LOCAL_PATH "${CMAKE_CURRENT_BINARY_DIR}/FetchHolohubOperator.cmake")
if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
     file(DOWNLOAD
         ${FETCH_HOLOHUB_OPERATOR_URL}
         ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH}
         SHOW_PROGRESS
         TLS_VERIFY ON
     )
     if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
         message(FATAL_ERROR "Failed to download FetchHolohubOperator.cmake from ${FETCH_HOLOHUB_OPERATOR_URL}")
     endif()
endif()
include(${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})

# Fetch the specific operator you need
fetch_holohub_operator(aja_source)

install(TARGETS aja_source
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/holohub/operators/camera/aja_source/lib")
install(TARGETS aja_source_python
    LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/holohub/operators/camera/aja_source/")
