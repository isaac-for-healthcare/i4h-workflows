cmake_minimum_required(VERSION 3.20)
project(pysolum)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
pybind11_add_module(pysolum pysolum.cpp)

set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/solum")

execute_process(
    COMMAND lsb_release -sr
    OUTPUT_VARIABLE UBUNTU_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Download Clarius library and header files
if(NOT EXISTS "${LIB_DIR}/libsolum.so")
    file(MAKE_DIRECTORY ${LIB_DIR})
    file(DOWNLOAD
        "https://github.com/clariusdev/solum/releases/download/v12.0.2/solum-12.0.2-linux.x86_64-gcc_ubuntu_${UBUNTU_VERSION}.zip"
         "${LIB_DIR}/solum.zip"
         )
    # Unzip the file
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIB_DIR}/solum.zip
        WORKING_DIRECTORY ${LIB_DIR}
    )

    # Cleanup
    execute_process(
        COMMAND rm solum.zip
        WORKING_DIRECTORY ${LIB_DIR}
    )
endif()

if(NOT EXISTS "${INCLUDE_DIR}/solum.h")
    file(MAKE_DIRECTORY ${INCLUDE_DIR})
    execute_process(
        COMMAND mv "${LIB_DIR}/solum" "${CMAKE_CURRENT_SOURCE_DIR}/include"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Header files
target_include_directories(pysolum PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/")

# Place lib in lib/ folder
set_target_properties(pysolum PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
)
