cmake_minimum_required(VERSION 3.18-3.23 FATAL_ERROR)

project(stream LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DISABLE_NVTX "Disable NVTX instrumentation" OFF)
if(DISABLE_NVTX)
    add_compile_definitions(NVTX_DISABLE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 17)

set(POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

include(cmake/CPM.cmake)
include(cmake/boost.cmake)
include(cmake/fmt.cmake)
include(cmake/NVTX.cmake)
include(cmake/cccl.cmake)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../Common" "${CMAKE_CURRENT_BINARY_DIR}/Common")

add_subdirectory(utilities)
add_subdirectory(header)
add_subdirectory(implementation)
