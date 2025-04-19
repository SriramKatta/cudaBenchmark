cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

project(stream LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)

set(POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)

include(cmake/CPM.cmake)
include(cmake/boost.cmake)
include(cmake/fmt.cmake)
include(cmake/NVTX.cmake)

include_directories(../../Common)

include_directories(header)
add_subdirectory(utilities)
add_subdirectory(implementation)
