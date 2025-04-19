cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(STREAM_BENCHMARK LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES  80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")


include(cmake/CPM.cmake)
include(cmake/cccl.cmake)

# Include directories and libraries
include_directories(../../Common)
include_directories(headers)

# Source file
# Add target for stream
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable( ${PROJECT_NAME} main.cpp)
target_link_libraries( ${PROJECT_NAME} PRIVATE CUDA::cudart)
target_link_libraries( ${PROJECT_NAME} PRIVATE CCCL::CUB)
set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP REQUIRED)
if(OPENMP_CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CUDA) 
    #target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${OpenMP_CXX_FLAGS}>)
endif(OPENMP_CUDA_FOUND)

