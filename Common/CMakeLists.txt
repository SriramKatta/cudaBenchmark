set(LIB_NAME cudahelper)

find_package(CUDAToolkit REQUIRED)

add_library(${LIB_NAME} INTERFACE)
target_include_directories(${LIB_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIB_NAME} INTERFACE CUDA::cudart)