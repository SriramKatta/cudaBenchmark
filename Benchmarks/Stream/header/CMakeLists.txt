set(LIBRARAY_NAME gpustream)
# set_source_files_properties(include/kernels.hpp PROPERTIES LANGUAGE CUDA)
add_library(${LIBRARAY_NAME} INTERFACE)
target_include_directories(${LIBRARAY_NAME}
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${LIBRARAY_NAME} INTERFACE fmt)
target_link_libraries(${LIBRARAY_NAME} INTERFACE nvtx3-c)
target_link_libraries(${LIBRARAY_NAME} INTERFACE cudahelper)
target_link_libraries(${LIBRARAY_NAME} INTERFACE CUDA::cudart)
