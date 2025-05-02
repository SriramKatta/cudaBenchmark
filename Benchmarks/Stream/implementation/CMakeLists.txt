#set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(${PROJECT_NAME} main.cu)
#target_link_libraries(${PROJECT_NAME} PUBLIC testkernels)
target_link_libraries(${PROJECT_NAME} PUBLIC utilities)
target_link_libraries(${PROJECT_NAME} PUBLIC gpustream)
target_link_libraries(${PROJECT_NAME} PUBLIC cudahelper)

