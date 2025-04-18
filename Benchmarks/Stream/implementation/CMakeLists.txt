set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC testkernels)
target_link_libraries(${PROJECT_NAME} PUBLIC utilities)
