set(TESTKERNELS testkernels)

file(GLOB HFILES include/*.hpp)
file(GLOB SRCFILES src/*.cpp)

set_source_files_properties(${SRCFILES} PROPERTIES LANGUAGE CUDA)

add_library( ${TESTKERNELS} ${SRCFILES} ${HFILES})
target_include_directories( ${TESTKERNELS} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${TESTKERNELS} PUBLIC fmt Boost::program_options)
