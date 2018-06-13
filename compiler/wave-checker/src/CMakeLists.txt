include_directories ($ENV{KAENA_EXT_PATH} $ENV{ARCH_ISA_PATH})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable(wave_graph_checker main.cpp meminfo.cpp wave_graph.cpp
    event_check.cpp)
target_link_libraries(wave_graph_checker boost_program_options)
