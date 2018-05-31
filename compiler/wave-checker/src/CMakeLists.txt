include_directories ($ENV{KAENA_EXT_PATH} $ENV{ARCH_ISA_PATH})
add_executable(wave_graph_checker main.cpp meminfo.cpp wave_graph.cpp)
target_link_libraries(wave_graph_checker boost_program_options)
