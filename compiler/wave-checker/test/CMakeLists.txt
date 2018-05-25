include_directories (${WaveGraphChecker_SOURCE_DIR}/src $ENV{KAENA_EXT_PATH} $ENV{ARCH_ISA_PATH} $ENV{GTEST_DIR}/include)
link_directories ($ENV{GTEST_DIR}/lib)
link_directories ($ENV{GTEST_DIR}/lib64)
add_executable (merge_addr_test merge_addr_range_test.cpp ${WaveGraphChecker_SOURCE_DIR}/src/meminfo.cpp)
add_executable (meminfo_test meminfo_test.cpp ${WaveGraphChecker_SOURCE_DIR}/src/meminfo.cpp)
add_executable (wc_gtest wc_gtest.cpp)
target_link_libraries(wc_gtest gtest pthread)

set(GOLDEN_LOGS
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_clean1.golden.log
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_act.golden.log
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_raw.golden.log
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_waw.golden.log
   )
foreach( file_i ${GOLDEN_LOGS})
  file(COPY ${file_i} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach(file_i)
