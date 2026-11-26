# C++ port of the NumPy flow-matching policy (M3). C++17, std only (+ maybe Eigen later), no ML libraries.
cmake_minimum_required(VERSION 3.16)
project(flowpolicy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(FP_NATIVE "Build with -O3 -march=native (use on the target machine, e.g. the Pi 5)" OFF)
if(FP_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

# Library: your sources go in src/ and include/flowpolicy/ and get added here.
add_library(flowpolicy
  src/fpbin.cpp
  src/tensor.cpp
  src/film.cpp
  src/policy.cpp
)
target_include_directories(flowpolicy PUBLIC include)
target_compile_options(flowpolicy PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

# Test harness: compares the C++ forward pass with the NumPy test vectors.
add_executable(test_vectors tests/test_vectors.cpp)
target_link_libraries(test_vectors PRIVATE flowpolicy)
target_compile_options(test_vectors PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
target_compile_definitions(test_vectors PRIVATE
  FP_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/weights/film_g.bin"
  FP_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/vectors.bin")

enable_testing()
add_test(NAME test_vectors COMMAND test_vectors)

# Policy behind a stdin/stdout pipe, for the closed-loop eval in Python (python/scripts/m3_eval_cpp.py).
add_executable(policy_server tools/policy_server.cpp)
target_link_libraries(policy_server PRIVATE flowpolicy)
target_compile_options(policy_server PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
target_compile_definitions(policy_server PRIVATE FP_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/weights/film_g.bin")

# Batch-1 latency benchmark (median over 1000 decisions, for 1 / 2 / 5 / 10 Euler steps).
add_executable(bench tools/bench.cpp)
target_link_libraries(bench PRIVATE flowpolicy)
target_compile_options(bench PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
target_compile_definitions(bench PRIVATE
  FP_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/weights/film_g.bin"
  FP_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors/vectors.bin"
  FP_FLAGS="${CMAKE_CXX_FLAGS_RELEASE}")
