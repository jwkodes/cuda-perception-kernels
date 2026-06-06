cmake_minimum_required(VERSION 3.18) # min version required for CUDA as a first class language
project(cuda-perception-kernels LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV 4.5 REQUIRED)
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

add_library(perception_kernels STATIC src/bilateral_filter.cu)
target_include_directories(perception_kernels PUBLIC include)

# benchmark
add_executable(run_benchmarks benchmarks/bench_bilateral_filter.cu)
target_link_libraries(run_benchmarks PRIVATE 
    benchmark::benchmark 
    ${OpenCV_LIBS}
    perception_kernels
)

# test
add_executable(run_tests tests/test_bilateral_filter.cu)
target_link_libraries(run_tests PRIVATE 
    GTest::GTest GTest::Main 
    ${OpenCV_LIBS}
    perception_kernels
)