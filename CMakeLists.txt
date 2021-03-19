cmake_minimum_required(VERSION 3.17)
project(supercomp_lab_1 CUDA)

set(CMAKE_CUDA_STANDARD 17)
set_property(TARGET PROPERTY CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
add_executable(supercomp_lab_1 main.cu)
target_link_directories(supercomp_lab_1 PUBLIC "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\lib\\x64")

set_target_properties(supercomp_lab_1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(supercomp_lab_1 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_precompile_headers(supercomp_lab_1 PUBLIC <cstdio> [["cuda_runtime.h"]] [["device_launch_parameters.h"]] [["curand.h"]])
target_link_libraries(supercomp_lab_1 PUBLIC curand)