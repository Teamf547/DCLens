cmake_minimum_required(VERSION 3.12)
project(lensing_core_project LANGUAGES CXX CUDA)

find_package(pybind11 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES native)

pybind11_add_module(lensing_core lensing_core.cu)

set_target_properties(lensing_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)