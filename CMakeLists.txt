cmake_minimum_required(VERSION 3.24)
# RTX 3060 Laptop uses SM 86. A fixed default also allows compilation in a
# restricted build session; override CMAKE_CUDA_ARCHITECTURES for another GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA target architectures")
endif()
project(DCLens LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)
find_package(CUDAToolkit REQUIRED)
pybind11_add_module(_cuda
    DCLens/src/bindings.cu DCLens/src/engine.cu DCLens/src/p3m.cu
    DCLens/src/ipm.cu DCLens/src/rasterizer.cu DCLens/src/lightcurve_extractor.cu)
target_link_libraries(_cuda PRIVATE OpenGL::OpenGL OpenGL::EGL CUDA::cufft CUDA::cudart)
target_compile_options(_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-O3>)
if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/DCLens")
endif()
install(TARGETS _cuda LIBRARY DESTINATION DCLens)
install(DIRECTORY DCLens/ DESTINATION DCLens FILES_MATCHING PATTERN "*.py")
