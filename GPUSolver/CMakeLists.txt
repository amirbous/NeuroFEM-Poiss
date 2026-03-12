cmake_minimum_required(VERSION 3.18)

project(APU_FEM VERSION 1.0.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

set(TARGET_EXE "poissfem")
set(SOURCES src/main.cpp src/IO.cpp src/ComputeModel.cpp src/Solver.cpp)

add_executable(${TARGET_EXE} ${SOURCES})

# CUDA and OpenMP
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Ginkgo
set(Ginkgo_DIR "/storage/home/abouslama/ginkgo/build/install/lib/cmake/Ginkgo")
find_package(Ginkgo REQUIRED)

# NVML library (explicit path to CUDA stubs)
set(NVML_LIBRARY "/usr/local/cuda/lib64/stubs/libnvidia-ml.so")

# CLI11 (header-only)
include(FetchContent)
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.4.2
)
FetchContent_MakeAvailable(cli11)

# Link libraries
target_link_libraries(${TARGET_EXE} PRIVATE
    CUDA::cublas
    CUDA::cudart
    Ginkgo::ginkgo
    OpenMP::OpenMP_CXX
    ${NVML_LIBRARY}
    CLI11::CLI11
)

# Include directories
target_include_directories(${TARGET_EXE} PRIVATE
    ${PROJECT_SOURCE_DIR}
    # If you have cudss headers, set CUDSS_INCLUDE_DIR above
    # ${CUDSS_INCLUDE_DIR}
)

# Compile options
target_compile_options(${TARGET_EXE} PRIVATE
    $<$<CONFIG:Debug>:-O0 -g3>
)

# Output directory
set_target_properties(${TARGET_EXE} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Copy data folder
file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
