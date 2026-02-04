cmake_minimum_required(VERSION 3.18)


project(APU_FEM VERSION 1.0.0 LANGUAGES CXX CUDA)



if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)

set(TARGET_EXE "poissfem")
set(SOURCES src/main.cpp src/IO.cpp src/ComputeModel.cpp)

add_executable(${TARGET_EXE} ${SOURCES})

find_library(NVML_LIBRARY nvidia-ml)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(Ginkgo_DIR "/usr/local/lib/cmake/Ginkgo")
find_package(Ginkgo REQUIRED)


#find_library(CUDSS_LIB NAMES cudss) in case cudss instead of ginkgo


target_link_libraries(${TARGET_EXE} PRIVATE
    CUDA::cublas
    CUDA::cudart
    Ginkgo::ginkgo
    OpenMP::OpenMP_CXX
    ${NVML_LIBRARY}
)

target_include_directories(${TARGET_EXE} PRIVATE 
    ${PROJECT_SOURCE_DIR} 
    ${CUDSS_INCLUDE_DIR}
)

target_compile_options(${TARGET_EXE} PRIVATE 
    $<$<CONFIG:Debug>:-O0 -g3> 
)

set_target_properties(${TARGET_EXE} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

file(COPY ${PROJECT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
