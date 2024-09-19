cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(manual_ae LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(manual_ae main.cu nn.cu)

set_target_properties(manual_ae PROPERTIES 
    CUDA_ARCHITECTURES "89"
    )

target_include_directories(manual_ae INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS})

target_link_directories(manual_ae PUBLIC 
    ${CUDA_LIBRARY_DIRS})

target_link_libraries(manual_ae PUBLIC
    ${CUDA_cublas_LIBRARY}
    CUDA::curand
    CUDA::toolkit
    )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(manual_ae PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()