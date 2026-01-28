cmake_minimum_required(VERSION 3.18)
project(CUDAImageProcessing CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add executable
add_executable(image_processor
    src/main.cu
    src/kernels.cu
)

# Set CUDA architecture
set_target_properties(image_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)

target_include_directories(image_processor PRIVATE src)