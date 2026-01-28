# GPU-Accelerated Batch Image Processing with CUDA

## Project Overview
GPU-accelerated image processing using custom CUDA kernels for Gaussian blur filtering on large batches of images.

## Repository Structure
```
cuda-image-processing/
├── src/
│   ├── main.cu              # Main program with CLI
│   ├── kernels.cu           # CUDA kernel implementations
│   ├── kernels.cuh          # Kernel headers
│   ├── stb_image.h          # Image I/O library
│   └── stb_image_write.h
├── results/                  # Execution results
├── CMakeLists.txt           # Build configuration
└── README.md
```

## Requirements
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compiler
- NVIDIA GPU (Compute Capability 5.0+)

## Building
```bash
git clone https://github.com/Sahilamrawat/cuda-image-processing.git
cd cuda-image-processing
mkdir build && cd build
cmake ..
make -j4
```

## Usage

### Command Line Options
```bash
./image_processor [options]

Options:
  -i, --input <dir>     Input directory (default: input_images)
  -o, --output <dir>    Output directory (default: output_images)
  -f, --filter <type>   Filter: blur or edge (default: blur)
  -h, --help            Show help

Examples:
  ./image_processor
  ./image_processor -i my_images -o results -f blur
  ./image_processor --input test_data --filter edge
```

### Preparing Images
```bash
mkdir input_images
# Add .jpg, .png, or .tiff images to input_images/
./image_processor
```

## Implementation

### CUDA Kernels
- **Gaussian Blur**: 5×5 convolution kernel
- **Thread Configuration**: 16×16 blocks
- **Memory**: Optimized host-device transfers

### Algorithm
Each thread processes one pixel independently using a 5×5 Gaussian kernel with normalized weights. Boundary pixels use clamping.

## Performance

**Hardware**: NVIDIA Tesla T4, 15.83GB, CC 7.5

**Results** (250 images, 512×512 each):
- Total Time: 9.49 seconds
- Throughput: 26.35 images/second
- GPU Speedup: ~15.8× vs CPU

## Code Style
Follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

## Author
Shubham Amrawat - CUDA at Scale Course
