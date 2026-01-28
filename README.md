# GPU-Accelerated Batch Image Processing with CUDA

## Project Overview

This project implements GPU-accelerated image processing using custom CUDA kernels to apply filters (Gaussian blur and Sobel edge detection) to hundreds of images in parallel.

### Key Features
- Custom CUDA kernels for image processing
- Gaussian blur filter (5x5 kernel)
- Sobel edge detection
- Batch processing of 200+ images
- Performance metrics and benchmarking

### Technologies Used
- **CUDA**: Custom GPU kernels
- **C++17**: Host code
- **STB Image Library**: Image I/O
- **CMake**: Build system

## Dataset

- **Source**: USC SIPI Image Database (https://sipi.usc.edu/database/)
- **Size**: 250 grayscale images
- **Resolution**: Various (512x512, 256x256)
- **Format**: TIFF/JPEG

## Performance Results

*Results from Google Colab with NVIDIA Tesla T4:*

- **Total Images**: 250
- **Processing Time**: ~3.5 seconds
- **Average Time per Image**: ~14 ms
- **Throughput**: ~71 images/second

### GPU Specifications
- **Device**: NVIDIA Tesla T4
- **Memory**: 15 GB GDDR6
- **Compute Capability**: 7.5

## Building and Running

### Requirements
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make
```

### Running
```bash
# Download sample images first
python3 download_images.py

# Run processing
./build/image_processor
```

## Project Structure
```
cuda-image-processing/
├── CMakeLists.txt
├── README.md
├── download_images.py
├── src/
│   ├── main.cu
│   ├── kernels.cu
│   ├── kernels.cuh
│   ├── stb_image.h
│   └── stb_image_write.h
├── input_images/
│   └── (250 images)
├── output_images/
│   └── (processed images)
└── results/
    ├── execution_log.txt
    └── proof_of_execution.png
```

## Implementation Details

### CUDA Kernels

**Gaussian Blur**: 5x5 convolution kernel applied to each pixel in parallel
```cpp
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output,
                                   int width, int height)
```

**Sobel Edge Detection**: Computes gradient magnitude for edge detection
```cpp
__global__ void sobelEdgeDetectionKernel(unsigned char* input, unsigned char* output,
                                         int width, int height)
```

### Parallelization Strategy
- Thread blocks: 16x16 threads
- Grid size: Calculated based on image dimensions
- Each thread processes one pixel

## Author

Your Name - Course Project for CUDA at Scale for the Enterprise

## License

MIT License