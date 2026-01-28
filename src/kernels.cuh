#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Kernel declarations
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output, 
                                   int width, int height);

__global__ void sobelEdgeDetectionKernel(unsigned char* input, unsigned char* output,
                                         int width, int height);

__global__ void grayscaleToRGBKernel(unsigned char* input, unsigned char* output,
                                     int width, int height);

// Host functions
void processImageGPU(unsigned char* h_input, unsigned char* h_output,
                     int width, int height, const char* filterType);

void processBatchGPU(unsigned char** h_inputs, unsigned char** h_outputs,
                     int* widths, int* heights, int numImages, const char* filterType);

#endif