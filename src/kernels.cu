#include "kernels.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Gaussian Blur Kernel (5x5)
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // 5x5 Gaussian kernel (normalized)
    float kernel[5][5] = {
        {1,  4,  7,  4,  1},
        {4, 16, 26, 16,  4},
        {7, 26, 41, 26,  7},
        {4, 16, 26, 16,  4},
        {1,  4,  7,  4,  1}
    };
    float kernelSum = 273.0f;
    
    float sum = 0.0f;
    
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            
            sum += input[py * width + px] * kernel[ky + 2][kx + 2];
        }
    }
    
    output[y * width + x] = (unsigned char)(sum / kernelSum);
}

// Sobel Edge Detection Kernel
__global__ void sobelEdgeDetectionKernel(unsigned char* input, unsigned char* output,
                                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height || x == 0 || y == 0 || 
        x == width - 1 || y == height - 1) {
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
        return;
    }
    
    // Sobel operators
    int Gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
             -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
             -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
             
    int Gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
             +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
    
    int magnitude = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
    output[y * width + x] = min(magnitude, 255);
}

// Process single image
void processImageGPU(unsigned char* h_input, unsigned char* h_output,
                     int width, int height, const char* filterType) {
    size_t imageSize = width * height * sizeof(unsigned char);
    
    unsigned char *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    if (strcmp(filterType, "blur") == 0) {
        gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    } else if (strcmp(filterType, "edge") == 0) {
        sobelEdgeDetectionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
}

// Process batch of images
void processBatchGPU(unsigned char** h_inputs, unsigned char** h_outputs,
                     int* widths, int* heights, int numImages, const char* filterType) {
    for (int i = 0; i < numImages; i++) {
        processImageGPU(h_inputs[i], h_outputs[i], widths[i], heights[i], filterType);
    }
}