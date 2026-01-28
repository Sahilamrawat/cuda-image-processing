
# CUDA Image Processing Project

## Overview
GPU-accelerated batch image processing using custom CUDA kernels for 
Gaussian blur filtering on 250 grayscale images.

## Files
- execution_log.txt: Complete console output
- performance_summary.txt: Performance metrics
- proof_of_execution.png: Visual comparison of input/output
- sample_outputs/: 10 sample processed images

## Key Results
- Successfully processed 250 images on NVIDIA Tesla T4
- Achieved 26.35 images/second throughput
- 15.8x speedup compared to CPU implementation
- Custom CUDA kernels with 16×16 thread blocks

## Technologies
- CUDA 12.5
- Custom kernels for Gaussian blur
- STB Image library for I/O
- Google Colab for execution
