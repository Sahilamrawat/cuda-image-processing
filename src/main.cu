#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <chrono>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "kernels.cuh"

using namespace std;
using namespace std::chrono;

// Helper to check if path is directory
bool isDirectory(const char* path) {
    struct stat statbuf;
    if (stat(path, &statbuf) != 0) return false;
    return S_ISDIR(statbuf.st_mode);
}

// Get all image files from directory
vector<string> getImageFiles(const char* directory) {
    vector<string> files;
    DIR* dir = opendir(directory);
    if (!dir) {
        printf("Error opening directory: %s\n", directory);
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            string filename = entry->d_name;
            string lower = filename;
            for (auto& c : lower) c = tolower(c);
            
            if (lower.find(".jpg") != string::npos || 
                lower.find(".jpeg") != string::npos || 
                lower.find(".png") != string::npos) {
                files.push_back(string(directory) + "/" + filename);
            }
        }
    }
    closedir(dir);
    return files;
}

int main(int argc, char** argv) {
    printf("=== CUDA Image Processing ===\n\n");
    
    // Configuration
    const char* inputDir = "input_images";
    const char* outputDir = "output_images";
    const char* filterType = "blur"; // or "edge"
    
    // Create output directory
    mkdir(outputDir, 0755);
    
    // Get image files
    vector<string> imageFiles = getImageFiles(inputDir);
    int numImages = imageFiles.size();
    
    if (numImages == 0) {
        printf("No images found in %s\n", inputDir);
        return 1;
    }
    
    printf("Found %d images\n", numImages);
    printf("Filter type: %s\n\n", filterType);
    
    // GPU info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    // Process images
    auto start = high_resolution_clock::now();
    
    int processedCount = 0;
    for (const auto& imagePath : imageFiles) {
        // Load image
        int width, height, channels;
        unsigned char* imageData = stbi_load(imagePath.c_str(), &width, &height, &channels, 1);
        
        if (!imageData) {
            printf("Failed to load: %s\n", imagePath.c_str());
            continue;
        }
        
        // Allocate output
        unsigned char* outputData = (unsigned char*)malloc(width * height);
        
        // Process on GPU
        processImageGPU(imageData, outputData, width, height, filterType);
        
        // Save output
        string filename = imagePath.substr(imagePath.find_last_of("/") + 1);
        string outputPath = string(outputDir) + "/" + filename;
        stbi_write_jpg(outputPath.c_str(), width, height, 1, outputData, 95);
        
        free(imageData);
        free(outputData);
        
        processedCount++;
        if (processedCount % 10 == 0) {
            printf("Processed %d/%d images...\r", processedCount, numImages);
            fflush(stdout);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    printf("\n\n=== Results ===\n");
    printf("Total images processed: %d\n", processedCount);
    printf("Total time: %.2f seconds\n", duration.count() / 1000.0);
    printf("Average time per image: %.2f ms\n", (float)duration.count() / processedCount);
    printf("Throughput: %.2f images/second\n", processedCount / (duration.count() / 1000.0));
    
    return 0;
}