#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "kernels.cuh"

using namespace std;
using namespace std::chrono;

void PrintUsage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -i, --input <dir>     Input directory (default: input_images)\n");
    printf("  -o, --output <dir>    Output directory (default: output_images)\n");
    printf("  -f, --filter <type>   Filter type: blur or edge (default: blur)\n");
    printf("  -h, --help            Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -i input_images -o output_images -f blur\n", program_name);
}

string ToLower(const string& str) {
    string result = str;
    transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

vector<string> GetImageFiles(const char* directory) {
    vector<string> files;
    DIR* dir = opendir(directory);
    if (!dir) {
        printf("Error: Cannot open directory '%s'\n", directory);
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            string filename = entry->d_name;
            string lower = ToLower(filename);
            
            if (lower.find(".jpg") != string::npos || 
                lower.find(".jpeg") != string::npos || 
                lower.find(".png") != string::npos ||
                lower.find(".tiff") != string::npos ||
                lower.find(".tif") != string::npos) {
                files.push_back(string(directory) + "/" + filename);
            }
        }
    }
    closedir(dir);
    return files;
}

int main(int argc, char** argv) {
    const char* input_dir = "input_images";
    const char* output_dir = "output_images";
    const char* filter_type = "blur";
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_dir = argv[++i];
        } else if ((arg == "-f" || arg == "--filter") && i + 1 < argc) {
            filter_type = argv[++i];
            if (strcmp(filter_type, "blur") != 0 && strcmp(filter_type, "edge") != 0) {
                printf("Error: Invalid filter type '%s'\n", filter_type);
                printf("Valid options: blur, edge\n");
                return 1;
            }
        } else {
            printf("Error: Unknown option '%s'\n", arg.c_str());
            PrintUsage(argv[0]);
            return 1;
        }
    }
    
    printf("=== CUDA Image Processing ===\n\n");
    printf("Configuration:\n");
    printf("  Input directory:  %s\n", input_dir);
    printf("  Output directory: %s\n", output_dir);
    printf("  Filter type:      %s\n\n", filter_type);
    
    mkdir(output_dir, 0755);
    
    vector<string> image_files = GetImageFiles(input_dir);
    int num_images = image_files.size();
    
    if (num_images == 0) {
        printf("Error: No images found in '%s'\n", input_dir);
        printf("Supported formats: .jpg, .jpeg, .png, .tiff, .tif\n");
        return 1;
    }
    
    printf("Found %d images\n\n", num_images);
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    auto start = high_resolution_clock::now();
    
    int processed_count = 0;
    
    for (const auto& image_path : image_files) {
        int width, height, channels;
        unsigned char* image_data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
        
        if (!image_data) {
            continue;
        }
        
        unsigned char* output_data = (unsigned char*)malloc(width * height);
        processImageGPU(image_data, output_data, width, height, filter_type);
        
        string filename = image_path.substr(image_path.find_last_of("/") + 1);
        size_t last_dot = filename.find_last_of(".");
        if (last_dot != string::npos) {
            filename = filename.substr(0, last_dot) + ".jpg";
        }
        
        string output_path = string(output_dir) + "/" + filename;
        stbi_write_jpg(output_path.c_str(), width, height, 1, output_data, 95);
        
        free(image_data);
        free(output_data);
        
        processed_count++;
        if (processed_count % 25 == 0) {
            printf("Progress: %d/%d images\r", processed_count, num_images);
            fflush(stdout);
        }
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    printf("\n\n=== Results ===\n");
    printf("Total images processed: %d\n", processed_count);
    printf("Total time: %.2f seconds\n", duration.count() / 1000.0);
    printf("Average time: %.2f ms per image\n", (float)duration.count() / processed_count);
    printf("Throughput: %.2f images/second\n", processed_count / (duration.count() / 1000.0));
    
    return 0;
}
