#include <cmath>
#include <cstdlib>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

__global__ void conv_forward_kernel(__half *output, const __half *input, const __half *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K, const int out_grid_width)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int nth_in = blockIdx.z;
    int map = blockIdx.y;
    int w0 = threadIdx.x;
    int h0 = threadIdx.y;
    int h_base = (blockIdx.x / out_grid_width) * BLOCK_SIZE;
    int w_base = (blockIdx.x % out_grid_width) * BLOCK_SIZE;
    int h = h_base + h0;
    int w = w_base + w0;
    __half sum = 0;

    for(int c = 0; c < Channel; c++) {
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                if(h < Height_out && w < Width_out) {
                    sum = __hadd(sum, __hmul(in_4d(nth_in, c, h + p, w + q), mask_4d(map, c, p , q)));
                }
            }
        }
    }
    if(h < Height_out && w < Width_out) {
        out_4d(nth_in, map, h, w) = (float) sum;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    *device_input_ptr = (float *) host_input;
    *device_mask_ptr = (float *) host_mask;
    *device_output_ptr = (float *) host_output;
    /*
    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    */
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void convert2Half(__half *output, const float *input, int nmem) {
    for(int i = 0; i < nmem; i++) {
        output[i] = __float2half(input[i]);
    }
}

__host__ void convert2Float(float *output, const __half *input, int nmem) {
    for(int i = 0; i < nmem; i++) {
        output[i] = __half2float(input[i]);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    __half *half_device_output;
    __half *half_device_input;
    __half *half_device_mask;

    size_t half_input_size = Batch * Channel * Width * Height * sizeof(__half);
    size_t half_output_size = Batch * Map_out * (Width - K + 1) * (Height - K + 1) * sizeof(__half);
    size_t half_mask_size = Map_out * Channel * K * K * sizeof(__half); 
    cudaMalloc(&half_device_input, half_input_size);
    cudaMalloc(&half_device_output, half_output_size);
    cudaMalloc(&half_device_mask, half_mask_size);

    __half *temp_input;
    __half *temp_mask;
    __half *temp_output;
    temp_input = (half *)malloc(half_input_size);
    temp_mask = (half *)malloc(half_mask_size);
    convert2Half(temp_input, device_input, Batch * Channel * Width * Height);
    convert2Half(temp_mask, device_mask, Map_out * Channel * K * K);

    cudaMemcpy(half_device_input, temp_input, half_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(half_device_mask, temp_mask, half_mask_size, cudaMemcpyHostToDevice);

    // Set the kernel dimensions and call the kernel
    int output_width = Width - K + 1;
    int output_height = Height - K + 1;
    int output_width_tiles = ceil(1.0f*output_width/BLOCK_SIZE);
    int output_height_tiles = ceil(1.0f*output_height/BLOCK_SIZE);

    // Using lecture slide implementation
    dim3 grid_dim(output_width_tiles * output_height_tiles, Map_out, Batch);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

    conv_forward_kernel<<<grid_dim, block_dim>>>(half_device_output, half_device_input, half_device_mask, Batch, Map_out, Channel, Height, Width, K, output_width_tiles);

    cudaDeviceSynchronize();

    temp_output = (__half *)malloc(half_output_size);
    cudaMemcpy(temp_output, half_device_output, half_output_size, cudaMemcpyDeviceToHost);
    
    convert2Float(device_output, temp_output, Batch * Map_out * (Width - K + 1) * (Height - K + 1));

    cudaFree(half_device_input);
    cudaFree(half_device_mask);
    free(temp_input);
    free(temp_mask);
    free(temp_output);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Free device memory
    free(device_input);
    cudaFree(device_output);
    free(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
