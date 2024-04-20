#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define BLOCK_SIZE 32
#define TILE_WIDTH 32

// OP_2 --> Base Input Matrix Unrolling & Tile Matrix Multiplication 

__global__ void conv_forward_kernel_unroll(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    // Unrolling constants
    const int Unroll_Width = Height_out * Width_out;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    const int Mask_size = K * K;

    // Access input as normal
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // Special acccess for unrolled matrix
    // (Batch, Row, Column)
    #define in_unrolled_3d(i2, i1, i0) output[(i2) * (Channel * Mask_size * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]

    // Insert your GPU convolution kernel code here
    // Shared mem
    /*
    int shared_tile_dim = BLOCK_SIZE + K - 1;
    int shared_tile_area = shared_tile_dim * shared_tile_dim;
    extern __shared__ float shared_mem[];
    float *shared_tile = shared_mem;
    float *shared_mask = shared_mem + shared_tile_area;
    */
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int channel, channel_thread, h_out, w_out, h_unroll, h_base, w_unroll;
    int nth_in = blockIdx.z;

    if(thread < Channel * Unroll_Width) {
        channel = thread / Unroll_Width; 
        channel_thread = thread % Unroll_Width;
        h_out = channel_thread / Width_out;
        w_out  = channel_thread % Width_out;
        w_unroll = h_out * Width_out + w_out;
        h_base = channel * K * K; 
        for(int p = 0; p < K; p++) {
            for(int q = 0; q < K; q++) {
                h_unroll = h_base + p * K + q;
                in_unrolled_3d(nth_in, h_unroll, w_unroll) = in_4d(nth_in, channel, h_out + p, w_out + q); 
            }
        }
    }

    #undef in_4d
    #undef in_unrolled_3d
}

// Compute C = A * B
__device__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subtileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subtileB[TILE_WIDTH][TILE_WIDTH];

  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float val = 0;
  int width = numAColumns; // = numBRows

  for(int i = 0; i < (width - 1)/TILE_WIDTH + 1; i++) {
    if(row < numARows && i * TILE_WIDTH + threadIdx.x < width) {
      subtileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * TILE_WIDTH + threadIdx.x];
    }
    else {
      subtileA[threadIdx.y][threadIdx.x] = 0;  
    }
    if(i * TILE_WIDTH + threadIdx.y < width && col < numBColumns) {
      subtileB[threadIdx.y][threadIdx.x] = B[(i * TILE_WIDTH + threadIdx.y) * numBColumns + col];
    }
    else {
      subtileB[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    for(int j = 0; j < TILE_WIDTH; j++) {
        val += subtileA[threadIdx.y][j] * subtileB[j][threadIdx.x];
    }
    __syncthreads();
  }
  if(row < numARows and col < numBColumns) {
    C[row * numBColumns + col] = val;
  }
}


// Compute C = A * B
__global__ void conv_forward_kernel_matmul(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  const int Mask_size = K * K;

  #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
  #define in_unrolled_3d(i2, i1, i0) output[(i2) * (Channel * Mask_size * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
  #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;

  // Unrolling constants
  const int Unroll_Width = Height_out * Width_out * K * K; // Also considering mask in width

  int thread = blockIdx.x * blockDim.x + threadIdx.x;
  int nth_in = blockIdx.z;
  int map = blockIdx.y;
  int channel = thread / Unroll_Width; 
  // Tiled access to unrolled input

  float *maskMat = (float *) &mask_4d(map, channel, 0, 0);
  float *inputMat = (float *) &in_unrolled_3d(nth_in, 0, 0);
  float *outputMat = (float *) &out_4d(nth_in, map, 0, 0);

  int numARows = Map_out;
  int numAColumns = Channel * K * K;
  int numBRows = Channel * K * K;
  int numBColumns = Height_out * Width_out;
  int numCRows = Map_out;
  int numCColumns = Height_out * Width_out;

  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  matrixMultiplyShared(maskMat, inputMat, outputMat, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  #undef in_4d
  #undef in_unrolled_3d
  #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    size_t input_size = Batch * Channel * Width * Height * sizeof(float);
    size_t output_size = Batch * Map_out * (Width - K + 1) * (Height - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float); 
    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);
    cudaMalloc(device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int output_width = Width - K + 1;
    int output_height = Height - K + 1;
    int num_blocks_unroll = ceil((Channel * output_width * output_height)/(BLOCK_SIZE*BLOCK_SIZE));
    int num_blocks_matmul = ceil((Channel * K * K * output_width * output_height)/(BLOCK_SIZE*BLOCK_SIZE));
    int shared_tile_mem = (BLOCK_SIZE + K - 1) * (BLOCK_SIZE + K - 1);
    int shared_mask_mem = K * K;

    // Unrolled Input
    float *device_unrolled_input;
    size_t unroll_input_size = Channel * output_width * output_height * sizeof(float);
    cudaMalloc(&device_unrolled_input, unroll_input_size);

    // Using lecture slide implementation
    dim3 unroll_grid_dim(num_blocks_unroll, Map_out, Batch);
    dim3 matmul_grid_dim(num_blocks_matmul, Map_out, Batch);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);
    size_t shared_mem = (shared_tile_mem + shared_mask_mem) * sizeof(float);

    // Launch unrolling kernel
    conv_forward_kernel_unroll<<<unroll_grid_dim, block_dim>>>(device_unrolled_input, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();

    // Launch matrix multiplication kernel
    conv_forward_kernel_matmul<<<matmul_grid_dim, block_dim, shared_mem>>>(device_output, device_unrolled_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();

    cudaFree(device_unrolled_input);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    size_t output_size = Batch * Map_out * (Width - K + 1) * (Height - K + 1) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
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
