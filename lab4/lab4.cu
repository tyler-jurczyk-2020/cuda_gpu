#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define KERNEL_RADIUS 1
#define KERNEL_DIM 3

//@@ Define constant memory for device kernel here
__constant__ float deviceMask[KERNEL_DIM][KERNEL_DIM][KERNEL_DIM]; // Constant memory

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
   __shared__ float T[TILE_WIDTH + 2*KERNEL_RADIUS][TILE_WIDTH + 2*KERNEL_RADIUS][TILE_WIDTH + 2*KERNEL_RADIUS];
   int row_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
   int col_o = blockIdx.x * TILE_WIDTH + threadIdx.x;
   int depth_o = blockIdx.z * TILE_WIDTH + threadIdx.z;
   int row_i = row_o - KERNEL_RADIUS;
   int col_i = col_o - KERNEL_RADIUS;
   int depth_i = depth_o - KERNEL_RADIUS;

   int input_loc = (depth_i * x_size * y_size) + (row_i * x_size) + col_i;
   int output_loc = (depth_o * x_size * y_size) + (row_o * x_size) + col_o;

   if(row_i >= 0 && row_i < y_size && 
   col_i >= 0 && col_i < x_size &&
   depth_i >= 0 && depth_i < z_size) {
       T[threadIdx.z][threadIdx.y][threadIdx.x] = input[input_loc];
   }
   else {
       T[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
   
   float pcell = 0;

    if(threadIdx.y < TILE_WIDTH &&
       threadIdx.x < TILE_WIDTH &&
       threadIdx.z < TILE_WIDTH) {
       for(int k = 0; k < KERNEL_DIM; k++) {
           for(int i = 0; i < KERNEL_DIM; i++) {
                for(int j = 0; j < KERNEL_DIM; j++) {
                   float kernelMask = T[threadIdx.z + k][threadIdx.y + i][threadIdx.x + j];
                   pcell += deviceMask[k][i][j] * kernelMask;
               }
           }
       }  
        if(row_o < y_size &&
           col_o < x_size &&
           depth_o < z_size) {
            output[output_loc] = pcell;
        }
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  size_t size_3d = (inputLength - 3) * sizeof(float);
  size_t kernelSize = kernelLength * sizeof(float);

  float *deviceInput;
  float *deviceOutput;

  cudaMalloc(&deviceInput, size_3d);
  cudaMalloc(&deviceOutput, size_3d);

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, size_3d, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceMask, hostKernel, kernelSize);

  //@@ Initialize grid and block dimensions here
  dim3 gridDim(ceil(x_size/(1.0f * TILE_WIDTH)), ceil(y_size/(1.0f * TILE_WIDTH)), ceil(z_size/(1.0f * TILE_WIDTH)));
  dim3 blockDim(TILE_WIDTH + 2*KERNEL_RADIUS, TILE_WIDTH + 2*KERNEL_RADIUS, TILE_WIDTH + 2*KERNEL_RADIUS);
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, size_3d, cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);
  
  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

