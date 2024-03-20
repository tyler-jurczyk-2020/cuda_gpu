// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void post_scan(float *input, float *output, float *add_amt, int len) {
    int input_loc = blockIdx.x * blockDim.x + threadIdx.x;
    int output_loc = input_loc;
    if(input_loc < len)
        output[output_loc] = input[input_loc] + add_amt;
    if(input_loc + blockDim.x < len)
        output[output_loc + blockDim.x] = input[input_loc + blockDim.x] + add_amt;
}

__global__ void scan(float *input, float *output, float *add_amt, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  float T[2*BLOCK_SIZE];
  int input_loc = blockIdx.x * blockDim.x + threadIdx.x;
  int output_loc = input_loc;
  int T_loc = threadIdx.x;

  // Load Shared Mem
  if(input_loc < len)
      T[T_loc] = input[input_loc];
  else
      T[T_loc] = 0;
  if(input_loc + blockDim.x < len)
      T[T_loc + blockDim.x] = input[input_loc + blockDim.x];
  else
      T[T_loc + blockDim.x] = 0;

  // Reduction Step
  for(int stride = 1; stride < 2*BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
  }

  // Post Scan Step
  for(int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index + stride < 2*BLOCK_SIZE) {
      T[index + stride] += T[index];
    }
  } 

  // Writeback To Global Mem
  if(output_loc < len)
      output[output_loc] = T[T_loc];
  if(output_loc + blockDim.x < len)
      output[input_loc + blockDim.x] = T[T_loc + blockDim.x];
  // Also need to write the add amount to the global memory
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *add_amt;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil(1.0f*numElements/(2*BLOCK_SIZE));

  dim3 gridDim(numBlocks, 1, 1);
  dim3 blockDim(BLOCK_SIZE, 1, 1);

  cudaMalloc(&add_amt, numBlocks * sizeof(float));

  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, add_amt, numElements);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  post_scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, add_amt, numElements);

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(add_amt);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

