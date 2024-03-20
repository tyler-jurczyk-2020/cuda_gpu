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

__global__ void post_scan(float *output, float *add_amt, int len) {
    int input_loc = blockIdx.x * 2*blockDim.x + threadIdx.x;
    int output_loc = input_loc;
    __syncthreads(); // Make sure add_amt values are correct
    if(input_loc + 2*blockDim.x < len)
        output[output_loc + 2*blockDim.x] += add_amt[blockIdx.x];
    if(input_loc + 3*blockDim.x < len)
        output[output_loc + 3*blockDim.x] += add_amt[blockIdx.x];
}

__global__ void scan(float *input, float *output, float *add_amt, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int input_loc = blockIdx.x * 2*blockDim.x + threadIdx.x;
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

  __syncthreads(); // Need to make sure all results of last iteration are available before writeback

  // Writeback To Global Mem
  if(output_loc < len)
      output[output_loc] = T[T_loc];
  if(output_loc + blockDim.x < len)
      output[output_loc + blockDim.x] = T[T_loc + blockDim.x];
  // Also need to write the add amount to the global memory
  int end_of_block = blockIdx.x * 2*blockDim.x + 2*blockDim.x;
  if(threadIdx.x == 0 && end_of_block < len)
      add_amt[blockIdx.x] = T[2*BLOCK_SIZE - 1];

  if(blockIdx.x == 0) {
      __syncthreads(); // Sync before performing aux scan

      // Reduction Step For AUX
      for(int stride = 1; stride < 2*BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < 2*BLOCK_SIZE && (index-stride) >= 0) {
          add_amt[index] += add_amt[index-stride];
        }
      }

      // Post Scan Step For AUX
      for(int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < 2*BLOCK_SIZE) {
          add_amt[index + stride] += add_amt[index];
        }
      }
  }
}

void check_output(int numElements, float *deviceOutput) {
  float *check_dev_out = (float *)malloc(numElements * sizeof(float));
  cudaMemcpy(check_dev_out, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < numElements; i++)
    wbLog(TRACE, check_dev_out[i]);
  free(check_dev_out);
}

void check_amount(int numBlocks, float *add_amt) {
  float *check_dev_out = (float *)malloc(numBlocks * sizeof(float));
  cudaMemcpy(check_dev_out, add_amt, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < numBlocks; i++)
    wbLog(TRACE, check_dev_out[i]);
  free(check_dev_out);
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

  cudaDeviceSynchronize();

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  post_scan<<<gridDim, blockDim>>>(deviceOutput, add_amt, numElements);

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

