// LAB 2 SP24

#include <wb.h>

#define BLOCK_DIM 4

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  // Simple implementation, should be improved upon

  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int width = numAColumns;
  if(r < numCRows && c < numCColumns) {
      float sum = 0;
      for(int k = 0; k < width; k++) {
        float a = A[r * numAColumns + k];
        float b = B[k * numBColumns + c];
        sum += a * b;
      }
      C[r * numCColumns + c] = sum;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  // Sizes of each matrix
  size_t sizeA = numARows * numAColumns * sizeof(float);
  size_t sizeB = numBRows * numBColumns * sizeof(float);
  size_t sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeC);

  //@@ Allocate GPU memory here
  float *deviceInputA;
  float *deviceInputB;
  float *deviceOutputC;
  cudaMalloc(&deviceInputA, sizeA);
  cudaMalloc(&deviceInputB, sizeB);
  cudaMalloc(&deviceOutputC, sizeC);  

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInputA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInputB, hostB, sizeB, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 gridDim(ceil((1.0f*numCColumns)/BLOCK_DIM), ceil((1.0f*numCRows)/BLOCK_DIM), 1);
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<gridDim, blockDim>>>(deviceInputA, deviceInputB, deviceOutputC,
                                        numARows, numAColumns,
                                        numBRows, numBColumns,
                                        numCRows, numCColumns);
  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceOutputC, sizeC, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(deviceInputA);
  cudaFree(deviceInputB);
  cudaFree(deviceOutputC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

