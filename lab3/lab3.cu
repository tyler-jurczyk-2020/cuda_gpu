#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
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
  dim3 gridDim(ceil((1.0f*numCColumns)/TILE_WIDTH), ceil((1.0f*numCRows)/TILE_WIDTH), 1);
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<gridDim, blockDim>>>(deviceInputA, deviceInputB, deviceOutputC,
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

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
