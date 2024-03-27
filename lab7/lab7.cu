// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128
#define BLOCK_DIM 32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void histogram_grayscale_conversion(float *input, unsigned char *buf, unsigned char *gray_buf, int *histogram, int width, int height, int channels) {
    int x = channels * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x_raw = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_loc = (y * width * channels) + x;
    int pixel_loc_raw = (y * width) + x_raw;
    if(x_raw < width && y < height) {
        for(int i = 0; i < channels; i++) {
            buf[pixel_loc + i] = (unsigned char)(255 * input[pixel_loc + i]);
        }
    }
    __syncthreads();

    if(x_raw < width && y < height) {
        unsigned char r = buf[pixel_loc];
        unsigned char g = buf[pixel_loc + 1];
        unsigned char b = buf[pixel_loc + 2];
        gray_buf[pixel_loc_raw] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
    __syncthreads();

    if(x_raw < width && y < height) {
        atomicAdd((unsigned int *)(histogram + gray_buf[pixel_loc_raw]), 1);
    }
}

__global__ void equalization(unsigned char *buf, int width, int height, int channels, float *cdf, float *output) {
    int x = channels * (blockIdx.x * blockDim.x + threadIdx.x);
    int x_raw = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_loc = (y * width * channels) + x;
    if(x_raw < width && y < height) {
        for(int i = 0; i < channels; i++) {
            unsigned char clamp_val = 255*(cdf[buf[pixel_loc + i]] - cdf[0])/(1.0 - cdf[0]);
            unsigned char max_val = clamp_val > 0 ? clamp_val : 0;
            output[pixel_loc + i] = (max_val < 255 ? max_val : 255)/255.0;
        }
    }
}

__global__ void histogram_scan(int *input, float *output, int len, int width, int height) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int input_loc = blockIdx.x * 2*blockDim.x + threadIdx.x;
  int output_loc = input_loc;
  int T_loc = threadIdx.x;
  int size = width * height;

  // Load Shared Mem
  if(input_loc < len)
      T[T_loc] = input[input_loc]/(1.0f*size);
  else
      T[T_loc] = 0;
  if(input_loc + blockDim.x < len)
      T[T_loc + blockDim.x] = input[input_loc + blockDim.x]/(1.0f*size);
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
}

void check_host_float_array(float *array, int amt) {
    for(int i = 0; i < amt; i++) {
        wbLog(TRACE, array[i]);
    }
}

int check_device_float_array(float *array, int amt) {
    float *local = (float *) malloc(amt * sizeof(float));
    wbCheck(cudaMemcpy(local, array, amt * sizeof(float), cudaMemcpyDeviceToHost));
    for(int i = 0; i < amt; i++) {
        wbLog(TRACE, local[i]);
    }
    free(local);
    return 0;
}

int check_device_char_array(unsigned char *array, int amt) {
    unsigned char *local = (unsigned char *) malloc(amt * sizeof(unsigned char));
    wbCheck(cudaMemcpy(local, array, amt * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    for(int i = 0; i < amt; i++) {
        wbLog(TRACE, (int) local[i]);
    }
    free(local);
    return 0;
}

int check_device_int_array(int *array, int amt) {
    int *local = (int *) malloc(amt * sizeof(int));
    wbCheck(cudaMemcpy(local, array, amt * sizeof(int), cudaMemcpyDeviceToHost));
    for(int i = 0; i < amt; i++) {
        wbLog(TRACE, local[i]);
    }
    free(local);
    return 0;
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInput;
  float *deviceOutput;
  unsigned char *buffer;
  unsigned char *gray_buf;
  int *histogram;
  float *cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  int image_size = imageWidth * imageHeight * imageChannels;
  int image_size_no_channel = imageWidth * imageHeight;
  hostInputImageData = (float *)wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(image_size * sizeof(float));
  wbImage_setData(outputImage, hostOutputImageData);

  //@@ insert code here
  wbCheck(cudaMalloc(&deviceInput, image_size * sizeof(float)));
  wbCheck(cudaMalloc(&deviceOutput, image_size * sizeof(float)));
  wbCheck(cudaMalloc(&buffer, image_size * sizeof(unsigned char)));
  wbCheck(cudaMalloc(&gray_buf, image_size_no_channel * sizeof(unsigned char) ));
  wbCheck(cudaMalloc(&histogram, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMalloc(&cdf, HISTOGRAM_LENGTH * sizeof(float)));

  wbCheck(cudaMemcpy(deviceInput, hostInputImageData, image_size * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(int)));

  int num_blocks_x = ceil((1.0f*imageWidth)/BLOCK_DIM);
  int num_blocks_y = ceil((1.0f*imageHeight)/BLOCK_DIM);

  dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);

  // Args: float *input, char *buf, char *gray_buf, int *histogram, int width, int height, int channels
  histogram_grayscale_conversion<<<grid_dim, block_dim>>>(deviceInput, buffer, gray_buf, histogram, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  // Check first kernel results
  // check_device_float_array(deviceInput, 10);
  // check_device_char_array(buffer, image_size);
  // check_device_char_array(gray_buf, image_size_no_channel);
  // check_device_int_array(histogram, HISTOGRAM_LENGTH);

  // Args: float *input, float *output, int len
  histogram_scan<<<1, BLOCK_SIZE>>>(histogram, cdf, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  // Check second kernel results
  // check_device_float_array(cdf, HISTOGRAM_LENGTH);
  
  // Args: char *buf, int width, int height, int channels, float *cdf, float *output
  equalization<<<grid_dim, block_dim>>>(buffer, imageWidth, imageHeight, imageChannels, cdf, deviceOutput);
  cudaDeviceSynchronize();
  // Check final kernel results
  // check_device_float_array(deviceOutput, image_size);
  
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutput, image_size * sizeof(float), cudaMemcpyDeviceToHost));

  // Testing
  // const char *solutionOutput = wbArg_getExpectedOutputFile(args);
  // hostOutputImageData = wbImage_getData(wbImport(solutionOutput)); 
  // check_host_float_array(hostOutputImageData, image_size);

  wbSolution(args, outputImage);
  //@@ insert code here

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(buffer);
  cudaFree(gray_buf);
  cudaFree(histogram);
  cudaFree(cdf);

  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

