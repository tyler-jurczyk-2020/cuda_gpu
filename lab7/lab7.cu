// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024

__global__ void histogram_grayscale_conversion(float *input, char *buf, char *gray_buf, int *histogram, int width, int height, int channels) {
    int x = channels * (blockIdx.x * blockDim.x + threadIdx.x);
    int x_raw = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_loc = (y * width) + x;
    int pixel_loc_raw = (y * width) + x_raw;
    if(x < 3*width && y < height) {
        for(int i = 0; i < channels; i++) {
            buf[pixel_loc + i] = (unsigned char)(255 * input[pixel_loc + i]);
        }
        __syncthreads();

        int r = buf[pixel_loc];
        int g = buf[pixel_loc + 1];
        int b = buf[pixel_loc + 2];
        gray_buf[pixel_loc_raw] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
        __syncthreads();

        atomicInc((unsigned int *)(histogram + gray_buf[pixel_loc_raw]), 1);
    }
}

__global__ void equalization(char *buf, int width, int height, int channels, float *cdf, float *output) {
    int x = channels * (blockIdx.x * blockDim.x + threadIdx.x);
    int y = (blockIdx.y * blockDim.y + threadIdx.y);
    int pixel_loc = (y * width) + x;
    if(x < 3*width && y < height) {
        for(int i = 0; i < channels; i++) {
            unsigned char clamp_val = 255*(cdf[buf[pixel_loc + i]] - cdf[0])/(1.0 - cdf[0]);
            float max_val = clamp_val > 0 ? clamp_val : 0;
            output[pixel_loc + i] = (max_val < 255.0 ? max_val : 255.0)/255.0;
        }
    }
}

__global__ void histogram_scan(int *input, float *output, int len) {
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
  char *buf;
  char *gray_buf;
  int  *histogram;
  float *cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  size_t image_size = imageWidth * imageHeight * imageChannels * sizeof(float);
  size_t image_size_no_channel = imageWidth * imageHeight * sizeof(float);
  size_t histogram_size = HISTOGRAM_LENGTH * sizeof(int);
  size_t cdf_size = HISTOGRAM_LENGTH * sizeof(int);
  hostInputImageData = (float *)wbImage_getData(inputImage);
  hostOutputImageData = (float *)malloc(image_size);

  //@@ insert code here
  cudaMalloc(&deviceInput, image_size);
  cudaMalloc(&deviceOutput, image_size);
  cudaMalloc(&buf, image_size);
  cudaMalloc(&gray_buf, image_size_no_channel);
  cudaMalloc(&histogram, histogram_size);
  cudaMalloc(&cdf, cdf_size);

  cudaMemcpy(deviceInput, hostInputImageData, image_size, cudaMemcpyHostToDevice);

  int num_blocks_x = ceil((1.0f*imageWidth)/(3 * BLOCK_SIZE));
  int num_blocks_y = ceil((1.0f*imageHeight)/BLOCK_SIZE);

  dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Args: float *input, char *buf, char *gray_buf, int *histogram, int width, int height, int channels
  histogram_grayscale_conversion<<<grid_dim, block_dim>>>(deviceInput, buf, gray_buf, histogram, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  // Args: float *input, float *output, int len
  histogram_scan<<<1, HISTOGRAM_LENGTH>>>(histogram, cdf, HISTOGRAM_LENGTH);
  cudaDeviceSynchronize();

  // Args: char *buf, int width, int height, int channels, float *cdf, float *output
  equalization<<<grid_dim, block_dim>>>(buf, imageWidth, imageHeight, imageChannels, cdf, deviceOutput);
  cudaDeviceSynchronize();

  wbSolution(args, outputImage);

  //@@ insert code here
  free(hostOutputImageData);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(buf);
  cudaFree(gray_buf);
  cudaFree(histogram);
  cudaFree(cdf);

  return 0;
}

