// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024

__global__ void grayscale_conversion(float *input, char *buf, char * gray_buf, int width, int height, int channels) {
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
    }
}
//@@ insert code here
__global__ void histogram_kernel(char *gray_buf, int width, int height, int *histogram){
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_loc = y * width + x;
    if(x < width && y < height) {
        atomicInc((unsigned int *)(histogram + gray_buf[pixel_loc]), 1);
    }
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

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


  //@@ insert code here

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}

