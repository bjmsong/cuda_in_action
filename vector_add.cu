#include <iostream>
#include <math.h>
#include <sys/time.h>

using namespace std;

double tick(){
    timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec + time.tv_usec*1E-6;
}

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  /*
  threadIdx.x: index of the current thread within its block
  blockIdx.x: index of the current thread block in the grid
  blockDim.x: number of threads in the block
  gridDim.x: number of blocks in the grid
  */ 
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+=stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<24;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  double t1 = tick();
  // Run kernel on 10M elements on the GPU
  int blockSize = 256;
  // 保证至少有N个threads
  int numBlocks = (N + blockSize - 1) / blockSize;
  // int numBlocks = 10;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  double t2 = tick();
  cout << "Completed in: " << t2 - t1 << " seconds" << endl;
  
  // Check for errors (all values should be 3.0f)
  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = fmax(maxError, fabs(y[i]-3.0f));
  // std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}