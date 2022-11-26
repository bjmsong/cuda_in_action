#include <cstddef>
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
__global__ void add(int n, float *A, float *B, float *C)
{
  /*
  threadIdx.x: index of the current thread within its block
  blockIdx.x: index of the current thread block in the grid
  blockDim.x: number of threads in the block
  gridDim.x: number of blocks in the grid
  */ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

int main(void)
{
    int N = 1<<24;
    size_t size = N*sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // initialize vecotrs on the host
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    double t1 = tick();
    // Run kernel on 10M elements on the GPU
    int blockSize = 256;
    // 保证至少有N个threads
    int numBlocks = (N + blockSize - 1) / blockSize;
    // int numBlocks = 10;
    add<<<numBlocks, blockSize>>>(N, d_A, d_B, d_C);
    
    // 为了统计时间，需要先做同步
    cudaDeviceSynchronize();
    double t2 = tick();
    cout << "Completed in: " << t2 - t1 << " seconds" << endl;

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}