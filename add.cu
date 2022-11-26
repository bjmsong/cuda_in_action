#include <iostream>
#include <math.h>

using namespace std;

__global__
void add(int *a, int *b, int *c)
{
    // 这边的*是解引用操作符
    *c = *a + *b;
}


int main(void){
    int a = 1, b = 2, c;
    int *d_a, *d_b, *d_c;  // device copies of a,b,c

    // Allocate space for device copies of a,b,c
    cudaMalloc((void **)&d_a, sizeof(int));
    cudaMalloc((void **)&d_b, sizeof(int));
    cudaMalloc((void **)&d_c, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    cout << a << " " << b << " " << c << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}