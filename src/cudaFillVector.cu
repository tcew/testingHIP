#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// to compile on pascal node:
// nvcc -arch=sm_60 -o cudaFillVector cudaFillVector.cu


__global__ void fillKernel(int N, int val, int *c_a){

  // find index of thread relative to thread-block
  int t = threadIdx.x;

  // find index of thread-block
  int b = blockIdx.x;

  // find number of threads in thread-block
  int B = blockDim.x;

  // construct map from thread and thread-block indices into linear array index
  int n = t + b*B;
  
  // check index is in range
  if(n<N)
    c_a[n] = val; // work done by thread
}


int main(int argc, char **argv){

  // 1. allocate HOST array
  int N = 1024;
  int *h_a = (int*) calloc(N, sizeof(int));

  // 2. allocate DEVICE array
  int *c_a;
  cudaMalloc(&c_a, N*sizeof(int));

  // 3. launch DEVICE fill kernel
  int T = 256; // number of threads per thread block 
  int val = 999; // value to fill DEVICE array with

  dim3 G( (N+T-1)/T ); // number of thread blocks to use
  dim3 B(T);
  
  fillKernel <<< G,B >>> (N, val, c_a);

  // 4. copy data from DEVICE array to HOST array
  cudaMemcpy(h_a, c_a, N*sizeof(int), cudaMemcpyDeviceToHost); 

  // 5. print out values on HOST
  for(int n=0;n<N;++n) printf("h_a[%d] = %d\n", n, h_a[n]);

  // 6. free arrays
  cudaFree(c_a); free(h_a); return 0;
}
    
