#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>

#include "hip/hip_runtime.h"

// to compile on pascal node:
// HIP_PLATFORM=nvcc hipcc -o hipFillVector hipFillVector.cpp

#define FAKE_CUDA 1

__global__ void fillKernel(int N, int val, int *c_a){

#if FAKE_CUDA==1
  // find index of thread relative to thread-block
  int t = threadIdx.x;

  // find index of thread-block
  int b = blockIdx.x;

  // find number of threads in thread-block
  int B = blockDim.x;
#else
  // find index of thread relative to thread-block
  int t = hipThreadIdx_x;

  // find index of thread-block
  int b = hipBlockIdx_x;

  // find number of threads in thread-block
  int B = hipBlockDim_x;
#endif
  
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
  hipMalloc((void**) &c_a, N*sizeof(int));

  // 3. launch DEVICE fill kernel
  int T = 256; // number of threads per thread block 
  int val = 999; // value to fill DEVICE array with

#if FAKE_CUDA==1
  fillKernel <<< dim3((N+T-1)/T), dim3(T) >>> (N, val, c_a);
#else
  hipLaunchKernelGGL(fillKernel, dim3((N+T-1)/T), dim3(T), 0, 0, N, val, c_a);
#endif

  // 4. copy data from DEVICE array to HOST array
  hipMemcpy(h_a, c_a, N*sizeof(int), hipMemcpyDeviceToHost); 

  // 5. print out values on HOST
  for(int n=0;n<N;++n) printf("h_a[%d] = %d\n", n, h_a[n]);

  // 6. free arrays
  hipFree(c_a); free(h_a); return 0;
}
    
