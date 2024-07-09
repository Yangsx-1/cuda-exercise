#include<cstdio>

__global__ void hello() {
  printf ("block id:%d thread id:%d hello world!\n", blockIdx.x, threadIdx.x);
}

int main() {
  hello<<<4, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}