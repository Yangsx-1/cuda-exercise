#include <iostream>

#define BLOCK_SIZE 64

__global__ void add (int n, float* x, float* y, float* result) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    result[id] = x[id] + y[id];
  }
}


int main() {
  int n = 10000;
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float* x, * y, * result;
  x = new float[n];
  y = new float[n];
  result = new float[n];
  for (int i = 0; i < n; ++i) {
    x[i] = i;
    y[i] = i + 1;
  }
  float *dx, *dy, *dr;
  size_t size = n * sizeof(float);
  cudaMalloc(&dx, size);
  cudaMalloc(&dy, size);
  cudaMalloc(&dr, size);
  cudaMemcpy(dx, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, y, size, cudaMemcpyHostToDevice);
  add<<<numBlocks, BLOCK_SIZE>>>(n, dx, dy, dr);
  std::cout << "after kernal" << std::endl;
  cudaDeviceSynchronize();
  cudaMemcpy(result, dr, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; ++i) {
    std::cout << result[i] << std::endl;
  }
  delete[] x;
  delete[] y;
  delete[] result;
  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dr);

  return 0;
}