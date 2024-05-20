#include <iostream>

__global__ void vectordot(int count, int* a, int* b, int *c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < count) {
    c[idx] = a[idx] * b[idx];
    idx += gridDim.x * blockDim.x;
  }
}

int main() {
  const int arr_len = 100000;
  const int block_dim = 32;
  const int grid_dim = 16;
  int* a = new int[arr_len];
  int* b = new int[arr_len];
  for (int i = 0; i < arr_len; ++i) {
    a[i] = i;
    b[i] = 10-i;
  }
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, arr_len*sizeof(int));
  cudaMalloc(&d_b, arr_len*sizeof(int));
  cudaMalloc(&d_c, arr_len*sizeof(int));
  cudaMemcpy(d_a, a, arr_len*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, arr_len*sizeof(int), cudaMemcpyHostToDevice);
  std::cout << "before kernal" << std::endl;
  vectordot<<<grid_dim, block_dim>>>(arr_len, d_a, d_b, d_c);
  std::cout << "after kernal" << std::endl;
  cudaDeviceSynchronize();
  int* c = new int[arr_len];
  cudaMemcpy(c, d_c, arr_len*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 20; ++i) {
    std::cout << c[i] << std::endl;
  }
  delete[] a;
  delete[] b;
  delete[] c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}