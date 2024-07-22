#include <cstdio>
#include <iostream>

#define CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << " (" << cudaGetErrorString(err) << ")\n";               \
            exit(err);                                                           \
        }                                                                        \
    }

const int N = 256;
const int tilesize = 16;

__global__ void transpose(int* input, int* output) {
  __shared__ int smem[tilesize][tilesize+1];
  int bx = blockIdx.x * tilesize;
  int by = blockIdx.y * tilesize;
  int nx = bx + threadIdx.x;
  int ny = by + threadIdx.y;
  if (nx < N && ny < N) {
    smem[threadIdx.y][threadIdx.x] = input[ny * N + nx];
  }
  __syncthreads();

  nx = by + threadIdx.x;
  ny = bx + threadIdx.y;
  if (nx < N && ny < N) {
    output[ny * N + nx] = smem[threadIdx.x][threadIdx.y];
  }
}

int main() {
  const int matrix_size = N * N;
  int *h_ptr = (int*)malloc(sizeof(int) * matrix_size);
  int *d_ptr, *d_out;
  CHECK(cudaMalloc(&d_ptr, matrix_size * sizeof(int)));
  CHECK(cudaMalloc(&d_out, matrix_size * sizeof(int)));
  for (int i = 0; i < matrix_size; ++i) {
    h_ptr[i] = i;
  }
  CHECK(cudaMemcpy(d_ptr, h_ptr, matrix_size*sizeof(int), cudaMemcpyHostToDevice));

  dim3 blocksize(tilesize, tilesize);
  int numblock = (N + tilesize - 1) / tilesize;
  dim3 numblocks(numblock, numblock);
  transpose<<<numblocks, blocksize>>>(d_ptr, d_out);
  CHECK(cudaMemcpy(h_ptr, d_out, matrix_size * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < 16; ++i) {
    printf("%d\t", h_ptr[i]);
  }

  return 0;
}