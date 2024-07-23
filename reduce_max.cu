#include <cstdio>

constexpr int WARPSIZE = 32;

template<int warp_size=WARPSIZE>
__device__ __forceinline__ float warp_reduce(float val) {
  #pragma unroll
  for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
  }
  return val;
}

template<int threadnum=128>
__device__ __forceinline__ float block_reduce(float val) {
  const int warpnum = threadnum / WARPSIZE;
  const int warp = threadIdx.x / WARPSIZE;
  const int lane = threadIdx.x % WARPSIZE;
  __shared__ float smem[warpnum];
  val = warp_reduce(val);
  if (lane == 0) smem[warp] = val;
  __syncthreads();

  val = lane < warpnum ? smem[lane] : 0;
  val = warp_reduce(val);
  return val;
}

__global__ void reduce(float* input, float* output, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  float val = idx < size ? input[idx] : 0;
  val = block_reduce(val);
  if (threadIdx.x == 0) output[blockIdx.x] = val;
}

const int tmpsize = 9*1024*1024;

int main() {
  int size = 8*1024*1024;
  float* h_input = (float*)malloc(sizeof(float)*size);
  for (int i = 0; i < size; ++i) {
    h_input[i] = i;
  }
  h_input[100] = tmpsize+1;
  float* d_input;
  cudaMalloc(&d_input, size*sizeof(float));
  cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);
  const int threadnum = 128;
  while (size > 0) {
    reduce<<<(size+threadnum-1)/threadnum, threadnum>>>(d_input, d_input, size);
    size /= threadnum;
  }
  float* output = new float{0};
  cudaMemcpy(output, &d_input[0], sizeof(float), cudaMemcpyDeviceToHost);
  printf("%f", *output);
  return 0;
}