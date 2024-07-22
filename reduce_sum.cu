#include <cstdio>

constexpr int WARPSIZE = 32;

template<const int warp_size=WARPSIZE>
__device__ __forceinline__ float warp_reduce(float val) {
  #pragma unroll
  for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int threadnum=128>
__device__ __forceinline__ float block_reduce(float val) {
  const int warpnum = (threadnum + WARPSIZE - 1) / WARPSIZE;
  int warp = threadIdx.x / WARPSIZE;
  int lane = threadIdx.x % WARPSIZE;
  __shared__ float smem[warpnum];
  val = warp_reduce(val);
  if(lane == 0) smem[warp] = val;
  __syncthreads();

  val = lane < warpnum ? smem[lane] : 0;
  val = warp_reduce(val);
  return val;
}

__global__ void reduce(float* input, float* output, int size) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  float val = offset < size ? input[offset] : 0.0;
  val = block_reduce(val);
  if(threadIdx.x == 0) output[blockIdx.x] = val;
}

int main() {
  const int n = 16*1024*1024;
  float* h_ptr = (float*)malloc(sizeof(float)*n);
  for (int i = 0; i < n; ++i) {
    h_ptr[i] = 1.0;
  }
  int threadnum = 128;
  float* input;
  cudaMalloc(&input, sizeof(float)*n);
  cudaMemcpy(input, h_ptr, sizeof(float)*n, cudaMemcpyHostToDevice);

  for (int i = n; i > 0; i /= threadnum) {
    reduce<<<(i+threadnum-1)/threadnum, threadnum>>>(input, input, i);
  }

  float* res = (float*)malloc(sizeof(float));
  cudaMemcpy(res, &input[0], sizeof(float), cudaMemcpyDeviceToHost);
  printf("%f", *res);

  return 0;
}