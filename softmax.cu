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

template<const int threadnum=128>
__global__ void softmax(float* x, float* y, float* total, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * blockDim.x + tid;
  float exp_var = offset < n ? expf(x[offset]) : 0;
  float var = block_reduce(exp_var);
  if (tid == 0) atomicAdd(total, var);
  __threadfence();
  if (offset < n) y[offset] = exp_var / *total;
}