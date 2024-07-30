#include <cstdio>

constexpr int warp_size = 32;

__device__ __forceinline__ float warp_reduce(float val) {
  #pragma unroll
  for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, mask);
  }
  return val;
}

__global__ void dot(float* a, float* b, float* y, int n) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int offset = bid * blockDim.x + tid;
  float val = offset < n ? a[offset] * b[offset] : 0;
  int warpnum = (blockDim.x + warp_size - 1) / warp_size;
  int warp = tid / warp_size;
  int lane = tid % warp_size;
  __shared__ float smem[warpnum];
  if (lane == 0) smem[warp] = warp_reduce(val);
  __syncthreads();
  val = lane < warpnum ? smem[lane] : 0;
  if (warp == 0) val = warp_reduce(val);
  if (tid == 0) atomicAdd(y, val);
}

int main() {
  return 0;
}