#ifndef KERNEL_CUH
#define KERNEL_CUH

#pragma once

template <typename VT>
__global__ void kernel(size_t N, VT *a) {
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t i = start; i <= N; i += stride) {
    a[i] += 1.0;
  }
}


#endif