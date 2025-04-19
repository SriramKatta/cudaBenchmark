#ifndef KERNELS_HPP
#define KERNELS_HPP
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
template <typename VT>
__global__
void stream_kernel(VT *data, size_t N) {
  const auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  const auto gridStride = blockDim.x * gridDim.x;
  for (auto idx = gridStart; idx < N; idx += gridStride)
    data[idx] += static_cast<VT>(1);
}


#endif