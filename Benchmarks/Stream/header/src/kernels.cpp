#include "kernels.hpp"

template <typename VT>
__global__ void testKernels::Stream(VT *data, size_t N) {
  const auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  const auto gridStride = blockDim.x * gridDim.x;
  for (size_t idx = gridStart; idx < N; idx += gridStride) {
    data[idx] += static_cast<VT>(1);
  }
}