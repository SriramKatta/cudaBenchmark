#ifndef KERNEL_CUH
#define KERNEL_CUH

#pragma once

template <typename VT>
__global__ void fillWith_n(VT *__restrict__ i_data, size_t N, VT fillval)
{
  auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  auto gridStride = blockDim.x * gridDim.x;
  for (size_t i = gridStart; i < N; i += gridStride)
  {
    i_data[i] = fillval;
  }
}


template <typename VT>
__global__ void copy(VT *__restrict__ i_data, VT *__restrict__ o_data, size_t N)
{
  auto gridStart = threadIdx.x + blockDim.x * blockIdx.x;
  auto gridStride = blockDim.x * gridDim.x;
  for (size_t i = gridStart; i < N; i += gridStride)
  {
    o_data[i] = i_data[i];
  }
}

#endif