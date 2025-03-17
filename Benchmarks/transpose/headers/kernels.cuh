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

template <typename VT>
__global__ void copyShared(VT *__restrict__ i_data, VT *__restrict__ o_data, size_t N)
{
  auto tix = threadIdx.x;
  auto gridStart = tix + blockDim.x * blockIdx.x;
  auto gridStride = blockDim.x * gridDim.x;
  extern __shared__ VT smem[];
  for (size_t i = gridStart; i < N; i += gridStride)
  {
    smem[tix] =  i_data[i];
  }
  __syncthreads();
  for (size_t i = gridStart; i < N; i += gridStride)
  {
    o_data[i] = smem[tix];
  }
}



#endif