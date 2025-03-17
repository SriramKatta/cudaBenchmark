#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#pragma once

#include <stddef.h>
#include <cuda_errror_handler.cuh>
#include <cuda.h>
namespace CH
{
  template <typename VT>
  inline VT *allocHost(size_t N)
  {
    VT *ptr;
    CHECK_CUDA_ERR(cuadMallocHost(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline VT *allocDevice(size_t N)
  {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMalloc(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline VT *allocManaged(size_t N)
  {
    VT *ptr;
    CHECK_CUDA_ERR(cuadMallocManaged(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline void memcpyH2D(const VT *host, const VT *dev, size_t N, cudaStream_t stream = 0)
  {
    CHECK_CUDA_ERR(cudaMemcpyAsync(dev, host, N * sizeof(VT), cudaMemcpyHostToDevice, stream));
  }

  template <typename VT>
  inline void memcpyD2H(const VT *dev, const VT *host, size_t N, cudaStream_t stream = 0)
  {
    CHECK_CUDA_ERR(cudaMemcpyAsync(host, dev, N * sizeof(VT), cudaMemcpyDeviceToHost, stream));
  }

  unsigned int getWarpSize(int devID = 0)
  {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    int ws;
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, devID));
    return ws;
  }

  
  unsigned int getSMCount(int devID = 0)
  {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    int SMcount;
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&SMcount, cudaDevAttrMultiProcessorCount, devID));
    return SMcount;
  }


}

#endif