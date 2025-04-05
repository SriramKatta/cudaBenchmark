#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH
#pragma once

#include <cuda.h>
#include <stddef.h>

#include "cuda_errror_handler.cuh"

namespace cuda_helpers {
  template <typename VT>
  inline VT *allocHost(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMallocHost(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline VT *allocDevice(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cudaMalloc(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline VT *allocManaged(size_t N) {
    VT *ptr;
    CHECK_CUDA_ERR(cuadMallocManaged(&ptr, N * sizeof(VT)));
    return ptr;
  }

  template <typename VT>
  inline size_t sizeInBytes(const VT *ptr, size_t N) {
    return N * sizeof(VT);
  }

  template <typename VT>
  inline float sizeInGBytes(const VT *ptr, size_t N) {
    return static_cast<float>(sizeInBytes(ptr, N)) / 1e9;
  }

  template <typename VT>
  inline void asyncMemcpyH2D(const VT *host, VT *dev, size_t N,
                             cudaStream_t stream = cudaStreamDefault) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(dev, host, N * sizeof(VT),
                                   cudaMemcpyHostToDevice, stream));
  }

  template <typename VT>
  inline void memcpyH2D(const VT *host, VT *dev, size_t N) {
    CHECK_CUDA_ERR(
      cudaMemcpy(dev, host, N * sizeof(VT), cudaMemcpyDeviceToHost));
  }

  template <typename VT>
  inline void asyncMemcpyD2H(const VT *dev, VT *host, size_t N,
                             cudaStream_t stream = cudaStreamDefault) {
    CHECK_CUDA_ERR(cudaMemcpyAsync(host, dev, N * sizeof(VT),
                                   cudaMemcpyDeviceToHost, stream));
  }

  template <typename VT>
  inline void memcpyD2H(const VT *dev, VT *host, size_t N) {
    CHECK_CUDA_ERR(
      cudaMemcpy(host, dev, N * sizeof(VT), cudaMemcpyDeviceToHost));
  }

  unsigned int getWarpSize(int devID = 0) {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    int ws;
    CHECK_CUDA_ERR(cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, devID));
    return ws;
  }

  unsigned int getSMCount(int devID = 0) {
    CHECK_CUDA_ERR(cudaSetDevice(devID));
    int SMcount;
    CHECK_CUDA_ERR(
      cudaDeviceGetAttribute(&SMcount, cudaDevAttrMultiProcessorCount, devID));
    return SMcount;
  }

}  // namespace cuda_helpers

#endif