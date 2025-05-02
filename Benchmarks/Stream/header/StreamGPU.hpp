#ifndef STREAMGPU_HPP
#define STREAMGPU_HPP
#pragma once


#include <cuda_runtime.h>
#include <fmt/format.h>
#include <limits>
#include <nvtx3/nvtx3.hpp>

#include "StreamGPU.hpp"
#include "cuda_error_handler.cuh"
#include "cuda_helper.cuh"
#include "cuda_timer.cuh"
#include "stream_helper.cuh"
#include "kernels.hpp"

namespace CH = cuda_helpers;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;

template <typename VT>
void setHostArr(size_t N, VT *host) {
  for (size_t i = 0; i < N; i++) {
    host[i] = 0.0;
  }
}

template <typename VT>
void benchmarkRunWithStreamPool(size_t NumReps, size_t N, size_t NumStreams,
                                VT *host, VT *dev, size_t NumBlocks,
                                size_t NumThredsPBlock) {
  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop stream pool"};
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    std::vector<SH::cudaStream> streams(NumStreams);
    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         streams[i]);
      chunkstart += currentChunkSize;
    }

    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, streams[i]>>>(
        dev + chunkstart, currentChunkSize);
      CHECK_CUDA_LASTERR("Stream Launch failure");
      chunkstart += currentChunkSize;
    }

    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         streams[i]);
      chunkstart += currentChunkSize;
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
  }
}

template <typename VT>
void benchmarkRunWithPerChunkStream(size_t NumReps, size_t N, size_t NumStreams,
                                    VT *host, VT *dev, size_t NumBlocks,
                                    size_t NumThredsPBlock) {
  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop perchunk stream"};
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    size_t chunkstart = 0;

    for (size_t i = 0; i < NumStreams; i++) {
      SH::cudaStream streams;
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);


      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         streams);

      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, streams>>>(
        dev + chunkstart, currentChunkSize);
      CHECK_CUDA_LASTERR("Kernel Launch failure");

      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         streams);
      chunkstart += currentChunkSize;
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
  }
}

template <typename VT>
void checkSolution(VT *data, size_t N, size_t reps) {
  VT error = static_cast<VT>(0);
  for (size_t i = 0; i < N; i++) {
    error += std::abs(data[i] - reps);
  }
  if (fabs(error) < std::numeric_limits<VT>::epsilon()) {
    fmt::print("Check failed\n", error);
    exit(EXIT_FAILURE);
  }
}


#endif