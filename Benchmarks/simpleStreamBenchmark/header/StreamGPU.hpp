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
#include "kernels.hpp"
#include "stream_helper.cuh"

namespace CH = cuda_helpers;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;

template <typename VT>
std::tuple<float, float, float> benchmarkStream(
  size_t NumReps, size_t N, VT *host, VT *dev,
  size_t NumBlocks, size_t NumThredsPBlock, bool verboseinfo) {

    float timerh2d{0.0};
  float timerd2h{0.0};
  float timerkern{0.0};
  TH::cudaTimer timer;


  for (size_t rep = 0; rep < NumReps; rep++) {

    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      if (verboseinfo) {
        timer.setStream(streams[i]);
        timer.start();
      }
      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         streams[i]);
      if (verboseinfo) {
        timer.stop();
        timerh2d += timer.elapsedSeconds();
      }
      chunkstart += currentChunkSize;
    }

    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      if (verboseinfo) {
        timer.setStream(streams[i]);
        timer.start();
      }
      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, streams[i]>>>(
        dev + chunkstart, currentChunkSize);
      if (verboseinfo) {
        timer.stop();
        timerkern += timer.elapsedSeconds();
      }
      CHECK_CUDA_LASTERR("Stream Launch failure");
      chunkstart += currentChunkSize;
    }

    for (size_t chunkstart = 0, i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      if (verboseinfo) {
        timer.setStream(streams[i]);
        timer.start();
      }
      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         streams[i]);
      if (verboseinfo) {
        timer.stop();
        timerd2h += timer.elapsedSeconds();
      }
      chunkstart += currentChunkSize;
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
  }

  return {timerh2d / NumReps, timerkern / NumReps, timerd2h / NumReps};
}

#endif