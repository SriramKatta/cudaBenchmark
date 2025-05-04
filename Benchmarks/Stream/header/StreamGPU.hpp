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
std::tuple<float, float, float> benchmarkRunWithStreamPool(
  size_t NumReps, size_t N, size_t NumStreams, VT *host, VT *dev,
  size_t NumBlocks, size_t NumThredsPBlock, bool verboseinfo) {

  size_t baseChunkSize = N / NumStreams;
  size_t remainder = N % NumStreams;

  float timerh2d{0.0};
  float timerd2h{0.0};
  float timerkern{0.0};
  TH::cudaTimer timer;


  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop stream pool"};
    std::vector<SH::cudaStream> streams(NumStreams);


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

template <typename VT>
std::tuple<float, float, float> benchmarkRunWithPerChunkStream(
  size_t NumReps, size_t N, size_t NumStreams, VT *host, VT *dev,
  size_t NumBlocks, size_t NumThredsPBlock, bool verboseinfo = true) {
  size_t baseChunkSize = N / NumStreams;
  size_t remainder = N % NumStreams;
  float timerh2d{0.0};
  float timerd2h{0.0};
  float timerkern{0.0};
  TH::cudaTimer h2d;
  TH::cudaTimer kernel;
  TH::cudaTimer d2h;
  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop perchunk stream"};
    size_t chunkstart = 0;
    for (size_t i = 0; i < NumStreams; i++) {
      SH::cudaStream streams;
      if (verboseinfo) {
        h2d.setStream(streams);
        kernel.setStream(streams);
        d2h.setStream(streams);
      }

      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);

      if (verboseinfo)
        h2d.start();

      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         streams);

      if (verboseinfo)
        h2d.stop();

      if (verboseinfo)
        kernel.start();

      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, streams>>>(
        dev + chunkstart, currentChunkSize);

      if (verboseinfo)
        kernel.stop();

      CHECK_CUDA_LASTERR("Kernel Launch failure");

      if (verboseinfo)
        d2h.start();

      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         streams);

      if (verboseinfo)
        d2h.stop();

      chunkstart += currentChunkSize;

      if (verboseinfo) {
        timerh2d += h2d.elapsedSeconds();
        timerd2h += d2h.elapsedSeconds();
        timerkern += kernel.elapsedSeconds();
      }
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
  }
  return {timerh2d / NumReps, timerkern / NumReps, timerd2h / NumReps};
}

template <typename VT>
void checkSolution(VT *data, size_t N, size_t reps) {
  VT error = static_cast<VT>(0);
  for (size_t i = 0; i < N; i++) {
    error += std::abs(data[i] - reps);
  }
  if (fabs(error) > 1e-12) {
    fmt::print("Check failed\n", error);
    for (size_t i = 0; i < N; ++i) {
      fmt::print("{} ", data[i]);
    }
    exit(EXIT_FAILURE);
  }
}


#endif