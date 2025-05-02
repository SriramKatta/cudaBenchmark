#include <fmt/format.h>
#include <exception>
#include <limits>
#include <nvtx3/nvtx3.hpp>

#include "cuda_errror_handler.cuh"
#include "cuda_helper.cuh"
#include "cuda_timer.cuh"
#include "kernels.hpp"
#include "parseCLA.hpp"
#include "stream_helper.cuh"

namespace CH = cuda_helpers;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;

template <typename VT>
void checkSolution(VT *data, size_t N, size_t reps);

template <typename VT>
void setHostArr(size_t N, VT *host);

template <typename VT>
void benchmarkRunWithPerChunkStream(size_t NumReps, size_t N, size_t NumStreams,
                                    VT *host, VT *dev, size_t NumBlocks,
                                    size_t NumThredsPBlock);

template <typename VT>
void benchmarkRunWithStreamPool(size_t NumReps, size_t N, size_t NumStreams,
                                VT *host, VT *dev, size_t NumBlocks,
                                size_t NumThredsPBlock);


int main(int argc, char const *argv[]) {
  size_t N;
  size_t NumReps;
  size_t NumBlocks;
  size_t NumThredsPBlock;
  size_t NumStreams;
  bool doCheck;
  try {
    parseCLA(argc, argv, N, NumReps, NumBlocks, NumThredsPBlock, NumStreams,
             doCheck);
  } catch (std::exception &e) {
    fmt::print("Error : {}\n", e.what());
    exit(1);
  }
  auto dev_ptr = CH::allocDevice<double>(N);
  auto host_ptr = CH::allocHost<double>(N);

  auto dev = dev_ptr.get();
  auto host = host_ptr.get();

  setHostArr(N, host);
  TH::cudaTimer fullwork;
  fullwork.start();
  benchmarkRunWithPerChunkStream(NumReps, N, NumStreams, host, dev, NumBlocks,
                                 NumThredsPBlock);
  fullwork.stop();

  if (doCheck) {
    checkSolution(host, N, NumReps);
  }

  auto elapsed_time = fullwork.elapsedSeconds() / NumReps;
  fmt::print("elapsed time V1 per rep is {}\n", elapsed_time);

  setHostArr(N, host);
  TH::cudaTimer fullwork2;
  fullwork2.start();
  benchmarkRunWithStreamPool(NumReps, N, NumStreams, host, dev, NumBlocks,
                             NumThredsPBlock);
  fullwork2.stop();

  if (doCheck) {
    checkSolution(host, N, NumReps);
  }

  elapsed_time = fullwork2.elapsedSeconds() / NumReps;
  fmt::print("elapsed time V2 per rep is {}\n", elapsed_time);

  return 0;
}

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
    nvtx3::scoped_range loop{"main loop V2"};
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
    nvtx3::scoped_range loop{"main loop V1"};
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    size_t chunkstart = 0;

    std::vector<SH::cudaStream> streams(NumStreams);
    std::vector<TH::cudaTimer> h2dTimers, d2hTimers, kernel;

    for (size_t i = 0; i < NumStreams; i++) {
      h2dTimers.push_back(streams[i]);
      d2hTimers.push_back(streams[i]);
      kernel.push_back(streams[i]);
    }

    for (size_t i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);

      h2dTimers[i].start();
      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         streams[i]);
      h2dTimers[i].stop();

      kernel[i].start();
      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, streams[i]>>>(
        dev + chunkstart, currentChunkSize);
      CHECK_CUDA_LASTERR("Kernel Launch failure");
      kernel[i].stop();

      d2hTimers[i].start();
      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         streams[i]);
      d2hTimers[i].stop();
      chunkstart += currentChunkSize;
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Report bandwidths
    for (size_t i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      size_t chunkBytes = currentChunkSize * sizeof(VT);

      float h2d_ms = h2dTimers[i].elapsedMilliseconds();
      float d2h_ms = d2hTimers[i].elapsedMilliseconds();

      float h2d_bw = (chunkBytes / (h2d_ms * 1e-3)) / 1e9;  // GB/s
      float d2h_bw = (chunkBytes / (d2h_ms * 1e-3)) / 1e9;  // GB/s

      std::cout << "Rep " << rep << ", Stream " << i << " | H2D: " << h2d_ms
                << " ms, BW: " << h2d_bw << " GB/s"
                << " | D2H: " << d2h_ms << " ms, BW: " << d2h_bw << " GB/s\n";
    }
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
