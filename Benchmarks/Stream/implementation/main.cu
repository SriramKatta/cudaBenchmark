#include <fmt/format.h>
#include <exception>
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
void checkSolution(VT *data, size_t N, size_t reps) {
  VT error = static_cast<VT>(0);
  for (size_t i = 0; i < N; i++) {
    error += std::abs(data[i] - reps);
  }
  fmt::print("error : {}\n", error);
}

int main(int argc, char const *argv[]) {
  size_t N;
  size_t NumReps;
  size_t NumBlocks;
  size_t NumThredsPBlock;
  size_t NumStreams;
  try {
    parseCLA(argc, argv, N, NumReps, NumBlocks, NumThredsPBlock, NumStreams);
  } catch (std::exception &e) {
    fmt::print("Error : {}\n", e.what());
    exit(1);
  }
  auto dev_ptr = CH::allocDevice<double>(N);
  auto host_ptr = CH::allocHost<double>(N);
  auto dev = dev_ptr.get();
  auto host = host_ptr.get();
  for (size_t i = 0; i < N; i++) {
    host[i] = 0.0;
  }
  TH::cudaTimer fullwork;
  fullwork.start();
  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop V1"};
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    size_t chunkstart = 0;
    for (size_t i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      SH::cudaStream stream;

      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         stream);
      stream_kernel<<<NumBlocks, NumThredsPBlock, 0, stream>>>(
        dev + chunkstart, currentChunkSize);
      CHECK_CUDA_LASTERR("Stream Launch failure");
      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         stream);
      chunkstart += currentChunkSize;
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
  }
  fullwork.stop();
  auto elapsed_time = fullwork.elapsedSeconds() / NumReps;
  fmt::print("elapsed time V1 per rep is {}\n", elapsed_time);
  //checkSolution(host, N, NumReps);

  fullwork.start();
  for (size_t rep = 0; rep < NumReps; rep++) {
    nvtx3::scoped_range loop{"main loop V2"};
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    //size_t chunkstart = 0;
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
  fullwork.stop();
  elapsed_time = fullwork.elapsedSeconds() / NumReps;
  fmt::print("elapsed time V2 per rep is {}\n", elapsed_time);


  return 0;
}
