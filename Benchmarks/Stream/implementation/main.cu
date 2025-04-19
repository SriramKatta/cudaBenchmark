#include <fmt/format.h>
#include <exception>

#include "cuda_errror_handler.cuh"
#include "cuda_helper.cuh"
#include "kernels.hpp"
#include "parseCLA.hpp"
#include "stream_helper.cuh"

namespace CH = cuda_helpers;
namespace SH = stream_helper;
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

  for (size_t rep = 0; rep < NumReps; rep++) {
    size_t baseChunkSize = N / NumStreams;
    size_t remainder = N % NumStreams;
    size_t chunkstart = 0;
    for (size_t i = 0; i < NumStreams; i++) {
      size_t currentChunkSize = baseChunkSize + (i < remainder ? 1 : 0);
      SH::cudaStream stream;

      CH::asyncMemcpyH2D(host + chunkstart, dev + chunkstart, currentChunkSize,
                         stream);
      stream_kernel<<<NumBlocks, NumThredsPBlock>>>(dev + chunkstart,
                                                    currentChunkSize);
      CHECK_CUDA_LASTERR("Stream Launch failure");
      CH::asyncMemcpyD2H(dev + chunkstart, host + chunkstart, currentChunkSize,
                         stream);
      chunkstart += currentChunkSize;
    }
  }
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  checkSolution(host, N, NumReps);

  return 0;
}
