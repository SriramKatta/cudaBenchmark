#include <fmt/format.h>
#include <exception>

#include "StreamGPU.hpp"
#include "parseCLA.hpp"
#include "cuda_timer.cuh"

namespace TH = cuda_timer_helper;


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
