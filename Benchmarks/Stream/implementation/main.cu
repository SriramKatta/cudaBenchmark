#include <fmt/format.h>
#include <exception>

#include "StreamGPU.hpp"
#include "cuda_timer.cuh"
#include "parseCLA.hpp"

namespace TH = cuda_timer_helper;


int main(int argc, char const *argv[]) {
  size_t N;
  size_t NumReps;
  size_t NumBlocks;
  size_t NumThredsPBlock;
  size_t NumStreams;
  bool verbose;
  bool doCheck;
  parseCLA(argc, argv, N, NumReps, NumBlocks, NumThredsPBlock, NumStreams,
           verbose, doCheck);

  auto dev_ptr = CH::allocDevice<double>(N);
  auto host_ptr = CH::allocHost<double>(N);

  auto sizeinGB = CH::sizeInGBytes(dev_ptr, N);


  auto dev = dev_ptr.get();
  auto host = host_ptr.get();

  CH::initHost(host_ptr, N, 0.0);
  TH::cudaTimer fullwork;

  fullwork.start();
  auto [h2dv1, kernv1, d2hv1] = benchmarkRunWithPerChunkStream(
    NumReps, N, NumStreams, host, dev, NumBlocks, NumThredsPBlock, verbose);
  fullwork.stop();

  if (doCheck) {
    checkSolution(host, N, NumReps);
  }

  auto elapsedV1 = fullwork.elapsedSeconds() / NumReps;

  CH::initHost(host_ptr, N, 0.0);

  fullwork.start();

  auto [h2dv2, kernv2, d2hv2] = benchmarkRunWithStreamPool(
    NumReps, N, NumStreams, host, dev, NumBlocks, NumThredsPBlock, verbose);

  fullwork.stop();

  if (doCheck) {
    checkSolution(host, N, NumReps);
  }

  auto elapsedV2 = fullwork.elapsedSeconds() / NumReps;

  if (verbose) {
    fmt::print(
      "D={},R={},B={},T={},S={}\n"
      "V1={:.8f},V1_h2d={:.3f},V1_kernel={:.3f},V1_d2h={:.3f}\n"
      "V2={:.8f},V2_h2d={:.3f},V2_kernel={:.3f},V2_d2h={:.3f}\n",
      static_cast<size_t>(sizeinGB * 1e9), NumReps, NumBlocks, NumThredsPBlock,
      NumStreams, elapsedV1, sizeinGB / h2dv1, sizeinGB * 2.0 / kernv1,
      sizeinGB / d2hv1, elapsedV2, sizeinGB / h2dv2, sizeinGB * 2.0 / kernv2,
      sizeinGB / d2hv2);
  }


  return 0;
}
