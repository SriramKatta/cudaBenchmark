#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "cuda_errror_handler.cuh"
#include "cuda_helper.cuh"
#include "cuda_timer.cuh"
#include "kernels.cuh"
#include "stream_helper.cuh"


#define NUM_REPS 30

namespace CH = cuda_helpers;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;

template <typename VT>
void fill(VT *ptr, size_t N) {
  for (size_t i = 0; i < N; i++) {
    ptr[i] = 1.0;
  }
}

int main(int argc, char const *argv[]) {
  if (argc != 2)
    argv[1] = "5";
  size_t N = std::pow(1.2, atoi(argv[1]));
  TH::cudaTimer streamtimer;
  TH::cudaTimer H2Dtimer;
  TH::cudaTimer D2Htimer;
  auto dev_data = CH::allocDevice<double>(N);
  auto hos_data = CH::allocHost<double>(N);
  fill(hos_data.get(), N);

  H2Dtimer.start();
  CH::memcpyH2D(hos_data, dev_data, N);
  H2Dtimer.stop();

  dim3 thpblks = CH::getWarpSize() * 16;
  dim3 blks = CH::getSMCount() * 20;

  streamtimer.start();
  for (size_t rep = 0; rep < NUM_REPS; rep++) {
    SH::cudaStream stream;
    streamkernel<<<blks, thpblks,0, stream>>>(dev_data.get(), N);
  }
  streamtimer.stop();

  D2Htimer.start();
  CH::memcpyD2H(dev_data, hos_data, N);
  D2Htimer.stop();

  auto datasize = CH::sizeInBytes(dev_data, N);
  auto H2DBW = datasize / H2Dtimer.elapsedSeconds() / 1e9;
  auto streamBW =
    2.0 * NUM_REPS * datasize / streamtimer.elapsedSeconds() / 1e9;
  auto D2HBW = datasize / D2Htimer.elapsedSeconds() / 1e9;

  printf(
    "datasize : %10lu B | H2Dbandwidth : %4.3f GB/s | D2Hbandwidth : %4.3f "
    "GB/s | kernbandwidth : %4.3f GB/s\n",
    datasize, H2DBW, D2HBW, streamBW);
  return 0;
}
