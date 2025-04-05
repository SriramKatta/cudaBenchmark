#include "cuda_helper.cuh"
#include "cuda_timer.cuh"
#include "kernels.cuh"
#include "stream_helper.cuh"

#include <stdio.h>

using VT = double;

#define REPS 30

namespace CH = cuda_helpers;
namespace EH = event_helper;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;


int main(int argc, char const *argv[]) {
  if (argc != 2)
    argv[1] = "5";
  size_t N = std::pow(1.2, atoi(argv[1]));
  auto d_data = CH::allocDevice<VT>(N);
  auto h_data = CH::allocHost<VT>(N);

  auto ws = CH::getWarpSize();
  auto smcount = CH::getSMCount();
  dim3 blks_1d(smcount * 20, 1, 1);
  dim3 thp_1d(ws * 32, 1, 1);

  for (size_t i = 0; i < N; i++) {
    h_data[i] = 0.0;
  }

  TH::cudaTimer H2Dtimer;
  TH::cudaTimer D2Htimer;
  TH::cudaTimer streamtimer;

  H2Dtimer.start();
  CH::memcpyH2D(h_data, d_data, N);
  H2Dtimer.stop();


  streamtimer.start();
  for (int rep = 0; rep < REPS; ++rep) {
    kernel<<<blks_1d, thp_1d>>>(N, d_data);
  }
  streamtimer.stop();

  D2Htimer.start();
  CH::memcpyD2H(d_data, h_data, N);
  D2Htimer.stop();

  auto H2D_Time = H2Dtimer.elapsedSeconds();
  auto D2H_Time = D2Htimer.elapsedSeconds();
  auto stream_time = streamtimer.elapsedSeconds() / REPS;


  float ds_gb = CH::sizeInGBytes(d_data, N);

  printf(
    "datasize : %lu B | H2D bandwidth %5.3f GB/s | D2H bandwidth %5.3f "
    "GB/s | stream bandwidthtime %5.3f GB/s\n",
    CH::sizeInBytes(d_data, N), ds_gb / H2D_Time, ds_gb / D2H_Time,
    2.0 * ds_gb / stream_time);

  return 0;
}
