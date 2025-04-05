#include "cuda_helper.cuh"
#include "cuda_timer.cuh"
#include "kernels.cuh"
#include "stream_helper.cuh"

#include <stdio.h>

using VT = double;

namespace CH = cuda_helpers;
namespace EH = event_helper;
namespace SH = stream_helper;
namespace TH = cuda_timer_helper;


int main(int argc, char const *argv[]) {
  if (argc != 2)
    argv[1] = "5";
  size_t N = std::pow(1.2, atoi(argv[1]));
  auto i_data = CH::allocDevice<VT>(N);
  auto h_data = CH::allocHost<VT>(N);

  auto ws = CH::getWarpSize();
  auto smcount = CH::getSMCount();
  dim3 blks_1d(smcount * 20, 1, 1);
  dim3 thp_1d(ws * 32, 1, 1);

  for (size_t i = 0; i < N; i++) {
    i_data[i] = 0.0;
  }

  TH::cudaTimer H2Dtimer;
  TH::cudaTimer D2Htimer;
  TH::cudaTimer streamtimer;


  auto H2D_Time = H2Dtimer.elapsedSeconds();
  auto D2H_Time = D2Htimer.elapsedSeconds();
  auto stream_time = streamtimer.elapsedSeconds();

  CH::memcpyD2H(o_data, h_data, N);

  float ds_gb = static_cast<float>(CH::sizeInBytes(i_data, N)) / 1e9;

  printf(
    "datasize : %3.3f GB | fill1 bandwidth %5.3f GB/s | fill2 bandwidth %5.3f "
    "GB/s | copy bandwidthtime %5.3f GB/s\n",
    ds_gb, ds_gb / fill_1_Time, ds_gb / fill_2_Time, ds_gb / copy_time);

  return 0;
}
