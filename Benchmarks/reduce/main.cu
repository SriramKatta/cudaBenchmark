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

template <typename VT>
inline void launchGpuCopy(VT *__restrict__ i_data, VT *__restrict__ o_data,
                          size_t N, dim3 blocks, dim3 threads,
                          cudaStream_t stream = 0) {
  copy<<<blocks, threads, 0, stream>>>(i_data, o_data, N);
  CHECK_CUDA_LASTERR("COPY KERNEL FAILED");
}

template <typename VT>
inline void launchGpuFill(VT *__restrict__ i_data, size_t N, VT fillval,
                          dim3 blocks, dim3 threads,
                          cudaStream_t stream = cudaStreamDefault) {
  fillWith_n<<<blocks, threads, 0, stream>>>(i_data, N, fillval);
  CHECK_CUDA_LASTERR("FILL WITH N KERNEL FAILED");
}

int main(int argc, char const *argv[]) {
  if (argc != 2)
    argv[1] = "23";
  size_t N = 1 << atoi(argv[1]);
  auto i_data = CH::allocDevice<VT>(N);
  auto o_data = CH::allocDevice<VT>(N);
  auto h_data = CH::allocHost<VT>(N);

  auto ws = CH::getWarpSize();
  auto smcount = CH::getSMCount();
  dim3 blks_1d(smcount * 100, 1, 1);
  dim3 thp_1d(ws * 32, 1, 1);

  SH::cudaStream fill1_s, fill2_s, compute_s;
  TH::cudaTimer fill1timer(fill1_s);
  TH::cudaTimer fill2timer(fill2_s);
  TH::cudaTimer computetimer(compute_s);

  fill1timer.start();
  launchGpuFill(i_data, N, static_cast<VT>(3), blks_1d, thp_1d, fill1_s);
  fill1timer.stop();

  fill2timer.start();
  launchGpuFill(o_data, N, static_cast<VT>(0), blks_1d, thp_1d, fill2_s);
  fill2timer.stop();

  fill1_s.synchronize();
  fill2_s.synchronize();

  computetimer.start();
  launchGpuCopy(i_data, o_data, N, blks_1d, thp_1d, compute_s);
  computetimer.stop();

  auto fill_1_Time = fill1timer.elapsedSeconds();
  auto fill_2_Time = fill2timer.elapsedSeconds();
  auto copy_time = computetimer.elapsedSeconds();

  CH::memcpyD2H(o_data, h_data, N);

  for (size_t i = 0; i < 5; i++) {
    //printf("h_data[%u] = %d | h_data[N - %u] = %d \n", i, h_data[i], i,
    //       h_data[N - i - 1]);
  }

  float ds_gb = static_cast<float>(CH::sizeInBytes(i_data, N)) / 1e9;

  printf(
    "datasize : %3.3f GB | fill1 bandwidth %5.3f GB/s | fill2 bandwidth %5.3f "
    "GB/s | copy bandwidthtime %5.3f GB/s\n",
    ds_gb, ds_gb / fill_1_Time, ds_gb / fill_2_Time, ds_gb / copy_time);

  return 0;
}
