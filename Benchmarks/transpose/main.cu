#include "kernels.cuh"
#include "cuda_helper.cuh"
#include "stream_helper.cuh"
#include "event_helper.cuh"

#include <stdio.h>

template <typename VT>
void launchGpuCopy(VT *__restrict__ i_data, VT *__restrict__ o_data, size_t N, dim3 blocks, dim3 threads, cudaStream_t stream = 0)
{
  copy<<<blocks, threads, 0, stream>>>(i_data, o_data, N);
}

template <typename VT>
void launchGpuFill(VT *__restrict__ i_data, size_t N, VT fillval, dim3 blocks, dim3 threads, cudaStream_t stream = 0)
{
  fillWith_n<<<blocks, threads, 0, stream>>>(i_data, N, fillval);
}

int main(int argc, char const *argv[])
{
  size_t N = 1 << 29;
  auto i_data = CH::allocDevice<float>(N);
  auto o_data = CH::allocDevice<float>(N);
  auto ws = CH::getWarpSize();
  auto smcount = CH::getSMCount();
  dim3 blks_1d(smcount * 40, 1, 1);
  dim3 thp_1d(ws * 16, 1, 1);

  SH::CudaStream fill1_s, fill2_s, compute_s;
  EH::CudaEvent fill1start_e;
  EH::CudaEvent fill1stop_e;
  EH::CudaEvent fill2start_e;
  EH::CudaEvent fill2stop_e;
  EH::CudaEvent computestart_e;
  EH::CudaEvent computestop_e;

  fill1start_e.record(fill1_s);
  launchGpuFill(i_data, N, 3.0f, blks_1d, thp_1d, fill1_s);
  fill1stop_e.record(fill1_s);

  fill2start_e.record(fill2_s);
  launchGpuFill(o_data, N, 0.0f, blks_1d, thp_1d, fill2_s);
  fill2stop_e.record(fill2_s);

  fill1stop_e.synchronize();
  fill2stop_e.synchronize();

  auto fill_1_Time = fill1stop_e.elapsedTimeSince(fill1start_e);
  auto fill_2_Time = fill2stop_e.elapsedTimeSince(fill2start_e);

  computestart_e.record(compute_s);
  launchGpuCopy(i_data, o_data, N, blks_1d, thp_1d, compute_s);
  computestop_e.record(compute_s);

  auto copy_time = computestop_e.elapsedTimeSince(computestart_e);

  printf("fill1 time %5.3f | fill2 time %5.3f | copy time %5.3f \n", fill_1_Time, fill_2_Time, copy_time);

  return 0;
}
