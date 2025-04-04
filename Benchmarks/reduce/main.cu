#include "kernels.cuh"
#include "cuda_helper.cuh"
#include "stream_helper.cuh"
#include "event_helper.cuh"

#include <stdio.h>

using VT = int;

template <typename VT>
inline void launchGpuCopy(VT *__restrict__ i_data, VT *__restrict__ o_data, size_t N, dim3 blocks, dim3 threads, cudaStream_t stream = 0)
{
  copy<<<blocks, threads, 0, stream>>>(i_data, o_data, N);
  CHECK_CUDA_LASTERR("COPY KERNEL FAILED");
}

template <typename VT>
inline void launchGpuFill(VT *__restrict__ i_data, size_t N, VT fillval, dim3 blocks, dim3 threads, cudaStream_t stream = cudaStreamDefault)
{
  fillWith_n<<<blocks, threads, 0, stream>>>(i_data, N, fillval);
  CHECK_CUDA_LASTERR("FILL WITH N KERNEL FAILED");
}

int main(int argc, char const *argv[])
{
  if (argc != 2)
    argv[1] = "20";
  size_t N = 1 << atoi(argv[1]);
  VT *i_data = CH::allocDevice<VT>(N);
  VT *o_data = CH::allocDevice<VT>(N);
  VT *h_data = CH::allocHost<VT>(N);

  auto ws = CH::getWarpSize();
  auto smcount = CH::getSMCount();
  dim3 blks_1d(smcount * 100, 1, 1);
  dim3 thp_1d(ws * 32, 1, 1);

  SH::CudaStream fill1_s, fill2_s, compute_s;
  EH::CudaEvent fill1start_e, fill1stop_e, fill2start_e, fill2stop_e, computestart_e, computestop_e;

  fill1start_e.record(fill1_s);
  launchGpuFill(i_data, N, 3, blks_1d, thp_1d, fill1_s);
  fill1stop_e.record(fill1_s);

  fill2start_e.record(fill2_s);
  launchGpuFill(o_data, N, 0, blks_1d, thp_1d, fill2_s);
  fill2stop_e.record(fill2_s);

  fill1stop_e.synchronize();
  fill2stop_e.synchronize();

  computestart_e.record(compute_s);
  launchGpuCopy(i_data, o_data, N, blks_1d, thp_1d, compute_s);
  computestop_e.record(compute_s);

  auto fill_1_Time = fill1stop_e.elapsedTimeSince(fill1start_e) / 1e3;
  auto fill_2_Time = fill2stop_e.elapsedTimeSince(fill2start_e) / 1e3;
  auto copy_time = computestop_e.elapsedTimeSince(computestart_e) / 1e3;

  CH::memcpyD2H(o_data, h_data, N);

  for (size_t i = 0; i < 5; i++)
  {
    printf("h_data[%d] = %d | h_data[N - %d] = %d \n", i, h_data[i], i, h_data[N - i - 1]);
  }

  size_t ds_gb = CH::sizeInBytes(i_data, N) / (1024 * 1024 * 1024);

  printf("fill1 bandwidth %5.3f | fill2 bandwidth %5.3f | copy bandwidthtime %5.3f \n", ds_gb / fill_1_Time, ds_gb / fill_2_Time, ds_gb / copy_time);

  return 0;
}
