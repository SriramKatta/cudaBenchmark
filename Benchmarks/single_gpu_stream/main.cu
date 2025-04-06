#include "cuda_helper.cuh"
#include "kernels.cuh"
#include "stream_helper.cuh"

#include "cuda_errror_handler.cuh"

#include <cuda_runtime.h>

namespace CH = cuda_helpers;
namespace SH = stream_helper;

int main(int argc, char const *argv[]) {
  size_t N = 512 * 1024 * 1024;
  auto devptr = CH::allocDevice<double>(N);
  auto hosptr = CH::allocHost<double>(N);
  CHECK_CUDA_ERR(cudaMemset(devptr.get(), 0, N * sizeof(double)));
  SH::cudaStream stream1;
  SH::cudaStream stream2;
  auto ptr = devptr.get();
  stream<<<50, 256, 0, stream1>>>(N, ptr);
  CHECK_CUDA_LASTERR("STREAM KERNEL LAUNCH FAILED");
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  CH::asyncMemcpyD2H(devptr, hosptr, N, stream2);
  CHECK_CUDA_ERR(cudaDeviceSynchronize());
  return 0;
}
