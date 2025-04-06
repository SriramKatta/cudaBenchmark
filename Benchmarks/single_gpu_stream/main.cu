#include "cuda_helper.cuh"
#include "kernels.cuh"
#include "stream_helper.cuh"

#include "cuda_errror_handler.cuh"

#include <cuda_runtime.h>

#define numstream 8

namespace CH = cuda_helpers;
namespace SH = stream_helper;

template <typename VT>
void fill(VT *ptr, size_t N) {
  for (size_t i = 0; i < N; i++) {
    ptr[i] = 1.0;
  }
}

int main(int argc, char const *argv[]) {
  size_t N = 512 * 1024 * 1024;
  auto dev_data = CH::allocDevice<double>(N);
  auto hos_data = CH::allocHost<double>(N);
  fill(hos_data.get(), N);
  size_t chunk = N / numstream;
#pragma omp parallel for
  for (size_t i = 0; i < numstream; i++) {
    SH::cudaStream stream;
    auto h_ptr = hos_data.get() + (i * chunk);
    auto d_ptr = dev_data.get() + (i * chunk);
    CH::asyncMemcpyH2D(h_ptr, d_ptr, chunk, stream);
    streamkernel<<<36, 256, 0, stream>>>(chunk, d_ptr);
    CH::asyncMemcpyD2H(d_ptr, h_ptr, chunk, stream);
  }

  return 0;
}
