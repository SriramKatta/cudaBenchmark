#include "cuda_helper.cuh"
#include "stream_helper.cuh"

namespace CH = cuda_helpers;
namespace SH = stream_helper;

int main(int argc, char const *argv[])
{
  auto devptr = CH::allocDevice<double>(50);
  auto hosptr = CH::allocHost<double>(50);
  SH::cudaStream stream1;
  CH::asyncMemcpyH2D(hosptr, devptr, 50);
  cudaDeviceSynchronize();
  return 0;
}
