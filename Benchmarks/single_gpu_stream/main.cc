#include "cuda_helper.cuh"

namespace CH = cuda_helpers;

int main(int argc, char const *argv[])
{
  auto ptr = CH::allocDevice<double>(50);
  return 0;
}
