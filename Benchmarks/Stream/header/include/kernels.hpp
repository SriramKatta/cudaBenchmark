#ifndef KERNELS_HPP
#define KERNELS_HPP
#pragma once

#include <stddef.h>

namespace testKernels {
  template <typename VT>
  __global__ void Stream(VT*, size_t);
}  // namespace testKernels


#endif