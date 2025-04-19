#ifndef STREAM_HELPER_CUH
#define STREAM_HELPER_CUH

#pragma once

#include <cuda_runtime.h>

#include "cuda_errror_handler.cuh"

namespace stream_helper {

  class cudaStream {
   public:
    // Default: use_default_stream = true -> use default stream
    // Otherwise, create new stream with given flags and priority
    cudaStream(unsigned int flags = cudaStreamDefault, int priority = 0) {
      CHECK_CUDA_ERR(cudaStreamCreateWithPriority(&stream_, flags, priority));
    }

    ~cudaStream() { cudaStreamDestroy(stream_); }

    cudaStream(const cudaStream &) = delete;
    cudaStream &operator=(const cudaStream &) = delete;

    cudaStream(cudaStream &&other) noexcept : stream_(other.stream_) {
      other.stream_ = nullptr;
    }

    cudaStream &operator=(cudaStream &&other) noexcept {
      if (this != &other) {
        cudaStreamDestroy(stream_);
      }
      stream_ = other.stream_;
      other.stream_ = nullptr;
      return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }

    int getPriority() const {
      int prio;
      CHECK_CUDA_ERR(cudaStreamGetPriority(stream_, &prio));
      return prio;
    }

    cudaStream_t operator*() const noexcept { return this->get(); }

    void synchronize() { CHECK_CUDA_ERR(cudaStreamSynchronize(stream_)); }

    operator cudaStream_t() const noexcept { return stream_; }

   private:
    cudaStream_t stream_ = nullptr;
  };

}  // namespace stream_helper

#endif
