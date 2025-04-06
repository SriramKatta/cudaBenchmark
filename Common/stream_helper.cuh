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
    cudaStream(bool use_default_stream = true,
               unsigned int flags = cudaStreamDefault, int priority = 0)
        : owns_stream_(!use_default_stream) {
      if (use_default_stream) {
        stream_ = 0;  // default stream (aka cudaStreamLegacy)
      } else {
        CHECK_CUDA_ERR(cudaStreamCreateWithPriority(&stream_, flags, priority));
      }
    }

    ~cudaStream() {
      if (owns_stream_ && stream_) {
        cudaStreamDestroy(stream_);
      }
    }

    cudaStream(const cudaStream &) = delete;
    cudaStream &operator=(const cudaStream &) = delete;

    cudaStream(cudaStream &&other) noexcept
        : stream_(other.stream_), owns_stream_(other.owns_stream_) {
      other.stream_ = nullptr;
      other.owns_stream_ = false;
    }

    cudaStream &operator=(cudaStream &&other) noexcept {
      if (this != &other) {
        if (owns_stream_ && stream_) {
          cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        owns_stream_ = other.owns_stream_;
        other.stream_ = nullptr;
        other.owns_stream_ = false;
      }
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
    bool owns_stream_ = false;
  };

}  // namespace stream_helper

#endif
