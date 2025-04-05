#ifndef STREAM_HELPER_CUH
#define STREAM_HELPER_CUH

#pragma once

#include "cuda_errror_handler.cuh"

namespace stream_helper {

  class cudaStream {
   public:
    cudaStream(unsigned int flags = cudaStreamDefault, int priority = 0) {
      CHECK_CUDA_ERR(cudaStreamCreateWithPriority(&stream_, flags, priority));
    }

    ~cudaStream() { cudaStreamDestroy(stream_); }

    cudaStream(const cudaStream &) = delete;  // Prevent copy
    cudaStream &operator=(const cudaStream &) = delete;

    cudaStream(cudaStream &&other) noexcept : stream_(other.stream_) {
      other.stream_ = nullptr;
    }

    cudaStream &operator=(cudaStream &&other) noexcept {
      if (this != &other) {
        cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
      }
      return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }

    int getPriority() const noexcept {
      int prio;
      cudaStreamGetPriority(stream_, &prio);
      return prio;
    }

    cudaStream_t operator*() const noexcept { return this->get(); }

    void synchronize() { CHECK_CUDA_ERR(cudaStreamSynchronize(stream_)); }

    operator cudaStream_t() const noexcept { return stream_; }

   private:
    cudaStream_t stream_;
  };

}  // namespace stream_helper

#endif