#ifndef STREAM_HELPER_CUH
#define STREAM_HELPER_CUH

#pragma once

#include "cuda_errror_handler.cuh"

namespace SH
{
  class CudaStream
  {
  public:
    CudaStream(unsigned int flags = cudaStreamDefault)
    {
      CHECK_CUDA_ERR(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream()
    {
      cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream &) = delete; // Prevent copy
    CudaStream &operator=(const CudaStream &) = delete;

    CudaStream(CudaStream &&other) noexcept : stream_(other.stream_)
    {
      other.stream_ = nullptr;
    }

    CudaStream &operator=(CudaStream &&other) noexcept
    {
      if (this != &other)
      {
        cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
      }
      return *this;
    }

    cudaStream_t get() const { return stream_; }

    cudaStream_t operator*()
    {
      return this->get();
    }

    void synchronize()
    {
      CHECK_CUDA_ERR(cudaStreamSynchronize(stream_));
    }

    operator cudaStream_t() const{
      return stream_;
    }

  private:
    cudaStream_t stream_;
  };

} // namespace SH

#endif