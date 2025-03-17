#ifndef EVENT_HELPER_CUH
#define EVENT_HELPER_CUH

#pragma once

#include "cuda_errror_handler.cuh"

namespace EH
{

  class CudaEvent
  {
  public:
    CudaEvent(unsigned int flags = cudaEventDefault)
    {
      cudaEventCreateWithFlags(&event_, flags);
    }

    ~CudaEvent()
    {
      cudaEventDestroy(event_);
    }

    CudaEvent(const CudaEvent &) = delete; // Prevent copy
    CudaEvent &operator=(const CudaEvent &) = delete;

    CudaEvent(CudaEvent &&other) noexcept : event_(other.event_)
    {
      other.event_ = nullptr;
    }

    CudaEvent &operator=(CudaEvent &&other) noexcept
    {
      if (this != &other)
      {
        cudaEventDestroy(event_);
        event_ = other.event_;
        other.event_ = nullptr;
      }
      return *this;
    }

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream = 0)
    {
      cudaEventRecord(event_, stream);
    }

    void synchronize()
    {
      cudaEventSynchronize(event_);
    }

    float elapsedTimeSince(const CudaEvent &start)
    {
      //this->synchronize();
      float milliseconds = 0.0f;
      cudaEventElapsedTime(&milliseconds, start.event_, event_);
      return milliseconds;
    }
    operator cudaEvent_t() const
    {
      return event_;
    }

  private:
    cudaEvent_t event_;
  };

} // namespace EH

#endif