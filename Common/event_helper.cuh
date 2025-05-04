#ifndef EVENT_HELPER_CUH
#define EVENT_HELPER_CUH

#pragma once

#include "cuda_error_handler.cuh"
#include "stream_helper.cuh"

namespace event_helper {
  namespace SH = stream_helper;
  class cudaEvent {
   public:
    cudaEvent(unsigned int flags = cudaEventDisableTiming) {
      CHECK_CUDA_ERR(cudaEventCreateWithFlags(&event_, flags));
    }

    ~cudaEvent() { cudaEventDestroy(event_); }

    cudaEvent(const cudaEvent &) = delete;  // Prevent copy
    cudaEvent &operator=(const cudaEvent &) = delete;

    cudaEvent(cudaEvent &&other) noexcept : event_(other.event_) {
      other.event_ = nullptr;
    }

    cudaEvent &operator=(cudaEvent &&other) noexcept {
      if (this != &other) {
        CHECK_CUDA_ERR(cudaEventDestroy(event_));
        event_ = other.event_;
        other.event_ = nullptr;
      }
      return *this;
    }

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream = 0) {
      CHECK_CUDA_ERR(cudaEventRecord(event_, stream));
    }

    void synchronize() const { CHECK_CUDA_ERR(cudaEventSynchronize(event_)); }

    float elapsedTimeSince(const cudaEvent &start) const {
      this->synchronize();
      float milliseconds = 0.0f;
      CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start.event_, event_));
      return milliseconds;
    }

    operator cudaEvent_t() const { return event_; }

    // Query event status
    bool isReady() const { return cudaEventQuery(event_) == cudaSuccess; }

    // Wait for event (alternative to synchronize)
    void wait() const { this->synchronize(); }

   private:
    cudaEvent_t event_;
  };

}  // namespace event_helper

#endif