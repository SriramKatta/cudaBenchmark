#ifndef CUDA_TIMER_CUH
#define CUDA_TIMER_CUH
#pragma once

#include "cuda_errror_handler.cuh"
#include "event_helper.cuh"
#include "stream_helper.cuh"

namespace cuda_timer_helper {
  namespace EH = event_helper;
  namespace SH = stream_helper;
  class cudaTimer {
   public:
    cudaTimer()
        : start_(cudaEventDefault), stop_(cudaEventDefault), stream_(0) {
      defstream = true;
    }

    cudaTimer(const SH::cudaStream &stream)
        : start_(cudaEventDefault), stop_(cudaEventDefault), stream_(stream) {}

    void start() {
      if (defstream)
        cudaEventRecord(start_);
      else
        cudaEventRecord(start_, stream_);
    }

    void stop() {
      if (defstream)
        cudaEventRecord(stop_);
      else
        cudaEventRecord(stop_, stream_);
    }

    float elapsedMilliseconds() {
      cudaEventSynchronize(stop_);
      cudaEventElapsedTime(&millisec, start_, stop_);
      return millisec;
    }

    float elapsedSeconds() {
      if (millisec < 5e-4) {
        this->elapsedMilliseconds();
      }
      return millisec * 1e-3;
    }


   private:
    EH::cudaEvent start_;
    EH::cudaEvent stop_;
    const SH::cudaStream &stream_;
    float millisec = 0.0f;
    bool defstream = false;
  };

}  // namespace cuda_timer_helper

#endif
