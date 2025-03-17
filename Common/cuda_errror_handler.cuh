#ifndef CUDA_ERROR_HANDLER_H
#define CUDA_ERROR_HANDLER_H

#pragma once

#include <stdio.h>
#include <cuda.h>

template <typename T>
inline void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error :" " %s : (%d) %s.\n", 
                file, line, errorMessage, static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERR(val) check((val), #val, __FILE__, __LINE__)

#define CHECK_CUDA_LASTERR(msg) __getLastCudaError(msg, __FILE__, __LINE__)

#endif
