#pragma once

#define CUDAKERNELCHECK(expr)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        expr;                                                                                                          \
                                                                                                                       \
        cudaError_t __err = cudaGetLastError();                                                                        \
        if (__err != cudaSuccess)                                                                                      \
        {                                                                                                              \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err));                          \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#include <stdio.h>

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)