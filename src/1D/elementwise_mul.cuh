#pragma once


__device__ __forceinline__ void complexMul(const double a_real, const double a_imag, const double b_real,
                                           const double b_imag, double *result_real, double *result_imag)
{
    *result_real = (a_real * b_real - a_imag * b_imag);
    *result_imag = (a_real * b_imag + a_imag * b_real);
}
