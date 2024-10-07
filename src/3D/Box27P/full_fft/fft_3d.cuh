#pragma once
#include "../../common/fft_overlap_save/fft_3d.cuh"

namespace flashfft3d_box27p {
// Coefficients are caller-provided; radius is the effective radius after fusion.
inline cudaError_t createFullFFTPlan(FlashFFT3DPlan** result,
                                     const double* coeff,int radius) {
    return flashfft3dCreatePlan(result,coeff,radius,FlashFFT3DBackend::FullFFT);
}
} // namespace flashfft3d_box27p
