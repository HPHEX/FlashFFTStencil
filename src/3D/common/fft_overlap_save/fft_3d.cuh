#pragma once
#include <cuda_runtime.h>

struct FlashFFT3DPlan;
// XYFFT is a retired selector retained for source compatibility only.
// CreatePlan returns cudaErrorNotSupported for it; only FullFFT is implemented.
enum class FlashFFT3DBackend { FullFFT, XYFFT };

// coeff[z][y][x], side=2*radius+1. Executes one periodic FP64 convolution.
// FullFFT transforms all three axes. Input/output contain width^3 doubles.
cudaError_t flashfft3dCreatePlan(FlashFFT3DPlan** result, const double* coeff,
                               int radius,
                               FlashFFT3DBackend backend=FlashFFT3DBackend::FullFFT);
cudaError_t flashfft3dExecute(const FlashFFT3DPlan* plan, const double* input,
                            double* output, int width, cudaStream_t stream=0);
cudaError_t flashfft3dDestroyPlan(FlashFFT3DPlan* plan);

// Split pipeline. Workspace owns periodic input tiles and tile-local outputs.
// Pack -> Compute -> Unpack must be ordered on one stream (or with events).
// Compute alone requires a completed Pack for the current input. A workspace
// cannot be shared by concurrent executions; distinct workspaces can coexist.
struct FlashFFT3DWorkspace;
cudaError_t flashfft3dCreateWorkspace(FlashFFT3DWorkspace** result,const FlashFFT3DPlan* plan,int width);
cudaError_t flashfft3dPack(const FlashFFT3DPlan* plan,FlashFFT3DWorkspace* workspace,const double* input,cudaStream_t stream=0);
cudaError_t flashfft3dCompute(const FlashFFT3DPlan* plan,FlashFFT3DWorkspace* workspace,cudaStream_t stream=0);
cudaError_t flashfft3dUnpack(const FlashFFT3DPlan* plan,const FlashFFT3DWorkspace* workspace,double* output,cudaStream_t stream=0);
// Complete all uses before destruction, on the owning CUDA device.
cudaError_t flashfft3dDestroyWorkspace(FlashFFT3DWorkspace* workspace);
