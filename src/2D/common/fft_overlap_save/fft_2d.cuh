#pragma once
#include <cuda_runtime.h>
struct FlashFFT2DPlan;
// Host row-major centered correlation coefficients, effective radius 1..3.
// The plan owns its device filter spectrum. Call on the device that will execute it.
cudaError_t flashfft2dCreatePlan(FlashFFT2DPlan** plan,const double* coefficients,int radius);
// FP64 contiguous periodic square grid, width >= 64. Buffers must be disjoint.
// Stream ordered, one FFT overlap-save kernel; no per-call allocation or workspace.
cudaError_t flashfft2dExecute(const FlashFFT2DPlan* plan,const double* input,double* output,int width,cudaStream_t stream=0);
// Complete all executions using this plan before destroying it, on its owning device.
cudaError_t flashfft2dDestroyPlan(FlashFFT2DPlan* plan);

// Split pipeline. Workspace owns periodic input tiles and tile-local outputs.
// Pack -> Compute -> Unpack must be ordered on one stream (or with events).
// Compute alone requires a completed Pack for the current input. A workspace
// cannot be shared by concurrent executions; distinct workspaces can coexist.
struct FlashFFT2DWorkspace;
cudaError_t flashfft2dCreateWorkspace(FlashFFT2DWorkspace** result,const FlashFFT2DPlan* plan,int width);
cudaError_t flashfft2dPack(const FlashFFT2DPlan* plan,FlashFFT2DWorkspace* workspace,const double* input,cudaStream_t stream=0);
cudaError_t flashfft2dCompute(const FlashFFT2DPlan* plan,FlashFFT2DWorkspace* workspace,cudaStream_t stream=0);
cudaError_t flashfft2dUnpack(const FlashFFT2DPlan* plan,const FlashFFT2DWorkspace* workspace,double* output,cudaStream_t stream=0);
// Complete all uses before destruction, on the owning CUDA device.
cudaError_t flashfft2dDestroyWorkspace(FlashFFT2DWorkspace* workspace);
