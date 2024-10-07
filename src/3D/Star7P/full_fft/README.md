# Star7P: FullFFT

Transforms X, Y and Z, then multiplies the complete filter spectrum and applies inverse FFTs. Supports Execute and the Pack / Compute / Unpack workspace API.

Include `fft_3d.cuh`, compile `fft_3d.cu`, and create the plan using:

```cpp
FlashFFT3DPlan* plan = nullptr;
cudaError_t status = flashfft3d_star7p::createFullFFTPlan(&plan, coefficients, effective_radius);
// Check status before using plan.
```

Use the common `flashfft3dExecute` / `flashfft3dDestroyPlan` functions after creation. FullFFT is the only retained 3D backend; the generic CreatePlan default also selects it.

The compilation entry includes the [common implementation](../../common/fft_overlap_save/README.md). Link one stencil `.cu` entry per executable. A single implementation can run independent plans for different shapes and coefficient sets. See [stencil usage](../README.md) for timestep fusion and coefficient support.
