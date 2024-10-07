# FP64 3D FFT overlap-save backend

This backend applies a periodic convolution to a contiguous cubic FP64 grid. The retained `FullFFT` backend transforms **all three axes**, multiplies a precomputed filter spectrum, performs the inverse transforms, and stores only the valid overlap-save region. It uses a custom 32×32×16 real/complex FFT, warp butterflies, and block-local shared memory. Execution requires one GPU kernel, with no global packing, output clearing, or atomic accumulation.

The optimization targets Box 27P and Star 7P. Three successive radius-one steps can be fused into one radius-three convolution: Box has 343 nonzero coefficients and Star has 63. The caller supplies the fused coefficients; this API does not perform timestep fusion itself. Arbitrary finite coefficients, including asymmetric coefficients, are supported.

Only FullFFT is retained. The XYFFT hybrid implementation has been removed; its enum selector remains for source compatibility and CreatePlan rejects it with `cudaErrorNotSupported`.

## API

Include `fft_3d.cuh` and compile/link `fft_3d.cu` with CUDA C++17, `-arch=sm_80`, and `-rdc=true`. For example:

```sh
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_80 -rdc=true \
  application.cu fft_3d.cu -o application
```

```cpp
FlashFFT3DPlan* plan = nullptr;
// Host coefficients indexed coeff[(z * side + y) * side + x].
// Offsets run from -radius through +radius, side = 2 * radius + 1.
cudaError_t status = flashfft3dCreatePlan(&plan, coeff, radius);
if (status != cudaSuccess) return status;
status = flashfft3dExecute(plan, device_input, device_output, width, stream);
if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cudaError_t cleanup = flashfft3dDestroyPlan(plan);
return status != cudaSuccess ? status : cleanup;
```

The operation is `out[z,y,x] = sum coeff[dz,dy,dx] * in[z+dz,y+dy,x+dx]`, with periodic wrapping. Effective radii 1–3 and widths at least 64 are supported, including widths that do not divide the tile size. Input/output arrays each contain `width³` doubles, with X contiguous; their memory regions must not overlap. Execution is asynchronous on the supplied stream. Synchronize all uses before destroying a plan. Plans own their filter spectra and can coexist with different coefficients on different streams. Create, execute, and destroy on the same CUDA device.

The current implementation explicitly supports SM80 devices and was validated on A100-SXM4-80GB with CUDA 12.1. The legacy 3D demo has been removed; use the library API or a stencil-specific FullFFT entry.

## Validation and timing

The original validation suite covered FullFFT and the now-removed XYFFT backend, Box/Star, one to three fused basic steps, four successive fused groups, random/signed/constant/impulse input, asymmetric coefficients, and widths 64, 65, 128, 129, and 192. It compares full outputs with an independent FP64 reference and checks sampled CPU results. Small cases also compare against explicitly executed basic timesteps. ConvStencil is compared on its supported grid sizes and Z-symmetric coefficients.

Performance comparisons use the same fused three-step operator, device-resident FP64 inputs, CUDA events, warmups, nine timing batches per trial, and five independent trials. Plan construction is excluded. ConvStencil's raw stencil kernel and its periodic pack + stencil + extraction pipeline are reported separately. Benchmark scripts and raw results live in the accompanying `optimization3d` evaluation directory; final measured results are documented there.

On A100-SXM4-80GB, five-trial medians for fused three-step execution were:

| Stencil | Grid | FullFFT (ms) | ConvStencil kernel (ms) | Kernel speedup |
|---|---:|---:|---:|---:|
| Box 27P | 384³ | 4.74890 | 7.80334 | 1.643× |
| Box 27P | 768³ | 37.23367 | 58.00013 | 1.558× |
| Star 7P | 384³ | 4.74691 | 5.17760 | 1.091× |
| Star 7P | 768³ | 37.21487 | 38.29622 | 1.029× |

FullFFT also beat the raw ConvStencil kernel at 64³, 128³, and 192³ for both shapes. The Star 768³ margin is small; these measurements do not establish a universal performance guarantee. The new backend passed 76 numerical cases (440 comparisons, maximum absolute error 1.95e-14), 12 multi-plan/stream API checks, and four CUDA sanitizer checks with zero errors.

## Split input / compute / output pipeline

The optional split API exposes preprocessing separately from the FFT computation:

```cpp
FlashFFT3DWorkspace* workspace = nullptr;
// Create the plan first. Allocate once, outside timing.
cudaError_t status = flashfft3dCreateWorkspace(&workspace, plan, width);
if (status != cudaSuccess) return status;
status = flashfft3dPack(plan, workspace, device_input, stream);
if (status == cudaSuccess) status = flashfft3dCompute(plan, workspace, stream);
if (status == cudaSuccess) status = flashfft3dUnpack(plan, workspace, device_output, stream);
if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cudaError_t cleanup = flashfft3dDestroyWorkspace(workspace);
return status != cudaSuccess ? status : cleanup;
```

`Pack` constructs periodic overlapping tiles. `Compute` performs the same forward FFT, filter multiplication, inverse FFT, and valid-region computation as the fused backend, reading/writing workspace buffers. `Unpack` extracts the result into a contiguous periodic grid. Compute-only timing requires previously packed current input; pipeline timing must include all three stages. Each call launches one kernel. No allocations occur in these three execution calls. Fixed filter-spectrum construction remains outside timing in both versions.

Workspace width is fixed at creation; radius and owning device must match the plan. A workspace can be reused with another plan having the same radius/device. Call the stages in order on one stream, or establish cross-stream dependencies with CUDA events. Complete all uses before destroying the workspace. Concurrent operations must use distinct workspaces. Calling Compute before Pack, or Unpack before Compute, produces undefined data. Repack after each input change, including each dependent timestep group. The existing `Execute` API remains the fused path and requires no workspace.

The split API supports the retained FullFFT implementation. Each block has a 32×32×16 real input tile and a compact `(32-2R)²*(16-2R)` output tile. With `s=32-2R`, `z=16-2R`, `blocks=ceil(width/s)²*ceil(width/z)`, workspace bytes are `blocks*(16384+s*s*z)*8`. At R=3, 768³ requires 12,831,033,600 bytes (about 11.95 GiB) of additional device memory.

Five-trial A100 FP64 measurements for 768³ fused-three-step operators:

| Stencil | Fused Execute (ms) | Split Compute (ms) | Split pipeline (ms) | Conv kernel (ms) | Conv pipeline (ms) |
|---|---:|---:|---:|---:|---:|
| Box 27P | 37.21846 | 35.24828 | 51.49287 | 57.88472 | 67.35262 |
| Star 7P | 37.22767 | 35.25048 | 51.49614 | 38.35494 | 47.83539 |

Compute-only is faster than fused Execute and ConvStencil for these cases. Additional packing/output traffic makes the split pipeline slower than fused Execute; Star split pipeline also loses to the ConvStencil pipeline. Prefer fused Execute for contiguous-grid production execution.

The split implementation passed 64 numerical benchmark cases across both dimensions, 24 multi-plan/stream API checks, and all eight dimension-specific CUDA sanitizer checks. Benchmark scripts and detailed raw results are in the accompanying `split_optimization` evaluation directory.

## Retained-only cleanup validation

After removing the original demos and XYFFT code, MTU A100 job 69133 compiled all six 2D stencil entries, both 3D FullFFT entries and both compatibility entries. The 2D and 3D APIs each passed 12 independent FP64 reference comparisons at widths 64 and 65, effective radii 1–3, with asymmetric signed coefficients, multiple plans and two streams. Execute and the split pipeline agreed; the retired XYFFT selector was rejected as documented. The retained 3D code passed memcheck, racecheck, initcheck and synccheck with zero errors. The 2D kernel files and the 3D FullFFT kernel body were unchanged by this cleanup; the performance tables above remain the earlier measurements. Raw logs and the removed-source backup are outside the repository in the accompanying `cleanup_fft_only` directory.
