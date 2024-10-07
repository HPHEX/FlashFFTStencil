# FP64 FFT overlap-save 2D backend

This backend executes stencil convolution through a 32×32 complex FFT, a filter-spectrum multiplication, and an inverse FFT. Each one-dimensional FFT uses an 8-point FP64 Tensor Core DFT followed by 4-point butterflies. It supports arbitrary centered correlation coefficients of effective radius 1–3, contiguous square FP64 arrays, periodic boundaries, and widths ≥64. The current executable target is compute capability 8.0; performance was tuned on an A100.

There is no global tile-packing buffer, separate packing kernel, output clearing, or `atomicAdd`. Inside the kernel, two real spatial tiles share the real and imaginary parts of one complex FFT. This is an internal encoding; the caller provides ordinary contiguous arrays. Every output point has one writer.

For effective radius R, a 32×32 tile keeps a (32−2R)² valid region. The kernel loads periodic halo values, discards the contaminated inverse-FFT border, and writes the valid region directly. The register FFT kernel loads the FP64 Tensor Core A fragments directly from global input, keeps the DFT accumulators and 4-point butterflies in registers, and uses a row-dependent permutation in shared memory for transposes. It uses three block barriers; warp barriers protect in-place row transforms. Even-width fused outputs use aligned 16-byte stores; odd widths use scalar stores. `Execute` and `Compute` use specializations of the same kernel. Filter frequencies use natural order. The implementation is in `register_fft.cuh`.

## Build and use

```sh
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_80 -rdc=true your_program.cu \
  src/2D/common/fft_overlap_save/fft_2d.cu -o your_program
```

```cpp
#include "src/2D/common/fft_overlap_save/fft_2d.cuh"
FlashFFT2DPlan* plan = nullptr;
// coefficients: host row-major (2*R+1)×(2*R+1), indexed by (dy+R,dx+R)
auto error = flashfft2dCreatePlan(&plan, coefficients, R);
// Check error; d_input and d_output are disjoint device FP64 arrays.
error = flashfft2dExecute(plan, d_input, d_output, width, stream);
// Check launch error, then synchronize stream to observe execution errors.
cudaStreamSynchronize(stream);
flashfft2dDestroyPlan(plan);
```

The mathematical operation is
`output[y,x] = Σ coefficients[dy+R,dx+R] * input[(y+dy) mod width,(x+dx) mod width]`.
The filter is reversed for convolution, and the unnormalized inverse FFT is compensated by 1/1024 in the filter spectrum. Plan creation precomputes the spectrum on the host and uploads it; it is excluded from steady-state execution timing.

Each plan owns its filter spectrum, so different coefficient sets can coexist. Create, execute, and destroy a plan on its owning device. Complete all work using a plan before destruction. The caller owns input/output arrays and any buffers used to alternate successive iterations.

A single execution applies the supplied effective coefficient matrix once. To fuse several basic time steps, supply their convolved coefficients, with the effective radius still ≤3. For example, three radius-1 Box steps use a radius-3, 7×7 filter. Repeated executions can alternate two output buffers.

The code preserves FP64 arithmetic. FFT rounding differs from direct convolution; the comparison uses `abs(error) ≤ 1e-9 + 1e-9*abs(reference)` and also reports maximum absolute error. The earlier shared-memory implementation was validated on an A100 with 86 cases, including asymmetric/signed coefficients, periodic boundaries, and four consecutive execution groups. The backend passed CUDA memcheck, racecheck, initcheck, and synccheck. These tests cover the measured cases; they do not establish correctness for every possible input.

## Split input / compute / output pipeline

The optional split API exposes preprocessing separately from the FFT computation:

```cpp
FlashFFT2DWorkspace* workspace = nullptr;
// Create the plan first. Allocate once, outside timing.
cudaError_t status = flashfft2dCreateWorkspace(&workspace, plan, width);
if (status != cudaSuccess) return status;
status = flashfft2dPack(plan, workspace, device_input, stream);
if (status == cudaSuccess) status = flashfft2dCompute(plan, workspace, stream);
if (status == cudaSuccess) status = flashfft2dUnpack(plan, workspace, device_output, stream);
if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
cudaError_t cleanup = flashfft2dDestroyWorkspace(workspace);
return status != cudaSuccess ? status : cleanup;
```

`Pack` constructs periodic overlapping tiles. `Compute` performs the same forward FFT, filter multiplication, inverse FFT, and valid-region computation as the fused backend, reading/writing workspace buffers. `Unpack` extracts the result into a contiguous periodic grid. Compute-only timing requires previously packed current input; pipeline timing must include all three stages. Each call launches one kernel. No allocations occur in these three execution calls. Fixed filter-spectrum construction remains outside timing in both versions.

Workspace width is fixed at creation; radius and owning device must match the plan. A workspace can be reused with another plan having the same radius/device. Call the stages in order on one stream, or establish cross-stream dependencies with CUDA events. Complete all uses before destroying the workspace. Concurrent operations must use distinct workspaces. Calling Compute before Pack, or Unpack before Compute, produces undefined data. Repack after each input change, including each dependent timestep group. The existing `Execute` API remains the fused path and requires no workspace.

Each block pairs two 32×32 real tiles in one complex array. With `s=32-2R`, `t=ceil(width/s)`, and `blocks=t*ceil(t/2)`, the workspace uses `blocks*32768` bytes for input and output combined. At R=3, this is 233,963,520 bytes for 3072² and 924,155,904 bytes for 6144². Tile-output padding is not consumed by Unpack.

Five-trial split-pipeline measurements on A100, FP64, periodic 3072² Box 49P: fused Execute 0.33050 ms, Compute 0.32317 ms, split pipeline 0.53376 ms, ConvStencil kernel 0.15903 ms, ConvStencil pipeline 0.34678 ms. These measurements describe the earlier shared-memory implementation, before the register FFT optimization below. Prefer fused Execute for contiguous-grid production execution.

The split implementation passed 64 numerical benchmark cases across both dimensions, 24 multi-plan/stream API checks, and all eight dimension-specific CUDA sanitizer checks. Benchmark scripts and detailed raw results are in the accompanying `split_optimization` evaluation directory.

## Register FFT optimization: Box 9P / Star 5P

The fused and split-compute paths now use the register FFT kernel. Both axes still use the full FP64 FFT algorithm; no spatial direct-stencil or X-FFT/Y-FIR substitution is selected. Arbitrary coefficients remain supported. Constant filter preparation stays outside timing.

Five-trial measurements on MTU A100-SXM4-80GB, periodic grids, with **three basic steps fused in both implementations**:

| Case | Width | Previous Execute ms | Register Execute ms | Compute ms | Conv kernel ms | Conv GPU pipeline ms | Execute GStencils/s | Conv kernel GStencils/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Box 9P | 3072 | 0.329011 | 0.198144 | 0.216115 | 0.157696 | 0.345549 | 142.88 | 179.53 |
| Box 9P | 6144 | 1.264282 | 0.760166 | 0.819200 | 0.598323 | 1.330483 | 148.98 | 189.27 |
| Star 5P | 3072 | 0.329011 | 0.198144 | 0.216115 | 0.157645 | 0.344986 | 142.88 | 179.59 |
| Star 5P | 6144 | 1.264435 | 0.760166 | 0.819354 | 0.598170 | 1.330432 | 148.98 | 189.32 |

`GStencils/s = width² × 3 / (milliseconds × 10⁶)`. The register implementation improves Execute by about 1.66× and beats the Conv GPU layout/kernel/extraction pipeline by about 1.74–1.75× for these large grids. It **does not yet beat the Conv single kernel**. Compute performs a complete FFT on workspace buffers; its extra global traffic makes it slower than Execute at these sizes. Split-pipeline timing must include Pack and Unpack. All metrics exclude one-time allocations and filter/lookup preparation.

The final implementation passed 46 numerical benchmark cases, 12 multi-plan/two-stream API comparisons, all four CUDA sanitizer tools, and additional large-grid validation accompanying 40 performance trials. Tested conditions include odd widths, asymmetric/signed coefficients, signed input, impulses, constants, and 12 dependent basic steps. Maximum absolute error across the final correctness and performance logs was 5.40013e-12; the existing 1e-9 absolute plus 1e-9 relative threshold was unchanged. Detailed measurements, source hashes, scripts and raw logs are in the accompanying `optimization2d_special` evaluation directory. Numerical FFT results need not be bitwise equal to direct convolution.
