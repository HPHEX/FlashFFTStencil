# Box25P

Basic stencil: Box, radius 2, 25 points. Box uses the full square; Star uses the center and the two coordinate axes. Coefficients are caller-provided centered correlation weights; no coefficient values are hard-coded by this directory.

| Implementation | Entry | API | Status |
|---|---|---|---|
| Register FP64 FFT | [`fft_overlap_save/fft_2d.cu`](fft_overlap_save/fft_2d.cu) | CreatePlan / Execute / Pack / Compute / Unpack | Current implementation |

The retained implementation performs full X/Y FFT overlap-save convolution. Compile one stencil entry per executable. The common kernel source is under `../common/`, so fixes apply to all stencil entries.

For one basic step, supply a 5×5 coefficient matrix and effective radius 2. To fuse steps, convolve the basic coefficients on the host and pass their effective matrix with radius `basic_radius × steps`. The backend supports at most 1 fused basic step(s) for this stencil (effective radius ≤3). A fused Star stencil acquires off-axis coefficients; do not discard them.

Example build, from the repository root:

```sh
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_80 your_program.cu \
  src/2D/Box25P/fft_overlap_save/fft_2d.cu -o your_program
```

Include `src/2D/Box25P/fft_overlap_save/fft_2d.cuh` in the caller. The grid is contiguous FP64, square, periodic, and width ≥64. Use the common backend README for stream ordering, buffer ownership, correctness limits and measured performance.
