# Star7P

Basic stencil: radius 1, 7 points. Star uses the center and the six axis neighbors. Coefficients are supplied by the caller; this directory does not hard-code their values.

| Implementation | Entry | Plan helper | Split API |
|---|---|---|---|
| Complete X/Y/Z FFT | [`full_fft/fft_3d.cu`](full_fft/fft_3d.cu) | `flashfft3d_star7p::createFullFFTPlan` | Supported |

The retained FullFFT entry uses the common backend source under `../common/fft_overlap_save/`. Link one stencil entry per executable. The generic CreatePlan default is FullFFT.

For one basic step, provide a 3×3×3 coefficient array and effective radius 1. Up to three basic steps can be fused by convolving their coefficients on the host; pass the resulting matrix with effective radius 2 or 3. A three-step Box has 343 supported positions and a three-step Star has 63; fused Star coefficients include off-axis positions and must not be discarded.

Example build, from the repository root:

```sh
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_80 -rdc=true \
  application.cu src/3D/Star7P/full_fft/fft_3d.cu -o application
```

Include `src/3D/Star7P/full_fft/fft_3d.cuh` and use the FullFFT helper. Input/output are distinct contiguous FP64 cubes, with X contiguous, periodic boundaries, and width ≥64. See the [common README](../common/fft_overlap_save/README.md) for coefficient indexing, stream ordering, measured performance and correctness limits.
