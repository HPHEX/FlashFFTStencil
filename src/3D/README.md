# 3D implementations by stencil type

Directories refer to the basic stencil before timestep fusion.

| Basic stencil | Radius | Complete FFT | Maximum fused basic steps |
|---|---:|---|---:|
| [Box27P](Box27P/README.md) | 1 | [full_fft](Box27P/full_fft/README.md) | 3 |
| [Star7P](Star7P/README.md) | 1 | [full_fft](Star7P/full_fft/README.md) | 3 |

```text
3D/
  Box27P/
    full_fft/
  Star7P/
    full_fft/
  common/
    fft_overlap_save/  # one canonical FullFFT source
  fft_overlap_save/    # compatibility for existing callers
```

`full_fft` performs FFTs along all three axes and supports both Execute and the optional split workspace pipeline. The XYFFT hybrid kernels and entries have been removed. The old XYFFT enum selector remains solely for source compatibility; CreatePlan returns `cudaErrorNotSupported` if it is requested.

Compile/link one `.cu` entry per executable. All entries include the canonical source; one implementation can serve different stencil types through independent coefficient plans.

The original `3d_main.cu`, `rfft_3d/` variants and their demo-only helpers have been removed. The old `fft_overlap_save/` path forwards to the retained optimized FullFFT implementation. See the [common backend documentation](common/fft_overlap_save/README.md) for the API and validation scope.
