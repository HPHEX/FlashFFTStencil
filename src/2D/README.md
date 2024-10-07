# 2D implementations by stencil type

Each stencil directory contains the retained register FFT implementation. The case names describe the **basic stencil**, not the larger effective filter after timestep fusion.

| Basic stencil | Radius | Register FFT | Maximum fused basic steps |
|---|---:|---|---:|
| [Box9P](Box9P/README.md) | 1 | [Box9P/fft_overlap_save](Box9P/fft_overlap_save/README.md) | 3 |
| [Star5P](Star5P/README.md) | 1 | [Star5P/fft_overlap_save](Star5P/fft_overlap_save/README.md) | 3 |
| [Box25P](Box25P/README.md) | 2 | [Box25P/fft_overlap_save](Box25P/fft_overlap_save/README.md) | 1 |
| [Star9P](Star9P/README.md) | 2 | [Star9P/fft_overlap_save](Star9P/fft_overlap_save/README.md) | 1 |
| [Box49P](Box49P/README.md) | 3 | [Box49P/fft_overlap_save](Box49P/fft_overlap_save/README.md) | 1 |
| [Star13P](Star13P/README.md) | 3 | [Star13P/fft_overlap_save](Star13P/fft_overlap_save/README.md) | 1 |

```text
2D/
  Box9P/   Star5P/
  Box25P/  Star9P/
  Box49P/  Star13P/
    fft_overlap_save/  # current register FFT compilation entry
  common/
    fft_overlap_save/  # one canonical current implementation
  fft_overlap_save/    # compatibility for existing callers
```

The stencil entries include the common implementation rather than copying kernels. Their public symbols are identical: compile/link exactly one entry per executable. One implementation supports multiple stencil types through independent coefficient plans. New genuinely shape-specific implementations can be added beneath the corresponding stencil directory.

The original demo, `rfft_2d/` variants and shared-memory FFT baseline have been removed. Unpublished direct-stencil experiments are archived outside this repository. Only the optimized register FFT remains under `src/2D/`; the old `fft_overlap_save/` path forwards to that same implementation.

See the [register FFT documentation](common/fft_overlap_save/README.md) for the measured performance and correctness scope. Directory organization does not change kernel arithmetic or previous benchmark results.
