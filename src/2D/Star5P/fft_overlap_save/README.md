# Star5P: fft_overlap_save

Compile `fft_2d.cu` and include `fft_2d.cuh` from this directory. The entry uses the [common implementation](../../common/fft_overlap_save/README.md). See [stencil usage](../README.md) for coefficients and fusion limits.

This is a compilation entry for the shared backend, not a distinct shape-specialized kernel. Link one entry only; multiple stencil coefficient sets can use separate plans in the same executable.
