# Compatibility path

The canonical register FFT backend is now in [`../common/fft_overlap_save`](../common/fft_overlap_save/README.md). Existing `fft_2d.cu` / `fft_2d.cuh` paths remain forwarding entry points. Compile either the old entry or a new entry once, not both.

See the [2D stencil directory index](../README.md) for stencil-specific implementation entries.
