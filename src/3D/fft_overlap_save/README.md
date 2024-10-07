# Compatibility path

The canonical 3D backend is now in [`../common/fft_overlap_save`](../common/fft_overlap_save/README.md). Existing `fft_3d.cu` / `fft_3d.cuh` paths remain forwarding entry points. Compile one implementation entry per executable.

See the [3D stencil directory index](../README.md) for FullFFT entries grouped by stencil type.
