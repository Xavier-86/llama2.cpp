// Zhenwu 810E PPU int8 inference: reusable activation quantization and merged
// Q/K/V and W1/W3 projections.
#define RUNQGPU_KERNEL_KIND KernelKind::Ppu
#include "../default/runqgpu_impl.cuh"
