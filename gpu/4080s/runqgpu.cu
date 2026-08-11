// RTX 4080 SUPER-specific int8 inference: fused warp-per-row GEMV.
#define RUNQGPU_KERNEL_KIND KernelKind::Fused
#include "../default/runqgpu_impl.cuh"
