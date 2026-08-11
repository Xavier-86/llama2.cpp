// Default int8 GPU inference: separate activation quantization and
// one block per output row.
#define RUNQGPU_KERNEL_KIND KernelKind::Naive
#include "runqgpu_impl.cuh"
