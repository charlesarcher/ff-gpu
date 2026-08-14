// M0 benchmark kernels — compiled TWICE (AMD + NVIDIA), same pattern as
// smoke_kernel.h. Two kernels:
//   M0_MEMSET_<arch>:  fill buffer with constant 0xDEADBEEF (pure write bandwidth)
//   M0_BW_SEQ_<arch>:  sequential write with index-based pattern (no RMW)
//
// Both kernels write 4 bytes per element, measuring raw memory write bandwidth.
// Uses SIEVE_KERNEL_ARCH (set via -D in the Makefile) to drive the per-arch
// kernel symbol rename, sharing the same compile-time mechanism as smoke_kernel.h.

#ifndef FF_M0_KERNEL_H
#define FF_M0_KERNEL_H

#define M0_KERN_CAT2(a, b) a##b
#define M0_KERN_CAT(a, b)  M0_KERN_CAT2(a, b)

// Reuse SIEVE_KERNEL_ARCH from the build system (gfx1201 or sm_120).
#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

#define M0_MEMSET_KERN  M0_KERN_CAT(M0_MEMSET_, SIEVE_KERNEL_ARCH)
#define M0_BW_SEQ_KERN  M0_KERN_CAT(M0_BW_SEQ_, SIEVE_KERNEL_ARCH)

// Fill buffer with constant 0xDEADBEEF (pure write bandwidth, no read dependency)
__global__ void M0_MEMSET_KERN(unsigned int* buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0xDEADBEEFu;
}

// Sequential write with index-based pattern (guarantees no coalescing tricks,
// each thread writes a unique value to a unique address — true sequential BW).
__global__ void M0_BW_SEQ_KERN(unsigned int* buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (unsigned int)i * 0x01010101u;
}

#endif  // FF_M0_KERNEL_H