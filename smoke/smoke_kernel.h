// Shared smoke-test kernel source, compiled TWICE — once per target arch —
// so ONE binary contains kernels for both the RX 9070 XT (gfx1201) and the
// RTX 5090 (sm_120), running the ROCm and CUDA runtimes side by side in one
// process (M0 dual-runtime smoke, plan todo 4). Todo 7's real
// SieveSlabKernel extends this exact pattern.
//
// The two compile paths (see Makefile) are:
//   AMD:    /opt/rocm/bin/hipcc -I/opt/rocm/include -DSIEVE_KERNEL_ARCH=gfx1201
//   NVIDIA: HIP_PLATFORM=nvidia /opt/rocm/bin/hipcc -x cu -arch=sm_120 \
//             -I/opt/rocm/include -DSIEVE_KERNEL_ARCH=sm_120
//
// Per-arch kernel symbol rename (Metis BLOCKER #3): the SAME __global__ name
// in both objects would duplicate the symbol at link time. The arch tag is
// pasted into the kernel name via a TWO-LEVEL macro expansion — a single-
// level paste yields the literal `SieveSlab_SIEVE_KERNEL_ARCH` (Oracle
// round-3 toolchain finding). SIEVE_KERNEL_ARCH is supplied per-compile with
// -D, so this one header emits SieveSlab_gfx1201 for the AMD object and
// SieveSlab_sm120 for the NVIDIA object.

#ifndef FF_SMOKE_KERNEL_H
#define FF_SMOKE_KERNEL_H

#define FF_KERN_CAT2(a, b) a##b
#define FF_KERN_CAT(a, b)  FF_KERN_CAT2(a, b)

#ifndef SIEVE_KERNEL_ARCH
#error "SIEVE_KERNEL_ARCH must be defined per compile (gfx1201 or sm_120)"
#endif

#define SIEVE_SMOKE_KERNEL FF_KERN_CAT(SieveSlab_, SIEVE_KERNEL_ARCH)

// TRIVIAL placeholder body — todo 7 replaces this with the real
// SieveSlabKernel body (byte i>>4, bit i>>1&7 sieve) behind the same macro.
__global__ void SIEVE_SMOKE_KERNEL(unsigned int* buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 1u;
}

#endif  // FF_SMOKE_KERNEL_H
