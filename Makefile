# ff-gpu Makefile — dual-arch HIP build (plan todo 4: toolchain sanity +
# dual-runtime smoke).
#
# One HIP/CUDA-compatible kernel source (smoke/smoke_kernel.h) is compiled
# TWICE, once per target arch:
#   - AMD (RX 9070 XT, gfx1201):    hipcc with the DEFAULT platform
#   - NVIDIA (RTX 5090, sm_120):    HIP_PLATFORM=nvidia hipcc -x cu -arch=sm_120
#                                   (delegates to /usr/local/cuda/bin/nvcc,
#                                    CUDA 13.3)
# Both device objects link into ONE binary that runs the CUDA and ROCm
# runtimes side by side in one process. Vendor TUs are split
# (smoke/cuda_smoke.cpp / smoke/hip_smoke.cpp) so no TU ever mixes vendor
# headers. Objects-only link: NEVER pass a .cu source on the link line —
# hipcc's nvidia backend applies `-x cu` to ALL inputs when any .cu is
# present, misparsing compiled .o files (Oracle round-3 toolchain finding).

# --- toolchain: full paths, neither compiler is on PATH ---
HIPCC := /opt/rocm/bin/hipcc
GXX   := g++
# nvcc is not invoked directly here: hipcc's nvidia backend locates CUDA 13.3
# and delegates to /usr/local/cuda/bin/nvcc.

# --- include / library search paths ---
ROCM_INC := -I/opt/rocm/include   # hipcc does NOT auto-add this for this
                                  # ROCm build — every compile needs it
CUDA_LIB := -L/usr/local/cuda/lib64   # NVIDIA object  -> -lcudart
ROCM_LIB := -L/opt/rocm/lib           # AMD object    -> -lamdhip64

# --- per-arch targets ---
ARCH_AMD := gfx1201   # RX 9070 XT (RDNA4)
ARCH_NV  := sm_120    # RTX 5090 (consumer Blackwell; needs CUDA 12.8+, we
                      # have 13.3 — sm_120a/sm_120f are NOT used)

# --- common host flags ---
HOST_CXXFLAGS := -O2 -Wall

# -DSIEVE_KERNEL_ARCH=<arch> drives the per-arch kernel symbol rename: the
# shared kernel header pastes the arch into the kernel name
# (SieveSlab_gfx1201 / SieveSlab_sm120) via a two-level macro, so the two
# objects never export the same __global__ symbol (Metis BLOCKER #3).
# NOTE: -Wall is valid for the clang-based AMD backend but NOT for nvcc
# (the NVIDIA backend rejects it: "nvcc fatal: Unknown option '-Wall'"),
# so NVFLAGS carries -O2 only.
AMDFLAGS := $(HOST_CXXFLAGS) $(ROCM_INC) -DSIEVE_KERNEL_ARCH=$(ARCH_AMD)
NVFLAGS  := -O2 $(ROCM_INC) -DSIEVE_KERNEL_ARCH=$(ARCH_NV)

BUILD := build
OBJS  := $(BUILD)/smoke_main.o $(BUILD)/hip_smoke.o $(BUILD)/cuda_smoke.o

# Pinned output path: every acceptance invocation in later todos uses
# ./ff_sieve from the repo root (bash does not search the cwd for bare
# command names — Momus round-7 NIT).
BIN := ff_sieve

.PHONY: all smoke clean

all: $(BIN)

# ---- AMD TU: DEFAULT HIP platform (hipcc as-is) ----
$(BUILD)/hip_smoke.o: smoke/hip_smoke.cpp smoke/smoke_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

# ---- NVIDIA TU: HIP_PLATFORM=nvidia -> hipcc delegates to nvcc ----
$(BUILD)/cuda_smoke.o: smoke/cuda_smoke.cpp smoke/smoke_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# ---- vendor-neutral host main (g++) ----
$(BUILD)/smoke_main.o: smoke/smoke_main.cpp | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

# ---- link: objects ONLY, never a .cu source on the link line ----
$(BIN): $(OBJS)
	$(GXX) -o $@ $(OBJS) $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

$(BUILD):
	mkdir -p $(BUILD)

smoke: all
	./$(BIN)

clean:
	rm -rf $(BUILD) $(BIN)
