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
OBJS  := $(BUILD)/main.o $(BUILD)/config.o $(BUILD)/budget.o \
         $(BUILD)/geometry.o $(BUILD)/device_registry.o \
         $(BUILD)/devabstraction.o \
         $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
         $(BUILD)/smoke_main.o $(BUILD)/hip_smoke.o $(BUILD)/cuda_smoke.o \
         $(BUILD)/sieve_engine.o $(BUILD)/gpu_prime.o $(BUILD)/cpu_search.o \
         $(BUILD)/pull_scheduler.o \
         $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
         $(BUILD)/m4_host_amd.o $(BUILD)/m4_host_nv.o \
         $(BUILD)/gpu_search_launcher.o

# SieveSlab kernel+host objects (todo 7: GPU PLAN §5).
# sieve_slab_kernel.cpp includes sieve_slab_kernel.h so the kernel body is
# emitted into each host .o.  Per-arch -DSIEVE_KERNEL_ARCH produces
# SieveSlab_gfx1201 / SieveSlab_sm120 symbols, and the host function is also
# arch-tagged (SieveSlabRun_gfx1201 / SieveSlabRun_sm120) to avoid duplicate
# symbols at link time (Metis BLOCKER #3).
SLAB_OBJS_AMD := $(BUILD)/sieve_slab_host_amd.o
SLAB_OBJS_NV  := $(BUILD)/sieve_slab_host_nv.o
SLAB_ALL_OBJS := $(SLAB_OBJS_AMD) $(SLAB_OBJS_NV)

# SieveSlab full-map engine objects (todo 8: GPU PLAN §5 engine).
# sieve_slab_engine.cpp includes sieve_slab_kernel.h so the kernel body is
# emitted into each host .o.  Per-arch -DSIEVE_KERNEL_ARCH produces
# arch-tagged symbols (SieveSlabEngineRun_gfx1201 / SieveSlabEngineRun_sm_120)
# via the same two-level macro-paste mechanism.
ENGINE_OBJS_AMD := $(BUILD)/sieve_slab_engine_amd.o
ENGINE_OBJS_NV  := $(BUILD)/sieve_slab_engine_nv.o
ENGINE_ALL_OBJS := $(ENGINE_OBJS_AMD) $(ENGINE_OBJS_NV)

# Pinned output path: every acceptance invocation in later todos uses
# ./ff_sieve from the repo root (bash does not search the cwd for bare
# command names — Momus round-7 NIT).
BIN := ff_sieve

# Pure-logic self-test for the todo-3 modules (g++ only, no vendor headers).
SELFTEST := tests/ff_budget_selftest

.PHONY: all smoke selftest abstraction-smoke slab-cmp bench m4 m4-order clean

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

# ---- todo 3: startup device enumeration (split vendor TUs, never mixed) ----
# AMD enumeration TU: DEFAULT HIP platform (hipcc as-is) — sees the 9070 XT.
$(BUILD)/hip_enum.o: src/hip_enum.cpp src/device_info.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

# NVIDIA enumeration TU: HIP_PLATFORM=nvidia (same path as cuda_smoke.o) —
# sees the 5090 only. Includes cuda_runtime.h, so -Wall stays out (NVFLAGS).
$(BUILD)/cuda_enum.o: src/cuda_enum.cpp src/device_info.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# ---- todo 6: vendor-neutral DevAbstraction (GPU_PLAN §5.2) ----
# Host TU (g++, NO vendor headers): logical device list via ff::mergeAndDedupe
# (bus-ID dedup) + dispatch to the split backend TUs below.
$(BUILD)/devabstraction.o: src/devabstraction.cpp src/devabstraction.h \
                           src/device_registry.h src/device_info.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

# AMD backend TU: DEFAULT HIP platform (hipcc as-is) — sees the 9070 XT only.
$(BUILD)/hip_devabstraction.o: src/hip_devabstraction.cpp \
                               src/devabstraction.h src/device_info.h \
                               smoke/smoke_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

# NVIDIA backend TU: HIP_PLATFORM=nvidia (nvcc path) — sees the 5090 only.
# Includes cuda_runtime.h, so -Wall stays out (NVFLAGS).
$(BUILD)/cuda_devabstraction.o: src/cuda_devabstraction.cpp \
                                src/devabstraction.h src/device_info.h \
                                smoke/smoke_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# Abstraction smoke test (own main(), NOT linked into ff_sieve): links the
# host TU + both backend TUs + the todo-3 enumeration/registry objects. The
# dual-runtime link line is objects-only (never a .cu source).
SMOKE_TEST := tests/abstraction_smoke
$(SMOKE_TEST): tests/abstraction_smoke.cpp src/devabstraction.h \
               $(BUILD)/devabstraction.o $(BUILD)/hip_devabstraction.o \
               $(BUILD)/cuda_devabstraction.o $(BUILD)/device_registry.o \
               $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ tests/abstraction_smoke.cpp \
	    $(BUILD)/devabstraction.o $(BUILD)/hip_devabstraction.o \
	    $(BUILD)/cuda_devabstraction.o $(BUILD)/device_registry.o \
	    $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

abstraction-smoke: $(SMOKE_TEST)
	./$(SMOKE_TEST)

# ---- todo 7: SieveSlabKernel (GPU_PLAN §5) ----
# sieve_slab_kernel.cpp includes the kernel header, so the kernel + host
# helper are emitted in the same .o.  Compiled per-arch with arch-tagged
# symbols so both AMD and NVIDIA objects coexist in one binary.
$(BUILD)/sieve_slab_host_amd.o: src/sieve_slab_kernel.cpp src/sieve_slab_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

$(BUILD)/sieve_slab_host_nv.o: src/sieve_slab_kernel.cpp src/sieve_slab_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# SieveSlab full-map engine: per-arch host+kernel (same pattern as sieve_slab_kernel).
$(BUILD)/sieve_slab_engine_amd.o: src/sieve_slab_engine.cpp src/sieve_slab_engine.h \
                                   src/sieve_slab_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

$(BUILD)/sieve_slab_engine_nv.o: src/sieve_slab_engine.cpp src/sieve_slab_engine.h \
                                  src/sieve_slab_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# slab_cmp test binary (g++ host, links per-arch host+kernel objects + runtime).
# STANDALONE test — NOT linked into ff_sieve.  Textually includes the
# untouched reference SegmentFill (via SEGMENT_FILL_RENAME macro) and drives
# byte-exact cmp vs the GPU kernel.
SLAB_CMP := tests/slab_cmp
$(SLAB_CMP): tests/slab_cmp.cpp src/sieve_slab_kernel.h \
             $(SLAB_ALL_OBJS) | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ tests/slab_cmp.cpp $(SLAB_ALL_OBJS) \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

slab-cmp: $(SLAB_CMP)
	./$(SLAB_CMP)

# ---- M4: GPU Freudenthal search kernel ----
# Per-arch compile of the kernel host entry point (same SIEVE_KERNEL_ARCH
# pattern as sieve_slab_kernel). The kernel body is in src/m4/gpu_search_kernel.h.
M4_OBJS_AMD := $(BUILD)/m4_host_amd.o
M4_OBJS_NV  := $(BUILD)/m4_host_nv.o
M4_ALL_OBJS := $(M4_OBJS_AMD) $(M4_OBJS_NV)

$(BUILD)/m4_host_amd.o: src/m4/gpu_search_kernel.cpp src/m4/gpu_search_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

$(BUILD)/m4_host_nv.o: src/m4/gpu_search_kernel.cpp src/m4/gpu_search_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# m4_kernel_unit test binary (g++ host, links per-arch kernel objects + DevAbstraction + runtime).
M4_TEST := tests/m4_kernel_unit_bin
$(M4_TEST): tests/m4_kernel_unit.cpp src/m4/gpu_search_kernel.h src/devabstraction.h \
            $(M4_ALL_OBJS) $(BUILD)/devabstraction.o $(BUILD)/device_registry.o \
            $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
            $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o \
            $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ tests/m4_kernel_unit.cpp \
	    $(M4_ALL_OBJS) $(BUILD)/devabstraction.o $(BUILD)/device_registry.o \
	    $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
	    $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o \
	    $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

m4: $(M4_TEST)
	./$(M4_TEST)

# ---- M4 ordered emission test (gpu_search_emission + kernel) ----
# Host-only test binary (g++). Links per-arch kernel objects + DevAbstraction
# + runtime. Runs the GPU Freudenthal search, emits results via GpuSearchEmit,
# and diffs against the reference ff_seg / golden output.
M4_ORDER_BIN := tests/m4_order_bin
$(M4_ORDER_BIN): tests/m4_order.cpp src/m4/gpu_search_emission.cpp \
                 src/m4/gpu_search_kernel.h src/devabstraction.h \
                 $(M4_ALL_OBJS) $(BUILD)/devabstraction.o \
                 $(BUILD)/device_registry.o $(BUILD)/hip_enum.o \
                 $(BUILD)/cuda_enum.o $(BUILD)/hip_devabstraction.o \
                 $(BUILD)/cuda_devabstraction.o $(BUILD)/gpu_prime.o \
                 $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
                 | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ tests/m4_order.cpp \
	    $(M4_ALL_OBJS) $(BUILD)/devabstraction.o $(BUILD)/device_registry.o \
	    $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
	    $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o \
	    $(BUILD)/gpu_prime.o \
	    $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

m4-order: $(M4_ORDER_BIN)
	./$(M4_ORDER_BIN)

# ---- M4 search launcher TU (g++, host-only — no SIEVE_KERNEL_ARCH) ----
# gpu_search_launcher.cpp compiles the host-side dispatch wrapper that
# resolves per-arch SearchKernelRunFn symbols from the per-arch M4 kernel
# objects at link time.  No device code — pure g++ host compile.
$(BUILD)/gpu_search_launcher.o: src/m4/gpu_search_launcher.cpp src/m4/gpu_search_launcher.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -c $< -o $@

# ---- M0 benchmark: kernel launch TUs (compiled per-arch like smoke) ----
$(BUILD)/hip_m0_kernels.o: src/hip_m0_kernels.cpp src/m0_kernel.h | $(BUILD)
	$(HIPCC) $(AMDFLAGS) -c $< -o $@

$(BUILD)/cuda_m0_kernels.o: src/cuda_m0_kernels.cpp src/m0_kernel.h | $(BUILD)
	HIP_PLATFORM=nvidia $(HIPCC) $(NVFLAGS) -x cu -arch=$(ARCH_NV) -c $< -o $@

# ---- M0 benchmark: host-only program (g++, links DevAbstraction + kernel TUs) ----
M0_BENCH := build/m0_bench

$(BUILD)/m0_benchmark.o: src/m0_benchmark.cpp src/devabstraction.h src/device_info.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

$(M0_BENCH): $(BUILD)/m0_benchmark.o $(BUILD)/devabstraction.o \
             $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
             $(BUILD)/device_registry.o \
             $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
             $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ $(BUILD)/m0_benchmark.o \
	    $(BUILD)/devabstraction.o \
	    $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
	    $(BUILD)/device_registry.o \
	    $(BUILD)/hip_enum.o $(BUILD)/cuda_enum.o \
	    $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64

bench: $(M0_BENCH)
	@echo "M0 bench binary built: $(M0_BENCH)"

# ---- todo 3: vendor-neutral modules (g++, no vendor headers) ----
$(BUILD)/main.o: src/main.cpp src/config.h src/budget.h src/geometry.h \
                  src/device_registry.h src/device_info.h \
                  src/sieve_engine.h src/gpu_prime.h src/cpu_search.h \
                  src/devabstraction.h \
                  src/m4/gpu_search_launcher.h src/m4/gpu_search_emission.cpp \
                  src/pull_scheduler.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/config.o: src/config.cpp src/config.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD)/budget.o: src/budget.cpp src/budget.h src/config.h src/device_info.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD)/geometry.o: src/geometry.cpp src/geometry.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

# ---- todo 8: full-map SieveEngine (g++ host, links per-arch engine + runtime) ----
$(BUILD)/sieve_engine.o: src/sieve_engine.cpp src/sieve_engine.h src/devabstraction.h \
                         src/geometry.h src/sieve_slab_engine.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

# ---- todo 10: M2 backing pool + weighted pulls (g++ host, std::thread) ----
$(BUILD)/pull_scheduler.o: src/pull_scheduler.cpp src/pull_scheduler.h \
                           src/config.h src/budget.h src/geometry.h \
                           src/device_info.h src/sieve_slab_engine.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -pthread -c $< -o $@

$(BUILD)/device_registry.o: src/device_registry.cpp src/device_registry.h \
                             src/device_info.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD)/gpu_prime.o: src/gpu_prime.cpp src/gpu_prime.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD)/cpu_search.o: src/cpu_search.cpp src/cpu_search.h src/gpu_prime.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -c $< -o $@

# ---- link: objects ONLY, never a .cu source on the link line ----
# -pthread: std::thread (pull_scheduler.o) needs libpthread at link time.
# DevAbstraction backends (hip_devabstraction.o + cuda_devabstraction.o) are
# linked here so ff_sieve can call ffdev::DevInit / DevAlloc / DevCopy / DevFree.
# --allow-multiple-definition: smoke TUs and backend TUs both include
# smoke/smoke_kernel.h; this flag lets them coexist in one binary.
$(BIN): $(OBJS) $(ENGINE_ALL_OBJS) $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o
	$(GXX) -pthread -o $@ $(OBJS) $(ENGINE_ALL_OBJS) \
	    $(BUILD)/hip_devabstraction.o $(BUILD)/cuda_devabstraction.o \
	    $(BUILD)/hip_m0_kernels.o $(BUILD)/cuda_m0_kernels.o \
	    $(CUDA_LIB) -lcudart $(ROCM_LIB) -lamdhip64 \
	    -Wl,--allow-multiple-definition

$(BUILD):
	mkdir -p $(BUILD)

$(SELFTEST): tests/budget_selftest.cpp src/config.cpp src/budget.cpp \
             src/geometry.cpp src/device_registry.cpp \
             src/config.h src/budget.h src/geometry.h src/device_registry.h \
             src/device_info.h | $(BUILD)
	$(GXX) $(HOST_CXXFLAGS) -Isrc -o $@ tests/budget_selftest.cpp src/config.cpp \
	    src/budget.cpp src/geometry.cpp src/device_registry.cpp

selftest: $(SELFTEST)
	./$(SELFTEST)

smoke: all
	./$(BIN)

clean:
	rm -rf $(BUILD) $(BIN)
