cmake_minimum_required(VERSION 3.28)

include(cmake/prelude.cmake)

project(
    ff-gpu
    VERSION 0.1.0
    DESCRIPTION "AMD HIP GPU prime sieve engine"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Vendor-neutral host library (g++, no vendor headers) ----
# Split from ff-gpu_lib so the pure-host budget_selftest can link without GPU
# runtimes, mirroring the Makefile's object granularity.

add_library(
    ff-gpu_core OBJECT
    source/config.cpp
    source/budget.cpp
    source/geometry.cpp
    source/device_registry.cpp
)

target_include_directories(
    ff-gpu_core ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(ff-gpu_core PUBLIC cxx_std_17)

# ---- Host library that depends on the per-arch backends (g++) ----
# cpu_search.cpp is deliberately NOT here: it calls PrintOutputTags, which is
# defined only in m4/gpu_search_emission.cpp (textually included into
# main.cpp), so cpu_search.o links ONLY into ff-gpu_exe — mirroring the
# Makefile, where no test binary ever linked cpu_search.o.

add_library(
    ff-gpu_lib OBJECT
    source/devabstraction.cpp
    source/sieve_engine.cpp
    source/gpu_prime.cpp
    source/pull_scheduler.cpp
    source/m4/gpu_search_launcher.cpp
    source/smoke/smoke_main.cpp
)

target_include_directories(
    ff-gpu_lib ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(ff-gpu_lib PUBLIC cxx_std_17)

# ---- ff_sieve-only host TU (g++) ----
# See the ff-gpu_lib comment: cpu_search.o needs PrintOutputTags, which exists
# only in the TU that textually includes m4/gpu_search_emission.cpp (main.cpp).

add_library(ff-gpu_cpu_search OBJECT source/cpu_search.cpp)

target_include_directories(
    ff-gpu_cpu_search ${warning_guard}
    PUBLIC
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_compile_features(ff-gpu_cpu_search PUBLIC cxx_std_17)

# ---- Per-arch AMD objects (custom hipcc commands, gfx1201) ----
# Same hand-rolled command the Makefile used. CMake's HIP language driver
# rejects the hipcc wrapper on this machine (CMake >= 4.0 wants raw Clang),
# so both vendor sides use plain hipcc custom commands; the objects are
# exposed to consumers via IMPORTED OBJECT libraries.

set(
    AMD_SOURCES
    source/hip_enum.cpp
    source/smoke/hip_smoke.cpp
    source/hip_devabstraction.cpp
    source/hip_m0_kernels.cpp
    source/m4/gpu_search_kernel.cpp
    source/sieve_slab_kernel.cpp
    source/sieve_slab_engine.cpp
)

set(AMD_OBJECTS "")

# Project headers each per-arch source includes — without these in DEPENDS,
# editing a kernel header silently leaves stale arch objects linked in.
set(
    AMD_SRC_HEADERS
    source/smoke/smoke_kernel.h
    source/devabstraction.h
    source/m0_kernel.h
    source/m4/gpu_search_kernel.h
    source/sieve_slab_kernel.h
    source/sieve_slab_engine.h
)

foreach(amd_src IN LISTS AMD_SOURCES)
    get_filename_component(amd_stem "${amd_src}" NAME_WE)
    set(amd_obj "${CMAKE_CURRENT_BINARY_DIR}/amd_objs/${amd_stem}_gfx1201.o")
    add_custom_command(
        OUTPUT "${amd_obj}"
        COMMAND
            ${CMAKE_COMMAND} -E env HIP_PLATFORM=amd
            /opt/rocm/bin/hipcc -O2 -Wall -I/opt/rocm/include
            "-I${PROJECT_SOURCE_DIR}/source"
            -DSIEVE_KERNEL_ARCH=gfx1201
            -c "${PROJECT_SOURCE_DIR}/${amd_src}" -o "${amd_obj}"
        DEPENDS
            "${PROJECT_SOURCE_DIR}/${amd_src}"
            ${AMD_SRC_HEADERS}
        COMMENT "hipcc (AMD, gfx1201): ${amd_src}"
    )
    list(APPEND AMD_OBJECTS "${amd_obj}")
endforeach()

add_library(ff-gpu_amd OBJECT IMPORTED GLOBAL)

set_target_properties(
    ff-gpu_amd PROPERTIES
    IMPORTED_OBJECTS "${AMD_OBJECTS}"
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/source"
)

add_custom_target(ff-gpu_amd_objects DEPENDS ${AMD_OBJECTS})

# ---- Per-arch NVIDIA objects (custom hipcc commands, sm_120) ----
# The SAME HIP-only sources as the AMD side, compiled through hipcc's NVIDIA
# backend — no direct CUDA calls exist anywhere in this project:
#   HIP_PLATFORM=nvidia hipcc -x cu -arch=sm_120 -DSIEVE_KERNEL_ARCH=sm_120
# Each object exports arch-tagged symbols (…_gfx1201 / …_sm_120), so both
# sets link into one binary that drives both runtimes side by side.

set(
    NV_SOURCES
    source/hip_enum.cpp
    source/smoke/hip_smoke.cpp
    source/hip_devabstraction.cpp
    source/hip_m0_kernels.cpp
    source/m4/gpu_search_kernel.cpp
    source/sieve_slab_kernel.cpp
    source/sieve_slab_engine.cpp
)

set(NV_OBJECTS "")

foreach(nv_src IN LISTS NV_SOURCES)
    get_filename_component(nv_stem "${nv_src}" NAME_WE)
    set(nv_obj "${CMAKE_CURRENT_BINARY_DIR}/nv_objs/${nv_stem}_sm120.o")
    add_custom_command(
        OUTPUT "${nv_obj}"
        COMMAND
            ${CMAKE_COMMAND} -E env HIP_PLATFORM=nvidia
            /opt/rocm/bin/hipcc -O2 -I/opt/rocm/include
            "-I${PROJECT_SOURCE_DIR}/source"
            -DFF_BACKEND_NV=1
            -DSIEVE_KERNEL_ARCH=sm_120 -x cu -arch=sm_120
            -c "${PROJECT_SOURCE_DIR}/${nv_src}" -o "${nv_obj}"
        DEPENDS
            "${PROJECT_SOURCE_DIR}/${nv_src}"
            ${AMD_SRC_HEADERS}
        COMMENT "hipcc (NVIDIA, sm_120): ${nv_src}"
    )
    list(APPEND NV_OBJECTS "${nv_obj}")
endforeach()

add_library(ff-gpu_nv OBJECT IMPORTED GLOBAL)

set_target_properties(
    ff-gpu_nv PROPERTIES
    IMPORTED_OBJECTS "${NV_OBJECTS}"
    INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/source"
)

add_custom_target(ff-gpu_nv_objects DEPENDS ${NV_OBJECTS})

# ---- Executable ----

add_executable(ff-gpu_exe source/main.cpp)
add_executable(ff-gpu::exe ALIAS ff-gpu_exe)

set_property(TARGET ff-gpu_exe PROPERTY OUTPUT_NAME ff_sieve)

target_compile_features(ff-gpu_exe PRIVATE cxx_std_17)

target_link_libraries(
    ff-gpu_exe PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_cpu_search
    ff-gpu_amd
    ff-gpu_nv
)

# ---- GPU runtimes / link flags (mirror Makefile link line) ----

find_package(Threads REQUIRED)

set(
    FF_GPU_RUNTIME_LIBS
    amdhip64
    cudart
    Threads::Threads
)

function(ff_gpu_setup_binary target)
    target_link_directories(
        ${target} PRIVATE
        /opt/rocm/lib
        /usr/local/cuda/lib64
    )
    target_link_libraries(${target} PRIVATE ${FF_GPU_RUNTIME_LIBS})
    # Kernel bodies live in headers that multiple TUs include (smoke_kernel.h,
    # sieve_slab_kernel.h, m4/gpu_search_kernel.h) — the Makefile relied on the
    # linker keeping the first definition; keep that behavior.
    target_link_options(${target} PRIVATE -Wl,--allow-multiple-definition)
    add_dependencies(${target} ff-gpu_amd_objects ff-gpu_nv_objects)
endfunction()

ff_gpu_setup_binary(ff-gpu_exe)

# ---- Benchmark (m0_bench) ----
# m0_benchmark.cpp owns its own main() and is never part of ff_sieve.

add_executable(m0_bench source/m0_benchmark.cpp)

target_compile_features(m0_bench PRIVATE cxx_std_17)

target_link_libraries(
    m0_bench PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(m0_bench)

# ---- Reference programs (plain g++, mirror Makefile.reference) ----
# Outputs stay in reference/ so scripts/regenerate_goldens.sh and verify.sh
# keep finding $ROOT/reference/ff_seg etc.

set(
    REFERENCE_PROGRAMS
    segmentedSieve.C:ff_seg
    penultimateBestParallelizedFastFruedenthalYet.C:pen
    penultimateBestParallelizedFastFruedenthalYet2.C:pen2
)

foreach(ref_spec IN LISTS REFERENCE_PROGRAMS)
    string(REPLACE ":" ";" ref_parts "${ref_spec}")
    list(GET ref_parts 0 ref_src)
    list(GET ref_parts 1 ref_bin)
    add_executable("${ref_bin}" "reference/${ref_src}")
    target_compile_features("${ref_bin}" PRIVATE cxx_std_17)
    set_property(
        TARGET "${ref_bin}" PROPERTY
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/reference"
    )
endforeach()

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Developer mode ----

if(NOT ff-gpu_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of ff-gpu"
  )
endif()

include(cmake/dev-mode.cmake)
