# Tests mirror the Makefile's link topology. Coarse object libraries are used
# everywhere except the pure-host budget_selftest (which must stay free of GPU
# runtimes); --allow-multiple-definition (from ff_gpu_setup_binary) absorbs the
# duplicated kernel bodies the same way the Makefile's link lines did.

# ---- Dual-runtime smoke: devabstraction + per-arch backends + enums ----

add_executable(abstraction_smoke source/abstraction_smoke.cpp)

target_compile_features(abstraction_smoke PRIVATE cxx_std_17)

target_link_libraries(
    abstraction_smoke PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(abstraction_smoke)

# ---- Pure-host budget/geometry sanity (no GPU runtimes linked) ----

add_executable(ff_budget_selftest source/budget_selftest.cpp)

# Task-15 (D) oracle: GpuPrime::IsPrime over an expanded canonical map.
# gpu_prime.cpp is vendor-free, so the pure-host link stays GPU-runtime-free.
target_sources(ff_budget_selftest PRIVATE ${PROJECT_SOURCE_DIR}/source/gpu_prime.cpp)

target_compile_features(ff_budget_selftest PRIVATE cxx_std_17)

target_link_libraries(ff_budget_selftest PRIVATE ff-gpu_core)

# ---- Slab kernel per-arch comparison ----

add_executable(slab_cmp source/slab_cmp.cpp)

target_compile_features(slab_cmp PRIVATE cxx_std_17)

target_link_libraries(
    slab_cmp PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(slab_cmp)

# ---- M4 kernel unit test ----

add_executable(m4_kernel_unit_bin source/m4_kernel_unit.cpp)

target_compile_features(m4_kernel_unit_bin PRIVATE cxx_std_17)

target_link_libraries(
    m4_kernel_unit_bin PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(m4_kernel_unit_bin)

# ---- M4 order comparison against reference ff_seg goldens ----

add_executable(m4_order_bin source/m4_order.cpp)

target_compile_features(m4_order_bin PRIVATE cxx_std_17)

target_link_libraries(
    m4_order_bin PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(m4_order_bin)

# ---- M4 Miller-Rabin differential: device dev_IsPrime vs host GpuPrime ----

add_executable(m4_mr_diff_bin source/m4_mr_diff.cpp)

target_compile_features(m4_mr_diff_bin PRIVATE cxx_std_17)

target_link_libraries(
    m4_mr_diff_bin PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(m4_mr_diff_bin)

# ---- Slab geometry bench (task 13): per-arch self-contained hipcc TUs ----
# Unlike the g++ host drivers above, this harness launches SieveSlabKernel
# directly (event-bracketed timing), so each binary is compiled BY hipcc for
# exactly one arch — same flags as the product object blocks, but the binary
# links only its own single object. Built here at the baked geometry defaults;
# the task-13 sweep builds scratch copies with the FF_SIEVE_* macros
# overridden via -D (see .omo/evidence/gpu-speedup/task-13-gpu-kernel-speedup/).

set(SLAB_BENCH_SRC test/source/slab_geom_bench.cpp)

# Absolute header-dep paths: this file is processed from test/, where the
# parent scope's relative AMD_SRC_HEADERS entries would resolve wrong.
set(SLAB_BENCH_DEPS "")
foreach(hdr IN LISTS AMD_SRC_HEADERS)
    list(APPEND SLAB_BENCH_DEPS "${PROJECT_SOURCE_DIR}/${hdr}")
endforeach()

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_amd"
    COMMAND
        ${CMAKE_COMMAND} -E env HIP_PLATFORM=amd
        /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1201
        -I/opt/rocm/include
        "-I${PROJECT_SOURCE_DIR}/source"
        -DSIEVE_KERNEL_ARCH=gfx1201
        "${PROJECT_SOURCE_DIR}/${SLAB_BENCH_SRC}"
        -o "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_amd"
    DEPENDS "${PROJECT_SOURCE_DIR}/${SLAB_BENCH_SRC}" ${SLAB_BENCH_DEPS}
    COMMENT "hipcc (AMD, gfx1201): slab_geom_bench"
    VERBATIM
)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_nv"
    COMMAND
        ${CMAKE_COMMAND} -E env HIP_PLATFORM=nvidia
        /opt/rocm/bin/hipcc -O3
        -I/opt/rocm/include
        "-I${PROJECT_SOURCE_DIR}/source"
        -DSIEVE_KERNEL_ARCH=sm_120 -x cu -arch=sm_120
        "${PROJECT_SOURCE_DIR}/${SLAB_BENCH_SRC}"
        -o "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_nv"
    DEPENDS "${PROJECT_SOURCE_DIR}/${SLAB_BENCH_SRC}" ${SLAB_BENCH_DEPS}
    COMMENT "hipcc (NVIDIA, sm_120): slab_geom_bench"
    VERBATIM
)

add_custom_target(slab_geom_bench_binaries ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_amd"
            "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_nv"
)

add_test(NAME slab_geom_bench_amd
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_amd" --reps=3)
add_test(NAME slab_geom_bench_nv
         COMMAND "${CMAKE_CURRENT_BINARY_DIR}/slab_geom_bench_nv" --reps=3)
add_test(NAME abstraction_smoke COMMAND abstraction_smoke)
add_test(NAME ff_budget_selftest COMMAND ff_budget_selftest)
add_test(NAME slab_cmp COMMAND slab_cmp)
add_test(NAME m4_kernel_unit_bin COMMAND m4_kernel_unit_bin)
add_test(NAME m4_order_bin COMMAND m4_order_bin)
add_test(NAME m4_mr_diff_bin COMMAND m4_mr_diff_bin)

# ---- Task-9 hostmap coverage invariant (poison-build trap) ----
# Poison twin of ff_sieve: SAME sources and link topology, compiled with
# FF_POISON_HOSTMAP=1 so the host map starts sentinel-filled (0xA5) instead
# of zero-filled. hostmap_coverage_test runs it across the task-18 run matrix
# (leg 65536 both search modes; leg 524288 tiled / mixed-grid / adaptive slab
# geometries) asserting stdout is byte-identical to the golden after timing
# normalization, plus a --dump-map case byte-compared against the CPU
# reference ff_seg map (full-byte coverage of the tiled expansion). The
# production ff-gpu_exe keeps FF_POISON_HOSTMAP OFF (see top-level
# CMakeLists.txt), so its behavior is untouched.

add_executable(ff_sieve_poison "${PROJECT_SOURCE_DIR}/source/main.cpp")

target_compile_features(ff_sieve_poison PRIVATE cxx_std_17)

target_compile_definitions(ff_sieve_poison PRIVATE FF_POISON_HOSTMAP=1)

target_link_libraries(
    ff_sieve_poison PRIVATE
    ff-gpu_core
    ff-gpu_lib
    ff-gpu_cpu_search
    ff-gpu_amd
    ff-gpu_nv
)

ff_gpu_setup_binary(ff_sieve_poison)

add_executable(hostmap_coverage_test source/hostmap_coverage.cpp)

target_compile_features(hostmap_coverage_test PRIVATE cxx_std_17)

target_compile_definitions(hostmap_coverage_test PRIVATE
    FF_COVERAGE_POISON_BIN="$<TARGET_FILE:ff_sieve_poison>"
    FF_COVERAGE_REF_BIN="$<TARGET_FILE:ff_seg>"
    FF_COVERAGE_GOLDENS_DIR="${PROJECT_SOURCE_DIR}/goldens"
)

add_dependencies(hostmap_coverage_test ff_sieve_poison ff_seg)

add_test(NAME hostmap_coverage_test COMMAND hostmap_coverage_test)

set_tests_properties(hostmap_coverage_test PROPERTIES TIMEOUT 300)
