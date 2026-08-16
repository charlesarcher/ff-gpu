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

# ---- CTest registration ----

add_test(NAME abstraction_smoke COMMAND abstraction_smoke)
add_test(NAME ff_budget_selftest COMMAND ff_budget_selftest)
add_test(NAME slab_cmp COMMAND slab_cmp)
add_test(NAME m4_kernel_unit_bin COMMAND m4_kernel_unit_bin)
add_test(NAME m4_order_bin COMMAND m4_order_bin)
