install(
    TARGETS ff-gpu_exe
    RUNTIME COMPONENT ff-gpu_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
