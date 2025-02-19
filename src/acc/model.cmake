
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that supports OpenACC as per CMake detection"
        "c++")

register_flag_optional(TARGET_DEVICE
        "[PGI/NVHPC only] This sets the `-target` flag, possible values are:
             gpu       - Globally set the target device to an NVIDIA GPU
             multicore - Globally set the target device to the host CPU
         Refer to `nvc++ --help` for the full list"
        "")


register_flag_optional(CUDA_ARCH
        "[PGI/NVHPC only] Only applicable if `TARGET_DEVICE` is set to `gpu`.
         Nvidia architecture in ccXY format, for example, sm_70 becomes cc70, will be passed in via `-gpu=` (e.g `cc70`)
         Possible values are:
             cc35  - Compile for compute capability 3.5
             cc50  - Compile for compute capability 5.0
             cc60  - Compile for compute capability 6.0
             cc62  - Compile for compute capability 6.2
             cc70  - Compile for compute capability 7.0
             cc72  - Compile for compute capability 7.2
             cc75  - Compile for compute capability 7.5
             cc80  - Compile for compute capability 8.0
             ccall - Compile for all supported compute capabilities
         Refer to `nvc++ --help` for the full list"
        "")

register_flag_optional(TARGET_PROCESSOR
        "[PGI/NVHPC only] This sets the `-tp` (target processor) flag, possible values are:
             px          - Generic x86 Processor
             bulldozer   - AMD Bulldozer processor
             piledriver  - AMD Piledriver processor
             zen         - AMD Zen architecture (Epyc, Ryzen)
             zen2        - AMD Zen 2 architecture (Ryzen 2)
             sandybridge - Intel SandyBridge processor
             haswell     - Intel Haswell processor
             knl         - Intel Knights Landing processor
             skylake     - Intel Skylake Xeon processor
             host        - Link native version of HPC SDK cpu math library
             native      - Alias for -tp host
        Refer to `nvc++ --help` for the full list"
        "")

register_flag_optional(OFFLOAD_FLAGS
   "OpenACC Offload Flags"
   ""
)

set(ACC_FLAGS_OFFLOAD_GNU -foffload=-lm)

macro(setup)
    find_package(OpenACC)

    if(OpenACC_CXX_FOUND)
        register_link_library(OpenACC::OpenACC_CXX)
    endif()

    separate_arguments(OFFLOAD_FLAGS)
    register_append_cxx_flags(ANY ${OFFLOAD_FLAGS})
    register_append_link_flags(${OFFLOAD_FLAGS})

endmacro()

