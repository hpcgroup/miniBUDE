
register_flag_optional(CMAKE_CXX_COMPILER
        "Any CXX compiler that is supported by CMake detection and Kokkos.
         See https://github.com/kokkos/kokkos#primary-tested-compilers-on-x86-are"
        "c++")

# compiler vendor and arch specific flags
set(KOKKOS_FLAGS_CPU_INTEL -qopt-streaming-stores=always)

macro(setup)

    cmake_policy(SET CMP0074 NEW) #see https://github.com/kokkos/kokkos/blob/master/BUILD.md

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_EXTENSIONS OFF)

    find_package(Kokkos REQUIRED)

    register_link_library(Kokkos::kokkos)

    if (${KOKKOS_BACK_END} STREQUAL "CUDA")
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)

        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

        set_source_files_properties(${IMPL_SOURCES} PROPERTIES LANGUAGE CUDA)
    elseif (${KOKKOS_BACK_END} STREQUAL "HIP")
        find_package(hip REQUIRED)

        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)

        set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
    endif ()

endmacro()


