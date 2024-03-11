macro(setup)
    # nothing to do here as hipcc does everything correctly, what a surprise!
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE HIP)
endmacro()
