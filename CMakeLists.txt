cmake_minimum_required(VERSION 3.16)
project(MyWasmProject VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(EMSCRIPTEN)
    add_executable(${PROJECT_NAME}
        src/main.c
    )

    # Use a consistent output name for the wasm module
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "only-gates-ai")

    # DWARF is generated at compile time, but Emscripten strips it at link time
    # unless a -g flag is also passed there. Debug info must be present in the
    # object files for it to survive into the linked wasm, so pass it here too.
    target_compile_options(${PROJECT_NAME} PRIVATE "-g")

    # Target standard ES6 modules for modern web apps
    target_link_options(${PROJECT_NAME} PRIVATE
        "-sEXPORT_ES6"
        "-sENVIRONMENT=web"
        "-sEXPORT_NAME=Module"

        # Functions we want to call from JS.
        # C functions get an underscore prefix in JS, so "func" is Module._func.
        # malloc/free are needed to pass JS arrays into C as pointers.
        "-sEXPORTED_FUNCTIONS=['_func', '_create_circuit', '_destroy_circuit', '_malloc', '_free']"

        # Emscripten helper utilities we want to use in JS
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"

        # Keep DWARF in a separate wasm file so the served binary stays small.
        # The main wasm points at this file via an external_debug_info section.
        "-gseparate-dwarf=only-gates-ai.debug.wasm"
    )

    # Change output extension from a native binary to JavaScript/Wasm
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".js")
endif()

include(CTest)
if(BUILD_TESTING AND NOT EMSCRIPTEN)
    add_subdirectory(tests)
endif()
