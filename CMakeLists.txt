cmake_minimum_required(VERSION 3.16)
project(MyWasmProject VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME}
    src/main.c
)

# Use a consistent output name for both native and Emscripten builds
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "only-gates-ai")

# Detect if we are cross-compiling with Emscripten
if(EMSCRIPTEN)
    # Target standard ES6 modules for modern web apps
    target_link_options(${PROJECT_NAME} PRIVATE
        "-sEXPORT_ES6"
        "-sENVIRONMENT=web"
        "-sEXPORT_NAME=Module"

        # Functions we want to call from JS.
        # C functions get an underscore prefix in JS, so "func" is Module._func.
        "-sEXPORTED_FUNCTIONS=['_func', '_create_circuit', '_destroy_circuit']"

        # Emscripten helper utilities we want to use in JS
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
    )

    # Change output extension from a native binary to JavaScript/Wasm
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".js")
endif()
