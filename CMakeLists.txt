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

    # Target standard ES6 modules for modern web apps, Node and workers. The
    # TypeScript package is built on this module, so allow it to run in tests
    # and server-side code as well as in the browser.
    target_link_options(${PROJECT_NAME} PRIVATE
        "-sEXPORT_ES6"
        "-sENVIRONMENT=web,node,worker"
        "-sEXPORT_NAME=Module"
        "-sALLOW_MEMORY_GROWTH=1"

        # Functions we want to call from JS.
        # C functions get an underscore prefix in JS, so "func" is Module._func.
        # malloc/free are needed to pass JS arrays into C as pointers.
        "-sEXPORTED_FUNCTIONS=['_func','_get_error','_create_circuit','_destroy_circuit','_circuit_set_input','_circuit_read_output','_circuit_feed_forward','_circuit_back_propergate','_circuit_get_inputs_len','_circuit_get_outputs_len','_circuit_get_shape_len','_circuit_get_shape','_circuit_get_cell_len','_circuit_get_connection_probabilities_len','_circuit_get_connection_probabilities','_circuit_get_cell_gate_probabilities','_circuit_get_activations_len','_circuit_get_activations','_create_hard_circuit','_destroy_hard_circuit','_hard_circuit_set_input','_hard_circuit_read_output','_hard_circuit_feed_forward','_hard_circuit_get_connections_len','_hard_circuit_get_connections','_hard_circuit_get_gates_len','_hard_circuit_get_gates','_hard_circuit_get_activations_len','_hard_circuit_get_activations','_create_trainer','_destroy_trainer','_trainer_reset','_trainer_set_learning_rate','_trainer_step','_malloc','_free']"

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
