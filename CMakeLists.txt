cmake_minimum_required(VERSION 3.16)
project(MyWasmProject VERSION 1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(EMSCRIPTEN)
    set(ONLY_GATES_SOURCES
        src/main.c
        src/tiny_gpt_infer.c
        src/tiny_gpt_quantize.c
    )

    # Link flags shared by every inference lane. Each lane exposes the same C
    # API, so callers can swap the module without touching the model wrapper.
    set(ONLY_GATES_LINK_OPTIONS
        "-sEXPORT_ES6"
        "-sENVIRONMENT=web,node,worker"
        "-sEXPORT_NAME=Module"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sEXPORTED_FUNCTIONS=['_func','_get_error','_create_circuit','_destroy_circuit','_circuit_set_input','_circuit_read_output','_circuit_feed_forward','_circuit_back_propergate','_circuit_get_inputs_len','_circuit_get_outputs_len','_circuit_get_shape_len','_circuit_get_shape','_circuit_get_cell_len','_circuit_get_connection_probabilities_len','_circuit_get_connection_probabilities','_circuit_get_cell_gate_probabilities','_circuit_get_activations_len','_circuit_get_activations','_create_hard_circuit','_destroy_hard_circuit','_hard_circuit_set_input','_hard_circuit_read_output','_hard_circuit_feed_forward','_hard_circuit_get_connections_len','_hard_circuit_get_connections','_hard_circuit_get_gates_len','_hard_circuit_get_gates','_hard_circuit_get_activations_len','_hard_circuit_get_activations','_create_trainer','_destroy_trainer','_trainer_reset','_trainer_set_learning_rate','_trainer_step','_tiny_gpt_create','_tiny_gpt_destroy','_tiny_gpt_forward','_tiny_gpt_forward_range','_tiny_gpt_reset_cache','_tiny_gpt_logits','_tiny_gpt_logits_row','_tiny_gpt_probabilities','_tiny_gpt_probabilities_at','_tiny_gpt_sample','_tiny_gpt_sample_at','_tiny_gpt_sample_at_penalized','_tiny_gpt_set_seed','_tiny_gpt_vocab_size','_tiny_gpt_context','_tiny_gpt_last_error','_tiny_gpt_buffer','_malloc','_free']"

        # Emscripten helper utilities we want to use in JS. The HEAP views must
        # be listed explicitly: recent Emscripten versions no longer attach them
        # to the module automatically.
        "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAP32','HEAPF32']"
    )

    # Scalar lane: the reference module, also the fallback when a browser does
    # not implement WebAssembly SIMD.
    add_executable(${PROJECT_NAME} ${ONLY_GATES_SOURCES})
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "only-gates-ai")
    target_compile_options(${PROJECT_NAME} PRIVATE "-g")
    target_link_options(${PROJECT_NAME} PRIVATE
        ${ONLY_GATES_LINK_OPTIONS}
        # Keep DWARF in a separate wasm file so the served binary stays small.
        "-gseparate-dwarf=only-gates-ai.debug.wasm"
    )

    # SIMD lane: identical arithmetic and exports, compiled with wasm SIMD.
    # Emscripten picks this up through __wasm_simd128__ in tiny_gpt_infer.c.
    add_executable(${PROJECT_NAME}-simd ${ONLY_GATES_SOURCES})
    set_target_properties(${PROJECT_NAME}-simd PROPERTIES OUTPUT_NAME "only-gates-ai-simd")
    target_compile_options(${PROJECT_NAME}-simd PRIVATE "-g" "-msimd128")
    target_link_options(${PROJECT_NAME}-simd PRIVATE
        ${ONLY_GATES_LINK_OPTIONS}
        "-msimd128"
        "-gseparate-dwarf=only-gates-ai-simd.debug.wasm"
    )

    # Change output extension from a native binary to JavaScript/Wasm
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".js")
    set_target_properties(${PROJECT_NAME}-simd PROPERTIES SUFFIX ".js")
endif()

include(CTest)
if(BUILD_TESTING AND NOT EMSCRIPTEN)
    add_subdirectory(tests)
endif()
