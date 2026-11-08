set(TEST_NAMES
    test_circuit
    test_gradients
    test_hard_circuit
    test_trainer
    test_attention
    test_attention_vectors
    test_tiny_gpt
)

foreach(test_name IN LISTS TEST_NAMES)
    add_executable(${test_name} ${test_name}.c)
    target_include_directories(${test_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/stub")
    target_link_libraries(${test_name} PRIVATE m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
