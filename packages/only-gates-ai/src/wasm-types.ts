/** Options passed to the generated Emscripten module factory. */
export interface EmscriptenModuleOptions {
  locateFile?(path: string, prefix: string): string;
  print?(text: string): void;
  printErr?(text: string): void;
  wasmBinary?: ArrayBuffer | Uint8Array;
  [key: string]: unknown;
}

/** Raw functions and memory views exposed by the C module. */
export interface OnlyGatesWasmModule {
  readonly HEAPU8: Uint8Array;
  readonly HEAP32: Int32Array;
  readonly HEAPF32: Float32Array;

  _malloc(size: number): number;
  _free(pointer: number): void;
  _get_error(): number;

  _create_circuit(shapePointer: number, shapeLength: number): number;
  _destroy_circuit(circuit: number): void;
  _circuit_set_input(circuit: number, inputPointer: number): void;
  _circuit_read_output(circuit: number, outputPointer: number): void;
  _circuit_feed_forward(circuit: number): void;
  _circuit_back_propergate(
    circuit: number,
    errorPointer: number,
    learningRate: number,
  ): void;
  _circuit_get_inputs_len(circuit: number): number;
  _circuit_get_outputs_len(circuit: number): number;
  _circuit_get_shape_len(circuit: number): number;
  _circuit_get_shape(circuit: number, index: number): number;
  _circuit_get_cell_len(circuit: number): number;
  _circuit_get_connection_probabilities_len(circuit: number): number;
  _circuit_get_connection_probabilities(circuit: number): number;
  _circuit_get_cell_gate_probabilities(
    circuit: number,
    cellIndex: number,
  ): number;
  _circuit_get_activations_len(circuit: number): number;
  _circuit_get_activations(circuit: number): number;

  _create_hard_circuit(circuit: number): number;
  _destroy_hard_circuit(hardCircuit: number): void;
  _hard_circuit_set_input(hardCircuit: number, inputPointer: number): void;
  _hard_circuit_read_output(
    hardCircuit: number,
    outputPointer: number,
  ): void;
  _hard_circuit_feed_forward(hardCircuit: number): void;
  _hard_circuit_get_connections_len(hardCircuit: number): number;
  _hard_circuit_get_connections(hardCircuit: number): number;
  _hard_circuit_get_gates_len(hardCircuit: number): number;
  _hard_circuit_get_gates(hardCircuit: number): number;
  _hard_circuit_get_activations_len(hardCircuit: number): number;
  _hard_circuit_get_activations(hardCircuit: number): number;

  _create_trainer(
    circuit: number,
    inputsPointer: number,
    outputsPointer: number,
    dataCount: number,
    learningRate: number,
  ): number;
  _destroy_trainer(trainer: number): void;
  _trainer_reset(trainer: number): void;
  _trainer_set_learning_rate(trainer: number, learningRate: number): void;
  _trainer_step(trainer: number): number;

  /* ------------------------------------------------------------------ */
  /* Tiny GPT integer inference                                           */
  /* ------------------------------------------------------------------ */
  _tiny_gpt_create(
    vocabSize: number,
    context: number,
    dim: number,
    layers: number,
    heads: number,
    mlp: number,
    packedWeightsPointer: number,
  ): number;
  _tiny_gpt_destroy(model: number): void;
  _tiny_gpt_forward(model: number, tokensPointer: number): void;
  _tiny_gpt_forward_range(
    model: number,
    tokensPointer: number,
    start: number,
    count: number,
  ): void;
  _tiny_gpt_reset_cache(model: number): void;
  _tiny_gpt_logits(model: number): number;
  _tiny_gpt_logits_row(model: number, position: number): number;
  _tiny_gpt_buffer(model: number, which: number): number;
  _tiny_gpt_probabilities(
    model: number,
    temperatureQ16: number,
    outPointer: number,
  ): void;
  _tiny_gpt_sample(
    model: number,
    temperatureQ16: number,
    topK: number,
  ): number;
  _tiny_gpt_set_seed(model: number, seed: number): void;
  _tiny_gpt_vocab_size(model: number): number;
  _tiny_gpt_context(model: number): number;
  _tiny_gpt_last_error(): number;
}
