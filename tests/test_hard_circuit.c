#include "test_common.h"

#include <string.h>

#include "../src/main.c"

static void test_hard_gate_truth_table(void) {
  // Rows follow the project's gate order, columns are AB = 00, 01, 10, 11.
  static const bool truth[16][4] = {
      {0, 0, 0, 0}, // zero
      {1, 0, 0, 0}, // nor
      {0, 0, 1, 0}, // lnt: a && !b
      {1, 1, 0, 0}, // not_a
      {0, 1, 0, 0}, // rnt: !a && b
      {1, 0, 1, 0}, // not_b
      {0, 1, 1, 0}, // xor
      {1, 1, 1, 0}, // nand
      {0, 0, 0, 1}, // and
      {1, 0, 0, 1}, // xnor
      {0, 1, 0, 1}, // proj_b
      {1, 1, 0, 1}, // imp: !a || b
      {0, 0, 1, 1}, // proj_a
      {1, 0, 1, 1}, // cimp: a || !b
      {0, 1, 1, 1}, // or
      {1, 1, 1, 1}, // one
  };

  for (int gate = 0; gate < 16; ++gate) {
    for (int index = 0; index < 4; ++index) {
      const bool a = (index & 2) != 0;
      const bool b = (index & 1) != 0;
      char name[64];
      snprintf(name, sizeof(name), "hard gate %d inputs %d,%d", gate, a, b);
      expect_true(name, hard_gate_forward(gate, a, b) == truth[gate][index]);
    }
  }
}

static void test_create_hard_circuit_argmax(void) {
  int shape[] = {2, 2, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for hard circuit", 0);
    return;
  }

  float connections[] = {
      0.1f, 0.9f, // hidden gate 0 pin a -> input 1
      0.8f, 0.2f, // hidden gate 0 pin b -> input 0
      0.3f, 0.7f, // hidden gate 1 pin a -> input 1
      0.4f, 0.6f, // hidden gate 1 pin b -> input 1
      0.2f, 0.8f, // output pin a -> hidden 1
      0.9f, 0.1f, // output pin b -> hidden 0
  };
  memcpy(circuit->connection_probabilities, connections, sizeof(connections));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  circuit->cells[0].gate_probabilities.and = 0.7f;
  circuit->cells[0].gate_probabilities.or = 0.2f;
  circuit->cells[0].gate_probabilities.xor = 0.1f;
  circuit->cells[1].gate_probabilities.nand = 0.6f;
  circuit->cells[1].gate_probabilities.xor = 0.3f;
  circuit->cells[1].gate_probabilities.one = 0.1f;

  HardCircuit *hard = create_hard_circuit(circuit);
  if (hard == NULL) {
    expect_true("create_hard_circuit", 0);
    destroy_circuit(circuit);
    return;
  }

  expect_true("hard connections_len", hard->connections_len == 6);
  expect_true("hard gates_len", hard->gates_len == 2);
  expect_true("hard wire 0", hard->connections[0] == 1);
  expect_true("hard wire 1", hard->connections[1] == 0);
  expect_true("hard wire 2", hard->connections[2] == 1);
  expect_true("hard wire 3", hard->connections[3] == 1);
  expect_true("hard wire 4", hard->connections[4] == 1);
  expect_true("hard wire 5", hard->connections[5] == 0);
  expect_true("hard gate 0 is AND", hard->gates[0] == 8);
  expect_true("hard gate 1 is NAND", hard->gates[1] == 7);

  bool input[] = {true, false};
  hard_circuit_set_input(hard, input);
  hard_circuit_feed_forward(hard);
  // hidden = {AND(0, 1), NAND(0, 0)} = {0, 1}, output = 0.5 * (1 + 0).
  expect_near("hard forward hidden 0", hard->activations[2], 0.0f, 1e-6f);
  expect_near("hard forward hidden 1", hard->activations[3], 1.0f, 1e-6f);
  expect_near("hard forward output", hard->outputs[0], 0.5f, 1e-6f);

  float output[1];
  hard_circuit_read_output(hard, output);
  expect_near("hard read output", output[0], 0.5f, 1e-6f);

  // Hard circuits copy their structure, so changing the soft circuit must not
  // affect an existing hard circuit.
  memset(circuit->connection_probabilities, 0,
         circuit->connection_probabilities_len *
             sizeof(*circuit->connection_probabilities));
  memset(circuit->cells[0].gate_probabilities.data, 0,
         16 * sizeof(*circuit->cells[0].gate_probabilities.data));
  expect_true("hard copy wire unchanged", hard->connections[0] == 1);
  expect_true("hard copy gate unchanged", hard->gates[0] == 8);

  destroy_hard_circuit(hard);
  destroy_circuit(circuit);
}

static void test_single_layer_hard_circuit(void) {
  int shape[] = {3};
  Circuit *circuit = create_circuit(shape, 1);
  if (circuit == NULL) {
    expect_true("create_circuit for single-layer hard circuit", 0);
    return;
  }
  HardCircuit *hard = create_hard_circuit(circuit);
  if (hard == NULL) {
    expect_true("create_hard_circuit for single layer", 0);
    destroy_circuit(circuit);
    return;
  }

  expect_true("single-layer hard connections_len", hard->connections_len == 0);
  expect_true("single-layer hard gates_len", hard->gates_len == 0);

  bool input[] = {true, false, true};
  hard_circuit_set_input(hard, input);
  hard_circuit_feed_forward(hard);
  expect_near("single-layer hard output 0", hard->outputs[0], 1.0f, 1e-6f);
  expect_near("single-layer hard output 1", hard->outputs[1], 0.0f, 1e-6f);
  expect_near("single-layer hard output 2", hard->outputs[2], 1.0f, 1e-6f);

  destroy_hard_circuit(hard);
  destroy_circuit(circuit);
}

static void test_ties_choose_lowest_index(void) {
  int shape[] = {2, 1, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for tie test", 0);
    return;
  }

  // Everything is tied at zero, so all selections should fall back to index 0.
  memset(circuit->connection_probabilities, 0,
         circuit->connection_probabilities_len *
             sizeof(*circuit->connection_probabilities));
  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));

  HardCircuit *hard = create_hard_circuit(circuit);
  if (hard == NULL) {
    expect_true("create_hard_circuit for tie test", 0);
    destroy_circuit(circuit);
    return;
  }

  expect_true("tie wire a", hard->connections[0] == 0);
  expect_true("tie wire b", hard->connections[1] == 0);
  expect_true("tie gate", hard->gates[0] == 0);

  bool input[] = {true, true};
  hard_circuit_set_input(hard, input);
  hard_circuit_feed_forward(hard);
  expect_near("tie hidden gate is zero", hard->activations[2], 0.0f, 1e-6f);
  expect_near("tie output", hard->outputs[0], 0.0f, 1e-6f);

  destroy_hard_circuit(hard);
  destroy_circuit(circuit);
}

static void test_hard_circuit_validation(void) {
  expect_true("create_hard_circuit(NULL) returns NULL",
              create_hard_circuit(NULL) == NULL);
  expect_true("create_hard_circuit(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  hard_circuit_feed_forward(NULL);
  expect_true("hard feed_forward(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  hard_circuit_set_input(NULL, NULL);
  expect_true("hard set_input(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  hard_circuit_read_output(NULL, NULL);
  expect_true("hard read_output(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  // Destroying NULL is a no-op.
  destroy_hard_circuit(NULL);
}

int main(void) {
  test_hard_gate_truth_table();
  test_create_hard_circuit_argmax();
  test_single_layer_hard_circuit();
  test_ties_choose_lowest_index();
  test_hard_circuit_validation();
  return test_report();
}
