#include "test_common.h"

#include <string.h>

#include "../src/main.c"

static void test_forward_and_derivatives(void) {
  int shape[] = {2, 2, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for forward test", 0);
    return;
  }

  // Layer 0 -> 1. Gate 0 receives (x0, x1); gate 1 receives (x1, x0).
  // Layer 1 -> output. The single output receives (cell0, cell1).
  float connections[] = {
      1, 0, 0, 1, // gate 0 pins 0/1
      0, 1, 1, 0, // gate 1 pins 0/1
      1, 0, 0, 1, // output pins 0/1
  };
  memcpy(circuit->connection_probabilities, connections, sizeof(connections));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  circuit->cells[0].gate_probabilities.and = 1.0f;
  circuit->cells[1].gate_probabilities.or = 1.0f;
  circuit->cells[2].gate_probabilities.or = 1.0f;

  circuit->inputs[0] = true;
  circuit->inputs[1] = false;
  circuit_feed_forward(circuit);

  expect_near("forward hidden AND(1, 0)", circuit->cells[0].out, 0.0f, 1e-6f);
  expect_near("forward hidden OR(0, 1)", circuit->cells[1].out, 1.0f, 1e-6f);
  expect_near("forward output OR(0, 1)", circuit->outputs[0], 1.0f, 1e-6f);
  expect_near("forward d AND / da", circuit->derivatives[0], 0.0f, 1e-6f);
  expect_near("forward d AND / db", circuit->derivatives[1], 1.0f, 1e-6f);
  expect_near("forward d OR / da", circuit->derivatives[2], 0.0f, 1e-6f);
  expect_near("forward d OR / db", circuit->derivatives[3], 1.0f, 1e-6f);
  expect_near("forward d output / da", circuit->derivatives[4], 0.0f, 1e-6f);
  expect_near("forward d output / db", circuit->derivatives[5], 1.0f, 1e-6f);

  destroy_circuit(circuit);
}

static void test_soft_gate_mixture(void) {
  int shape[] = {2, 1, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for mixture test", 0);
    return;
  }

  float connections[] = {
      1, 0, // hidden pin a -> x0
      0, 1, // hidden pin b -> x1
      1,    // output pin a -> hidden
      1,    // output pin b -> hidden
  };
  memcpy(circuit->connection_probabilities, connections, sizeof(connections));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  circuit->cells[0].gate_probabilities.and = 0.5f;
  circuit->cells[0].gate_probabilities.or = 0.5f;
  circuit->cells[1].gate_probabilities.proj_a = 1.0f;

  circuit->inputs[0] = true;
  circuit->inputs[1] = false;
  circuit_feed_forward(circuit);

  // AND(1, 0) = 0, OR(1, 0) = 1.
  expect_near("mixture output", circuit->outputs[0], 0.5f, 1e-6f);
  expect_near("mixture d/da", circuit->derivatives[0], 0.5f, 1e-6f);
  expect_near("mixture d/db", circuit->derivatives[1], 0.5f, 1e-6f);
  expect_near("output projection d/da", circuit->derivatives[2], 1.0f, 1e-6f);
  expect_near("output projection d/db", circuit->derivatives[3], 0.0f, 1e-6f);

  destroy_circuit(circuit);
}

static void test_single_layer_circuit(void) {
  int shape[] = {3};
  Circuit *circuit = create_circuit(shape, 1);
  if (circuit == NULL) {
    expect_true("create_circuit for single-layer test", 0);
    return;
  }

  circuit->inputs[0] = true;
  circuit->inputs[1] = false;
  circuit->inputs[2] = true;
  circuit_feed_forward(circuit);

  expect_near("single-layer output[0]", circuit->outputs[0], 1.0f, 1e-6f);
  expect_near("single-layer output[1]", circuit->outputs[1], 0.0f, 1e-6f);
  expect_near("single-layer output[2]", circuit->outputs[2], 1.0f, 1e-6f);

  destroy_circuit(circuit);
}

static void test_backpropagation_smoke(void) {
  int shape[] = {2, 1, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for backprop test", 0);
    return;
  }

  // The output gate is proj_a, so it copies the hidden cell, which is a
  // deterministic AND.
  float connections[] = {
      1, 0, // hidden pin a -> x0
      0, 1, // hidden pin b -> x1
      1,    // output pin a -> hidden
      1,    // output pin b -> hidden
  };
  memcpy(circuit->connection_probabilities, connections, sizeof(connections));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  circuit->cells[0].gate_probabilities.and = 1.0f;
  circuit->cells[1].gate_probabilities.proj_a = 1.0f;
  circuit->inputs[0] = true;
  circuit->inputs[1] = false;

  circuit_feed_forward(circuit);
  expect_near("backprop forward before", circuit->outputs[0], 0.0f, 1e-6f);

  float error[] = {-1.0f};
  circuit_back_propergate(circuit, error, 0.0f);

  expect_near("backprop dLoss/d hidden", circuit->gradients[2], -1.0f, 1e-6f);
  expect_near("backprop dLoss/d input0", circuit->gradients[0], 0.0f, 1e-6f);
  expect_near("backprop dLoss/d input1", circuit->gradients[1], -1.0f, 1e-6f);

  circuit_back_propergate(circuit, error, 0.1f);
  circuit_feed_forward(circuit);
  expect_true("backprop moved output toward target",
              circuit->outputs[0] > 0.0f);
  expect_near("backprop hidden pin a normalized",
              circuit->connection_probabilities[0] +
                  circuit->connection_probabilities[1],
              1.0f, 1e-6f);
  expect_near("backprop hidden pin b normalized",
              circuit->connection_probabilities[2] +
                  circuit->connection_probabilities[3],
              1.0f, 1e-6f);

  float gate_sum = 0.0f;
  for (int k = 0; k < 16; ++k) {
    gate_sum += circuit->cells[0].gate_probabilities.data[k];
  }
  expect_near("backprop gate probabilities normalized", gate_sum, 1.0f, 1e-6f);

  float output_gate_sum = 0.0f;
  for (int k = 0; k < 16; ++k) {
    output_gate_sum += circuit->cells[1].gate_probabilities.data[k];
  }
  expect_near("backprop output gate probabilities normalized", output_gate_sum,
              1.0f, 1e-6f);

  destroy_circuit(circuit);
}

static void test_invalid_params(void) {
  circuit_feed_forward(NULL);
  expect_true("feed_forward(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  circuit_back_propergate(NULL, NULL, 0.0f);
  expect_true("back_propergate(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);
}

int main(void) {
  test_forward_and_derivatives();
  test_soft_gate_mixture();
  test_single_layer_circuit();
  test_backpropagation_smoke();
  test_invalid_params();
  return test_report();
}
