#include "test_common.h"

#include <string.h>

#include "../src/main.c"

static float loss(Circuit *circuit, float target) {
  circuit_feed_forward(circuit);
  float difference = circuit->outputs[0] - target;
  return 0.5f * difference * difference;
}

static void test_finite_difference_gradients(void) {
  int shape[] = {2, 1, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for gradient test", 0);
    return;
  }

  float *connections = circuit->connection_probabilities;
  float connections_backup[] = {
      0.7f, 0.3f, // hidden pin a
      0.2f, 0.8f, // hidden pin b
      1.0f,       // output pin a
      1.0f,       // output pin b
  };
  memcpy(connections, connections_backup, sizeof(connections_backup));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  float *gate = circuit->cells[0].gate_probabilities.data;
  gate[1] = 0.35f;  // nor
  gate[6] = 0.15f;  // xor
  gate[8] = 0.20f;  // and
  gate[11] = 0.10f; // imp
  gate[14] = 0.20f; // or

  float *output_gate = circuit->cells[1].gate_probabilities.data;
  output_gate[6] = 0.25f;  // xor
  output_gate[8] = 0.25f;  // and
  output_gate[12] = 0.25f; // proj_a
  output_gate[14] = 0.25f; // or

  float gate_backup[32];
  memcpy(gate_backup, gate, 16 * sizeof(float));
  memcpy(gate_backup + 16, output_gate, 16 * sizeof(float));

  circuit->inputs[0] = true;
  circuit->inputs[1] = false;

  const float target = 1.0f;
  const float eps = 1e-3f;
  const float learning_rate = 1e-3f;

  float numerical_connection_gradients[6] = {0};
  float numerical_gate_gradients[32] = {0};

  // Compute ambient gradients by perturbing without renormalizing.
  for (int group = 0; group < 2; ++group) {
    const int start = group * 2;
    for (int i = 0; i < 2; ++i) {
      memcpy(connections, connections_backup, sizeof(connections_backup));
      connections[start + i] += eps;
      float plus = loss(circuit, target);

      memcpy(connections, connections_backup, sizeof(connections_backup));
      connections[start + i] -= eps;
      float minus = loss(circuit, target);

      numerical_connection_gradients[start + i] = (plus - minus) / (2 * eps);
    }
  }

  for (int i = 0; i < 32; ++i) {
    memcpy(connections, connections_backup, sizeof(connections_backup));
    memcpy(gate, gate_backup, 16 * sizeof(float));
    memcpy(output_gate, gate_backup + 16, 16 * sizeof(float));

    float *parameter = i < 16 ? gate + i : output_gate + (i - 16);
    *parameter += eps;
    float plus = loss(circuit, target);

    memcpy(connections, connections_backup, sizeof(connections_backup));
    memcpy(gate, gate_backup, 16 * sizeof(float));
    memcpy(output_gate, gate_backup + 16, 16 * sizeof(float));

    *parameter -= eps;
    float minus = loss(circuit, target);

    numerical_gate_gradients[i] = (plus - minus) / (2 * eps);
  }

  // Restore the parameters and apply one analytic update.
  memcpy(connections, connections_backup, sizeof(connections_backup));
  memcpy(gate, gate_backup, 16 * sizeof(float));
  memcpy(output_gate, gate_backup + 16, 16 * sizeof(float));
  circuit_feed_forward(circuit);

  const float output_before = circuit->outputs[0];
  float error[] = {output_before - target};
  circuit_back_propergate(circuit, error, learning_rate);

  const float loss_before =
      0.5f * (output_before - target) * (output_before - target);
  const float loss_after = loss(circuit, target);
  expect_true("gradient step reduces loss", loss_after < loss_before);

  for (int group = 0; group < 2; ++group) {
    const int start = group * 2;
    const float mean_gradient = (numerical_connection_gradients[start] +
                                 numerical_connection_gradients[start + 1]) /
                                2.0f;
    float predicted[2];
    for (int i = 0; i < 2; ++i) {
      predicted[i] =
          connections_backup[start + i] -
          learning_rate *
              (numerical_connection_gradients[start + i] - mean_gradient);
    }
    normalize_probabilities(predicted, 2);

    char name[64];
    snprintf(name, sizeof(name), "connection[%d]", start);
    expect_near(name, connections[start], predicted[0], 2e-4f);
    snprintf(name, sizeof(name), "connection[%d]", start + 1);
    expect_near(name, connections[start + 1], predicted[1], 2e-4f);
  }

  for (int cell = 0; cell < 2; ++cell) {
    float *cell_gate = circuit->cells[cell].gate_probabilities.data;
    float mean_gate_gradient = 0.0f;
    for (int i = 0; i < 16; ++i) {
      mean_gate_gradient += numerical_gate_gradients[cell * 16 + i];
    }
    mean_gate_gradient /= 16.0f;

    float predicted_gate[16];
    for (int i = 0; i < 16; ++i) {
      predicted_gate[i] =
          gate_backup[cell * 16 + i] -
          learning_rate *
              (numerical_gate_gradients[cell * 16 + i] - mean_gate_gradient);
    }
    normalize_probabilities(predicted_gate, 16);

    for (int i = 0; i < 16; ++i) {
      char name[64];
      snprintf(name, sizeof(name), "gate[%d][%d]", cell, i);
      expect_near(name, cell_gate[i], predicted_gate[i], 2e-4f);
    }
  }

  destroy_circuit(circuit);
}

int main(void) {
  test_finite_difference_gradients();
  return test_report();
}
