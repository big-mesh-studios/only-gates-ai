#include "test_common.h"

#include <string.h>

#include "../src/main.c"

static float dataset_loss(Circuit *circuit, const bool *inputs,
                          const float *targets, int sample_count) {
  float total = 0.0f;
  for (int sample = 0; sample < sample_count; ++sample) {
    for (int i = 0; i < circuit->inputs_len; ++i) {
      circuit->inputs[i] = inputs[sample * circuit->inputs_len + i];
    }
    circuit_feed_forward(circuit);
    for (int i = 0; i < circuit->outputs_len; ++i) {
      float difference =
          circuit->outputs[i] - targets[sample * circuit->outputs_len + i];
      total += 0.5f * difference * difference;
    }
  }
  return total / (float)sample_count;
}

static void test_trainer_trains_or(void) {
  int shape[] = {2, 1, 1};
  Circuit *circuit = create_circuit(shape, 3);
  if (circuit == NULL) {
    expect_true("create_circuit for trainer test", 0);
    return;
  }

  float connections[] = {
      0.6f, 0.4f, // hidden pin a
      0.4f, 0.6f, // hidden pin b
      1.0f,       // output pin a
      1.0f,       // output pin b
  };
  memcpy(circuit->connection_probabilities, connections, sizeof(connections));

  memset(circuit->cells, 0, circuit->cell_len * sizeof(Cell));
  float *gate = circuit->cells[0].gate_probabilities.data;
  gate[1] = 0.25f;  // nor
  gate[6] = 0.25f;  // xor
  gate[8] = 0.25f;  // and
  gate[14] = 0.25f; // or

  const bool inputs[] = {
      false, false, // 0, 0 -> 0
      false, true,  // 0, 1 -> 1
      true,  false, // 1, 0 -> 1
      true,  true,  // 1, 1 -> 1
  };
  const float targets[] = {0.0f, 1.0f, 1.0f, 1.0f};
  const int sample_count = 4;
  const float learning_rate = 0.05f;

  // Compute the expected first-step loss before creating the trainer.
  for (int i = 0; i < circuit->inputs_len; ++i) {
    circuit->inputs[i] = inputs[i];
  }
  circuit_feed_forward(circuit);
  float first_expected = 0.0f;
  for (int i = 0; i < circuit->outputs_len; ++i) {
    float difference = circuit->outputs[i] - targets[i];
    first_expected += 0.5f * difference * difference;
  }

  Trainer *trainer =
      create_trainer(circuit, inputs, targets, sample_count, learning_rate);
  if (trainer == NULL) {
    expect_true("create_trainer", 0);
    destroy_circuit(circuit);
    return;
  }

  const float first_loss = trainer_step(trainer);
  expect_near("trainer first-step loss", first_loss, first_expected, 1e-6f);
  expect_true("trainer advances after a step", trainer->index == 1);

  const float initial_loss =
      dataset_loss(circuit, inputs, targets, sample_count);

  for (int step = 0; step < sample_count * 500; ++step) {
    float step_loss = trainer_step(trainer);
    expect_true("trainer loss is finite and non-negative",
                isfinite(step_loss) && step_loss >= 0.0f);
    if (test_failures != 0) {
      break;
    }
  }

  const float final_loss = dataset_loss(circuit, inputs, targets, sample_count);
  printf("trainer average loss: %.8f -> %.8f\n", initial_loss, final_loss);
  expect_true("trainer reduces average loss", final_loss < initial_loss);

  trainer_set_learning_rate(trainer, 0.01f);
  trainer_reset(trainer);
  expect_true("trainer reset returns to first sample", trainer->index == 0);
  expect_near("trainer set_learning_rate", trainer->learning_rate, 0.01f,
              1e-6f);

  destroy_trainer(trainer);
  destroy_circuit(circuit);
}

static void test_trainer_validation(void) {
  expect_true("create_trainer(NULL) returns NULL",
              create_trainer(NULL, NULL, NULL, 0, 0.0f) == NULL);
  expect_true("create_trainer(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);

  expect_near("trainer_step(NULL) returns zero", trainer_step(NULL), 0.0f,
              0.0f);
  expect_true("trainer_step(NULL) sets invalid params",
              get_error() == ERR_INVALID_PARAMS);
}

int main(void) {
  test_trainer_trains_or();
  test_trainer_validation();
  return test_report();
}
