#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  union {
    float data[16];
    struct {
      /**
       * Probability for the zero gate: returns `false`
       */
      float zero;
      /**
       * Probability for the nor gate: returns `!(a || b)`
       */
      float nor;
      /**
       * Probability for the left not implication gate: returns `a && !b` (aka
       * `!(a <- b)`)
       */
      float lnt;
      /**
       * Probability for the not a gate: returns `!a`
       */
      float not_a;
      /**
       * Probability for the right not implication gate: returns `!a && b` (aka
       * `!(a -> b)`)
       */
      float rnt;
      /**
       * Probability for the not b gate: returns `!b`
       */
      float not_b;
      /**
       * Probability for the xor gate: returns `(a && !b) || (!a && b)` (aka `a
       * xor b`)
       */
      float xor ;
      /**
       * Probability for the nand gate: returns `!(a && b)`
       */
      float nand;
      /**
       * Probability for the and gate: returns `a && b`
       */
      float and;
      /**
       * Probability for the xnor gate: returns `!(a xor b)`
       */
      float xnor;
      /**
       * Probability for the proj b gate: returns `b`
       */
      float proj_b;
      /**
       * Probability for the implies gate: returns `!a || b` (aka `a -> b`)
       */
      float imp;
      /**
       * Probability for the proj a gate: returns `a`
       */
      float proj_a;
      /**
       * Probability for the converse implies gate: returns `a || !b` (aka `a <-
       * b`)
       */
      float cimp;
      /**
       * Probability for the or gate: returns `a || b`
       */
      float or ;
      /**
       * Probability for the one gate: returns `true`
       */
      float one;
    };
  };
} GateProbabilities;

typedef struct {
  GateProbabilities gate_probabilities;
  float out;
} Cell;

typedef struct {
  bool *inputs;
  int inputs_len;
  float *outputs;
  int outputs_len;
  int *shape;
  int shape_len;
  Cell *cells;
  int cell_len;
  /**
   * Wire probabilities for each layer transition, concatenated in layer order.
   * For a transition with `P` outputs in the previous layer and `C` gates in
   * the current layer, the two pins of gate `g` occupy `2 * P` consecutive
   * floats:
   *
   *   connection_probabilities[base + (2 * g + pin) * P + j]
   *
   * is the probability of connecting input pin `pin` of gate `g` to output `j`
   * of the previous layer. Each pin distribution is expected to be normalized.
   */
  float *connection_probabilities;
  int connection_probabilities_len;
  /**
   * Scratch buffer holding the activations of every layer, concatenated in
   * layer order. The first `shape[0]` entries are the inputs converted to
   * floats, followed by every hidden layer and finally the output layer.
   *
   * Allocated up-front so that feed-forward and back-propagation never need to
   * allocate memory while training.
   */
  float *activations;
  int activations_len;
  /**
   * Local derivatives for back-propagation, two floats per gate in layer order.
   * For gate `g`, `derivatives[2 * g]` is `d(out) / d(a)` and
   * `derivatives[2 * g + 1]` is `d(out) / d(b)`, where `a` and `b` are the two
   * input pins of that gate. Hidden cells come first in the same order as
   * `cells`, followed by the output layer.
   */
  float *derivatives;
  int derivatives_len;
  /**
   * Scratch buffer for backward-pass deltas (`dLoss / d activation`), using the
   * same layout as `activations`. Allocated up-front so back-propagation never
   * allocates memory.
   */
  float *gradients;
  int gradients_len;
} Circuit;

/**
 * Deterministic version of a trained `Circuit`, analogous to diff-logic's
 * evaluation mode. Every pin reads a single previous-layer neuron (the wire
 * with the highest probability) and every hidden cell uses a single gate (the
 * gate with the highest probability), so forward evaluation performs no
 * probability arithmetic.
 *
 * The output layer has no gate, matching `Circuit`. It averages its two
 * selected inputs, so outputs are 0, 0.5 or 1.
 *
 * A hard circuit owns copies of everything it needs and stays valid after the
 * source `Circuit` changes or is destroyed.
 */
typedef struct {
  bool *inputs;
  int inputs_len;
  float *outputs;
  int outputs_len;
  int *shape;
  int shape_len;
  /**
   * Selected wire indices for each layer transition, concatenated in layer
   * order. For a transition with `C` gates:
   *
   *   connections[base + (2 * gate + pin)] = previous-layer index
   *
   * where `pin` is 0 for input `a` and 1 for input `b`.
   */
  int *connections;
  int connections_len;
  /**
   * Selected gate index (0..15) per hidden cell, using the same ordering as
   * `Circuit.cells`.
   */
  int *gates;
  int gates_len;
  /** Activation scratch for every layer, matching `Circuit.activations`. */
  float *activations;
  int activations_len;
} HardCircuit;

/**
 * Iterates over a bulk dataset one sample per `trainer_step`.
 *
 * `inputs` points to `data_count * circuit->inputs_len` booleans stored
 * sample-by-sample. `outputs` points to `data_count * circuit->outputs_len`
 * floats with the same sample ordering. The arrays are owned by the caller and
 * must stay alive while the trainer exists.
 */
typedef struct {
  Circuit *circuit;
  const bool *inputs;
  const float *outputs;
  int data_count;
  int index;
  float learning_rate;
  float *error;
} Trainer;

const int ERR_NONE = 0;
const int ERR_OUT_OF_MEMORY = 1;
const int ERR_INVALID_SHAPE = 2;
const int ERR_INVALID_PARAMS = 3;

static int _last_error = ERR_NONE;

static inline void clear_error(void) { _last_error = ERR_NONE; }

static inline void set_error(int error_code) { _last_error = error_code; }

EMSCRIPTEN_KEEPALIVE
int get_error() { return _last_error; }

/**
 * Evaluates gate `gate` and its partial derivatives with respect to inputs `a`
 * and `b`. Using one switch for both values keeps the arithmetic inline in the
 * feed-forward loop instead of making sixteen indirect function calls.
 */
static inline void soft_gate_forward_and_derivative(int gate, float a, float b,
                                                    float *value, float *da,
                                                    float *db) {
  switch (gate) {
  case 0: // zero
    *value = 0.0f;
    *da = 0.0f;
    *db = 0.0f;
    break;
  case 1: // nor: 1 - a - b + a * b
    *value = 1.0f - (a + b - a * b);
    *da = b - 1.0f;
    *db = a - 1.0f;
    break;
  case 2: // lnt: a - a * b
    *value = a - a * b;
    *da = 1.0f - b;
    *db = -a;
    break;
  case 3: // not_a: 1 - a
    *value = 1.0f - a;
    *da = -1.0f;
    *db = 0.0f;
    break;
  case 4: // rnt: b - a * b
    *value = b - a * b;
    *da = -b;
    *db = 1.0f - a;
    break;
  case 5: // not_b: 1 - b
    *value = 1.0f - b;
    *da = 0.0f;
    *db = -1.0f;
    break;
  case 6: // xor: a + b - 2 * a * b
    *value = a + b - 2.0f * a * b;
    *da = 1.0f - 2.0f * b;
    *db = 1.0f - 2.0f * a;
    break;
  case 7: // nand: 1 - a * b
    *value = 1.0f - a * b;
    *da = -b;
    *db = -a;
    break;
  case 8: // and: a * b
    *value = a * b;
    *da = b;
    *db = a;
    break;
  case 9: // xnor: 1 - a - b + 2 * a * b
    *value = 1.0f - (a + b - 2.0f * a * b);
    *da = 2.0f * b - 1.0f;
    *db = 2.0f * a - 1.0f;
    break;
  case 10: // proj_b: b
    *value = b;
    *da = 0.0f;
    *db = 1.0f;
    break;
  case 11: // imp: 1 - a + a * b
    *value = 1.0f - a + a * b;
    *da = b - 1.0f;
    *db = a;
    break;
  case 12: // proj_a: a
    *value = a;
    *da = 1.0f;
    *db = 0.0f;
    break;
  case 13: // cimp: 1 - b + a * b
    *value = 1.0f - b + a * b;
    *da = b;
    *db = a - 1.0f;
    break;
  case 14: // or: a + b - a * b
    *value = a + b - a * b;
    *da = 1.0f - b;
    *db = 1.0f - a;
    break;
  case 15: // one
    *value = 1.0f;
    *da = 0.0f;
    *db = 0.0f;
    break;
  default:
    *value = 0.0f;
    *da = 0.0f;
    *db = 0.0f;
    break;
  }
}

void softmax(const float *input, float *output, int size) {
  if (size <= 0) {
    return;
  }
  float max_val = input[0];
  for (int i = 1; i < size; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }
  for (int i = 0; i < size; i++) {
    output[i] /= sum;
  }
}

/**
 * Projects a group of values back onto the probability simplex after a
 * gradient step: negative values are clamped to zero and the rest are
 * renormalized. If the step zeroed everything, fall back to a uniform
 * distribution.
 */
static void normalize_probabilities(float *probabilities, int size) {
  if (size <= 0) {
    return;
  }
  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    if (probabilities[i] < 0.0f) {
      probabilities[i] = 0.0f;
    }
    sum += probabilities[i];
  }
  if (sum > 0.0f) {
    const float inverse_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
      probabilities[i] *= inverse_sum;
    }
  } else {
    const float uniform = 1.0f / (float)size;
    for (int i = 0; i < size; ++i) {
      probabilities[i] = uniform;
    }
  }
}

EMSCRIPTEN_KEEPALIVE
Circuit *create_circuit(int shape[], int shape_len) {
  clear_error();
  if (shape_len <= 0) {
    set_error(ERR_INVALID_SHAPE);
    return NULL;
  }
  for (int i = 0; i < shape_len; ++i) {
    if (shape[i] <= 0) {
      set_error(ERR_INVALID_SHAPE);
      return NULL;
    }
  }
  Circuit *result = malloc(sizeof(Circuit));
  bool inputs_allocated = false;
  bool outputs_allocated = false;
  bool shape_allocated = false;
  bool cells_allocated = false;
  bool connection_probabilities_allocated = false;
  bool activations_allocated = false;
  bool derivatives_allocated = false;
  bool gradients_allocated = false;
  if (result == NULL) {
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
#define memory_cleanup()                                                       \
  {                                                                            \
    if (inputs_allocated) {                                                    \
      free(result->inputs);                                                    \
    }                                                                          \
    if (outputs_allocated) {                                                   \
      free(result->outputs);                                                   \
    }                                                                          \
    if (shape_allocated) {                                                     \
      free(result->shape);                                                     \
    }                                                                          \
    if (cells_allocated) {                                                     \
      free(result->cells);                                                     \
    }                                                                          \
    if (connection_probabilities_allocated) {                                  \
      free(result->connection_probabilities);                                  \
    }                                                                          \
    if (activations_allocated) {                                               \
      free(result->activations);                                               \
    }                                                                          \
    if (derivatives_allocated) {                                               \
      free(result->derivatives);                                               \
    }                                                                          \
    if (gradients_allocated) {                                                 \
      free(result->gradients);                                                 \
    }                                                                          \
    free(result);                                                              \
  }
  result->inputs = malloc(shape[0] * sizeof((*result->inputs)));
  if (result->inputs == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  inputs_allocated = true;
  result->outputs = malloc(shape[shape_len - 1] * sizeof((*result->outputs)));
  if (result->outputs == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  outputs_allocated = true;
  result->shape = malloc(shape_len * sizeof(*shape));
  if (result->shape == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  shape_allocated = true;
  memcpy(result->shape, shape, shape_len * sizeof(*shape));
  result->shape_len = shape_len;
  result->inputs_len = shape[0];
  result->outputs_len = shape[shape_len - 1];
  int cell_len = 0;
  for (int i = 1; i < shape_len - 1; ++i) {
    cell_len += shape[i];
  }
  if (cell_len > 0) {
    result->cells = malloc(cell_len * sizeof(Cell));
    if (result->cells == NULL) {
      memory_cleanup();
      set_error(ERR_OUT_OF_MEMORY);
      return NULL;
    }
    cells_allocated = true;
  } else {
    result->cells = NULL;
  }
  result->cell_len = cell_len;
  int connection_len = 0;
  for (int i = 1; i < shape_len; ++i) {
    int num_pre_gates = shape[i - 1];
    int num_gates = shape[i];
    connection_len += (num_pre_gates * num_gates) << 1;
  }
  if (connection_len > 0) {
    result->connection_probabilities =
        malloc(connection_len * sizeof(*result->connection_probabilities));
    if (result->connection_probabilities == NULL) {
      memory_cleanup();
      set_error(ERR_OUT_OF_MEMORY);
      return NULL;
    }
    connection_probabilities_allocated = true;
  } else {
    result->connection_probabilities = NULL;
  }
  result->connection_probabilities_len = connection_len;
  int activations_len = 0;
  for (int i = 0; i < shape_len; ++i) {
    activations_len += shape[i];
  }
  result->activations = malloc(activations_len * sizeof(*result->activations));
  if (result->activations == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  activations_allocated = true;
  result->activations_len = activations_len;
  int derivatives_len = (cell_len + shape[shape_len - 1]) << 1;
  result->derivatives = malloc(derivatives_len * sizeof(*result->derivatives));
  if (result->derivatives == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  derivatives_allocated = true;
  result->derivatives_len = derivatives_len;
  result->gradients = malloc(activations_len * sizeof(*result->gradients));
  if (result->gradients == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  gradients_allocated = true;
  result->gradients_len = activations_len;
#undef memory_cleanup
  return result;
}

EMSCRIPTEN_KEEPALIVE
void destroy_circuit(Circuit *circuit) {
  free(circuit->inputs);
  free(circuit->outputs);
  free(circuit->shape);
  free(circuit->cells);
  free(circuit->connection_probabilities);
  free(circuit->activations);
  free(circuit->derivatives);
  free(circuit->gradients);
  free(circuit);
}

EMSCRIPTEN_KEEPALIVE
void circuit_set_input(Circuit *circuit, bool input[]) {}

EMSCRIPTEN_KEEPALIVE
void circuit_read_output(Circuit *circuit, float output[]) {}

EMSCRIPTEN_KEEPALIVE
void circuit_feed_forward(Circuit *circuit) {
  if (circuit == NULL || circuit->shape == NULL || circuit->inputs == NULL ||
      circuit->activations == NULL || circuit->derivatives == NULL ||
      circuit->shape_len <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();

  const int shape_len = circuit->shape_len;
  const int *shape = circuit->shape;
  float *activations = circuit->activations;
  float *derivatives = circuit->derivatives;
  const int input_len = shape[0];

  // Seed the activation buffer with the boolean inputs as floats.
  for (int i = 0; i < input_len; ++i) {
    activations[i] = circuit->inputs[i] ? 1.0f : 0.0f;
  }

  int prev_offset = 0;
  int prev_len = input_len;
  int connection_offset = 0;
  int cell_offset = 0;
  int derivative_offset = 0;

  for (int layer = 1; layer < shape_len; ++layer) {
    const int curr_len = shape[layer];
    const int curr_offset = prev_offset + prev_len;
    const float *prev = activations + prev_offset;
    float *curr = activations + curr_offset;
    const float *connections =
        circuit->connection_probabilities + connection_offset;
    const bool is_output_layer = (layer == shape_len - 1);

    for (int gate = 0; gate < curr_len; ++gate) {
      // Each pin has one probability per output in the previous layer. The
      // probabilities are expected to already be normalized (e.g. softmaxed).
      const float *wire_a = connections + (gate << 1) * prev_len;
      const float *wire_b = wire_a + prev_len;
      float a = 0.0f;
      float b = 0.0f;
      for (int j = 0; j < prev_len; ++j) {
        a += wire_a[j] * prev[j];
        b += wire_b[j] * prev[j];
      }

      float out;
      float d_a;
      float d_b;
      if (is_output_layer) {
        // The output layer has no gate, so combine its two pins evenly.
        out = 0.5f * (a + b);
        d_a = 0.5f;
        d_b = 0.5f;
      } else {
        Cell *cell = &circuit->cells[cell_offset + gate];
        const float *gate_probabilities = cell->gate_probabilities.data;
        float cell_out = 0.0f;
        float cell_d_a = 0.0f;
        float cell_d_b = 0.0f;
        for (int k = 0; k < 16; ++k) {
          const float probability = gate_probabilities[k];
          if (probability == 0.0f) {
            continue;
          }
          float value;
          float gate_d_a;
          float gate_d_b;
          soft_gate_forward_and_derivative(k, a, b, &value, &gate_d_a,
                                           &gate_d_b);
          cell_out += probability * value;
          cell_d_a += probability * gate_d_a;
          cell_d_b += probability * gate_d_b;
        }
        cell->out = cell_out;
        out = cell_out;
        d_a = cell_d_a;
        d_b = cell_d_b;
      }

      curr[gate] = out;
      // Cache the chain-rule terms back-propagation needs from this gate.
      derivatives[derivative_offset++] = d_a;
      derivatives[derivative_offset++] = d_b;
    }

    if (!is_output_layer) {
      cell_offset += curr_len;
    }
    connection_offset += (prev_len * curr_len) << 1;
    prev_offset = curr_offset;
    prev_len = curr_len;
  }

  const int output_len = shape[shape_len - 1];
  const float *output_activations = activations + prev_offset;
  for (int i = 0; i < output_len; ++i) {
    circuit->outputs[i] = output_activations[i];
  }
}

/** Returns the index of the largest value. Ties resolve to the lowest index. */
static int argmax_probability(const float *values, int size) {
  int best = 0;
  for (int i = 1; i < size; ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

/** Evaluates gate `gate` for boolean inputs, using the project's gate order. */
static inline bool hard_gate_forward(int gate, bool a, bool b) {
  switch (gate) {
  case 0: // zero
    return false;
  case 1: // nor
    return !(a || b);
  case 2: // lnt: a && !b
    return a && !b;
  case 3: // not_a
    return !a;
  case 4: // rnt: !a && b
    return !a && b;
  case 5: // not_b
    return !b;
  case 6: // xor
    return a != b;
  case 7: // nand
    return !(a && b);
  case 8: // and
    return a && b;
  case 9: // xnor
    return a == b;
  case 10: // proj_b
    return b;
  case 11: // imp: !a || b
    return !a || b;
  case 12: // proj_a
    return a;
  case 13: // cimp: a || !b
    return a || !b;
  case 14: // or
    return a || b;
  case 15: // one
    return true;
  default:
    return false;
  }
}

/**
 * Builds a deterministic `HardCircuit` by taking the highest-probability wire
 * for every pin and the highest-probability gate for every hidden cell. Ties
 * resolve to the lowest index, which keeps the conversion deterministic.
 */
EMSCRIPTEN_KEEPALIVE
HardCircuit *create_hard_circuit(Circuit *circuit) {
  clear_error();
  if (circuit == NULL || circuit->shape == NULL || circuit->shape_len <= 0 ||
      circuit->inputs == NULL || circuit->outputs == NULL ||
      (circuit->connection_probabilities_len > 0 &&
       circuit->connection_probabilities == NULL) ||
      (circuit->cell_len > 0 && circuit->cells == NULL)) {
    set_error(ERR_INVALID_PARAMS);
    return NULL;
  }

  HardCircuit *result = calloc(1, sizeof(*result));
  if (result == NULL) {
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }

  const int shape_len = circuit->shape_len;
  const int input_len = circuit->shape[0];
  const int output_len = circuit->shape[shape_len - 1];
  int activations_len = 0;
  for (int i = 0; i < shape_len; ++i) {
    activations_len += circuit->shape[i];
  }
  const int connections_len = (activations_len - input_len) << 1;
  const int gates_len = circuit->cell_len;

#define hard_memory_cleanup()                                                  \
  {                                                                            \
    free(result->inputs);                                                      \
    free(result->outputs);                                                     \
    free(result->shape);                                                       \
    free(result->connections);                                                 \
    free(result->gates);                                                       \
    free(result->activations);                                                 \
    free(result);                                                              \
  }

  result->inputs = malloc(input_len * sizeof(*result->inputs));
  result->outputs = malloc(output_len * sizeof(*result->outputs));
  result->shape = malloc(shape_len * sizeof(*result->shape));
  result->activations = malloc(activations_len * sizeof(*result->activations));
  if (connections_len > 0) {
    result->connections =
        malloc(connections_len * sizeof(*result->connections));
  }
  if (gates_len > 0) {
    result->gates = malloc(gates_len * sizeof(*result->gates));
  }
  if (result->inputs == NULL || result->outputs == NULL ||
      result->shape == NULL || result->activations == NULL ||
      (connections_len > 0 && result->connections == NULL) ||
      (gates_len > 0 && result->gates == NULL)) {
    hard_memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  memcpy(result->shape, circuit->shape, shape_len * sizeof(*result->shape));
  result->inputs_len = input_len;
  result->outputs_len = output_len;
  result->shape_len = shape_len;
  result->connections_len = connections_len;
  result->gates_len = gates_len;
  result->activations_len = activations_len;

  int connection_offset = 0;
  int hard_connection_offset = 0;
  int cell_offset = 0;
  int gate_offset = 0;
  for (int layer = 1; layer < shape_len; ++layer) {
    const int prev_len = circuit->shape[layer - 1];
    const int curr_len = circuit->shape[layer];
    const float *layer_connections =
        circuit->connection_probabilities + connection_offset;
    int *hard_connections = result->connections + hard_connection_offset;
    const bool is_output_layer = (layer == shape_len - 1);

    for (int gate = 0; gate < curr_len; ++gate) {
      for (int pin = 0; pin < 2; ++pin) {
        const float *probabilities =
            layer_connections + ((gate << 1) + pin) * prev_len;
        hard_connections[(gate << 1) + pin] =
            argmax_probability(probabilities, prev_len);
      }
    }

    if (!is_output_layer) {
      for (int gate = 0; gate < curr_len; ++gate) {
        const float *probabilities =
            circuit->cells[cell_offset + gate].gate_probabilities.data;
        result->gates[gate_offset + gate] =
            argmax_probability(probabilities, 16);
      }
      cell_offset += curr_len;
      gate_offset += curr_len;
    }

    connection_offset += (prev_len * curr_len) << 1;
    hard_connection_offset += curr_len << 1;
  }
#undef hard_memory_cleanup

  return result;
}

EMSCRIPTEN_KEEPALIVE
void destroy_hard_circuit(HardCircuit *hard_circuit) {
  if (hard_circuit == NULL) {
    return;
  }
  free(hard_circuit->inputs);
  free(hard_circuit->outputs);
  free(hard_circuit->shape);
  free(hard_circuit->connections);
  free(hard_circuit->gates);
  free(hard_circuit->activations);
  free(hard_circuit);
}

EMSCRIPTEN_KEEPALIVE
void hard_circuit_set_input(HardCircuit *hard_circuit, bool input[]) {
  if (hard_circuit == NULL || input == NULL || hard_circuit->inputs == NULL ||
      hard_circuit->inputs_len <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();
  memcpy(hard_circuit->inputs, input,
         hard_circuit->inputs_len * sizeof(*hard_circuit->inputs));
}

EMSCRIPTEN_KEEPALIVE
void hard_circuit_read_output(HardCircuit *hard_circuit, float output[]) {
  if (hard_circuit == NULL || output == NULL || hard_circuit->outputs == NULL ||
      hard_circuit->outputs_len <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();
  memcpy(output, hard_circuit->outputs,
         hard_circuit->outputs_len * sizeof(*hard_circuit->outputs));
}

/**
 * Runs a deterministic forward pass. Hidden gates read their single selected
 * wire and evaluate a single gate; the output layer averages its two selected
 * inputs.
 */
EMSCRIPTEN_KEEPALIVE
void hard_circuit_feed_forward(HardCircuit *hard_circuit) {
  if (hard_circuit == NULL || hard_circuit->shape == NULL ||
      hard_circuit->shape_len <= 0 || hard_circuit->inputs == NULL ||
      hard_circuit->outputs == NULL || hard_circuit->activations == NULL ||
      (hard_circuit->connections_len > 0 &&
       hard_circuit->connections == NULL) ||
      (hard_circuit->gates_len > 0 && hard_circuit->gates == NULL)) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();

  const int shape_len = hard_circuit->shape_len;
  const int *shape = hard_circuit->shape;
  float *activations = hard_circuit->activations;
  const int input_len = shape[0];

  for (int i = 0; i < input_len; ++i) {
    activations[i] = hard_circuit->inputs[i] ? 1.0f : 0.0f;
  }

  int prev_offset = 0;
  int prev_len = input_len;
  int connection_offset = 0;
  int gate_offset = 0;

  for (int layer = 1; layer < shape_len; ++layer) {
    const int curr_len = shape[layer];
    const int curr_offset = prev_offset + prev_len;
    const float *prev = activations + prev_offset;
    float *curr = activations + curr_offset;
    const int *connections = hard_circuit->connections + connection_offset;
    const bool is_output_layer = (layer == shape_len - 1);

    for (int gate = 0; gate < curr_len; ++gate) {
      const bool a = prev[connections[(gate << 1)]] != 0.0f;
      const bool b = prev[connections[(gate << 1) + 1]] != 0.0f;
      if (is_output_layer) {
        curr[gate] = 0.5f * ((a ? 1.0f : 0.0f) + (b ? 1.0f : 0.0f));
      } else {
        curr[gate] =
            hard_gate_forward(hard_circuit->gates[gate_offset + gate], a, b)
                ? 1.0f
                : 0.0f;
      }
    }

    if (!is_output_layer) {
      gate_offset += curr_len;
    }
    connection_offset += curr_len << 1;
    prev_offset = curr_offset;
    prev_len = curr_len;
  }

  const int output_len = shape[shape_len - 1];
  const float *output_activations = activations + prev_offset;
  memcpy(hard_circuit->outputs, output_activations,
         output_len * sizeof(*hard_circuit->outputs));
}

EMSCRIPTEN_KEEPALIVE
void circuit_back_propergate(Circuit *circuit, float error[],
                             float learning_rate) {
  if (circuit == NULL || error == NULL || circuit->shape == NULL ||
      circuit->activations == NULL || circuit->derivatives == NULL ||
      circuit->gradients == NULL || circuit->shape_len <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();

  const int shape_len = circuit->shape_len;
  const int *shape = circuit->shape;
  const int last_layer = shape_len - 1;
  const int output_len = shape[last_layer];
  const int cell_len = circuit->cell_len;

  // Seed the output layer's deltas with dLoss / dOutput.
  int curr_offset = circuit->activations_len - output_len;
  for (int i = 0; i < output_len; ++i) {
    circuit->gradients[curr_offset + i] = error[i];
  }

  int connection_offset = circuit->connection_probabilities_len;
  int hidden_cell_offset = cell_len;

  for (int layer = last_layer; layer >= 1; --layer) {
    const int prev_len = shape[layer - 1];
    const int curr_len = shape[layer];
    const int prev_offset = curr_offset - prev_len;
    const bool is_output_layer = (layer == last_layer);
    const float *prev = circuit->activations + prev_offset;
    const float *curr_deltas = circuit->gradients + curr_offset;
    float *prev_deltas = circuit->gradients + prev_offset;

    connection_offset -= (prev_len * curr_len) << 1;
    float *connections = circuit->connection_probabilities + connection_offset;

    if (!is_output_layer) {
      hidden_cell_offset -= curr_len;
    }

    for (int j = 0; j < prev_len; ++j) {
      prev_deltas[j] = 0.0f;
    }

    for (int gate = 0; gate < curr_len; ++gate) {
      const float grad_out = curr_deltas[gate];
      const int derivative_index = is_output_layer
                                       ? ((cell_len + gate) << 1)
                                       : ((hidden_cell_offset + gate) << 1);
      const float grad_a = grad_out * circuit->derivatives[derivative_index];
      const float grad_b =
          grad_out * circuit->derivatives[derivative_index + 1];

      float *wire_a = connections + (gate << 1) * prev_len;
      float *wire_b = wire_a + prev_len;

      // Recompute this gate's pins, accumulate deltas for the previous layer
      // and collect the raw wire gradients. The original connection values are
      // used for gradients and pin values, before they are overwritten.
      float a = 0.0f;
      float b = 0.0f;
      float gradient_sum_a = 0.0f;
      float gradient_sum_b = 0.0f;
      for (int j = 0; j < prev_len; ++j) {
        const float connection_a = wire_a[j];
        const float connection_b = wire_b[j];
        const float prev_value = prev[j];

        a += connection_a * prev_value;
        b += connection_b * prev_value;
        prev_deltas[j] += grad_a * connection_a + grad_b * connection_b;

        gradient_sum_a += grad_a * prev_value;
        gradient_sum_b += grad_b * prev_value;
      }

      if (learning_rate != 0.0f) {
        // Project the gradient onto the sum-to-one constraint by removing its
        // mean, then keep each pin non-negative and normalized.
        const float mean_gradient_a = gradient_sum_a / (float)prev_len;
        const float mean_gradient_b = gradient_sum_b / (float)prev_len;
        for (int j = 0; j < prev_len; ++j) {
          const float prev_value = prev[j];
          wire_a[j] -= learning_rate * (grad_a * prev_value - mean_gradient_a);
          wire_b[j] -= learning_rate * (grad_b * prev_value - mean_gradient_b);
        }
        normalize_probabilities(wire_a, prev_len);
        normalize_probabilities(wire_b, prev_len);
      }

      if (!is_output_layer && learning_rate != 0.0f) {
        Cell *cell = &circuit->cells[hidden_cell_offset + gate];
        float *gate_probabilities = cell->gate_probabilities.data;
        float gate_gradients[16];
        float gradient_sum = 0.0f;
        for (int k = 0; k < 16; ++k) {
          float value;
          float unused_da;
          float unused_db;
          soft_gate_forward_and_derivative(k, a, b, &value, &unused_da,
                                           &unused_db);
          // dLoss / d(gate probability k) = dLoss/dOut * gate_k(a, b).
          gate_gradients[k] = grad_out * value;
          gradient_sum += gate_gradients[k];
        }
        const float mean_gradient = gradient_sum / 16.0f;
        for (int k = 0; k < 16; ++k) {
          gate_probabilities[k] -=
              learning_rate * (gate_gradients[k] - mean_gradient);
        }
        normalize_probabilities(gate_probabilities, 16);
      }
    }

    curr_offset = prev_offset;
  }
}

EMSCRIPTEN_KEEPALIVE
Trainer *create_trainer(Circuit *circuit, const bool inputs[],
                        const float outputs[], int data_count,
                        float learning_rate) {
  clear_error();
  if (circuit == NULL || inputs == NULL || outputs == NULL || data_count <= 0 ||
      circuit->inputs_len <= 0 || circuit->outputs_len <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return NULL;
  }
  Trainer *result = malloc(sizeof(*result));
  if (result == NULL) {
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  result->error = malloc(circuit->outputs_len * sizeof(*result->error));
  if (result->error == NULL) {
    free(result);
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  result->circuit = circuit;
  result->inputs = inputs;
  result->outputs = outputs;
  result->data_count = data_count;
  result->index = 0;
  result->learning_rate = learning_rate;
  return result;
}

EMSCRIPTEN_KEEPALIVE
void destroy_trainer(Trainer *trainer) {
  if (trainer == NULL) {
    return;
  }
  free(trainer->error);
  free(trainer);
}

EMSCRIPTEN_KEEPALIVE
void trainer_reset(Trainer *trainer) {
  if (trainer == NULL) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();
  trainer->index = 0;
}

EMSCRIPTEN_KEEPALIVE
void trainer_set_learning_rate(Trainer *trainer, float learning_rate) {
  if (trainer == NULL) {
    set_error(ERR_INVALID_PARAMS);
    return;
  }
  clear_error();
  trainer->learning_rate = learning_rate;
}

/**
 * Runs one sample through the circuit, applies one back-propagation step and
 * returns the sample loss:
 *
 *   0.5 * sum((output - target)^2)
 *
 * The trainer advances to the next sample, wrapping back to the first one after
 * the last sample. No memory is allocated during a step.
 */
EMSCRIPTEN_KEEPALIVE
float trainer_step(Trainer *trainer) {
  if (trainer == NULL || trainer->circuit == NULL || trainer->inputs == NULL ||
      trainer->outputs == NULL || trainer->error == NULL ||
      trainer->data_count <= 0) {
    set_error(ERR_INVALID_PARAMS);
    return 0.0f;
  }
  clear_error();

  Circuit *circuit = trainer->circuit;
  const int input_len = circuit->inputs_len;
  const int output_len = circuit->outputs_len;
  const int input_offset = trainer->index * input_len;
  const int output_offset = trainer->index * output_len;

  for (int i = 0; i < input_len; ++i) {
    circuit->inputs[i] = trainer->inputs[input_offset + i];
  }

  circuit_feed_forward(circuit);
  if (get_error() != ERR_NONE) {
    return 0.0f;
  }

  float loss = 0.0f;
  for (int i = 0; i < output_len; ++i) {
    const float difference =
        circuit->outputs[i] - trainer->outputs[output_offset + i];
    trainer->error[i] = difference;
    loss += 0.5f * difference * difference;
  }

  circuit_back_propergate(circuit, trainer->error, trainer->learning_rate);

  ++trainer->index;
  if (trainer->index >= trainer->data_count) {
    trainer->index = 0;
  }
  return loss;
}

EMSCRIPTEN_KEEPALIVE
void func() { printf("Hello from C\n"); }
