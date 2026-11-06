#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <stdio.h>
#include <stdlib.h>

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
  float *connection_probabilities;
  int connection_probabilities_len;
} Circuit;

const int ERR_NONE = 0;
const int ERR_OUT_OF_MEMORY = 1;
const int ERR_INVALID_SHAPE = 2;
const int ERR_INVALID_PARAMS = 3;

static int _last_error = ERR_NONE;

static inline void clear_error(void) { _last_error = ERR_NONE; }

static inline void set_error(int error_code) { _last_error = error_code; }

EMSCRIPTEN_KEEPALIVE
int get_error() { return _last_error; }

float soft_gate_zero(float _a, float _b) { return 0.0f; }

float soft_gate_nor(float a, float b) { return 1.0f - (a + b - a * b); }

float soft_gate_lnt(float a, float b) { return a - a * b; }

float soft_gate_not_a(float a, float b) { return 1.0f - a; }

float soft_gate_rnt(float a, float b) { return b - a * b; }

float soft_gate_not_b(float a, float b) { return 1.0f - b; }

float soft_gate_xor(float a, float b) { return a + b - 2.0f * a * b; }

float soft_gate_nand(float a, float b) { return 1 - a * b; }

float soft_gate_and(float a, float b) { return 1.0f - a * b; }

float soft_gate_xnor(float a, float b) { return 1.0f - (a + b - 2.0f * a * b); }

float soft_gate_proj_b(float _a, float b) { return b; }

float soft_gate_imp(float a, float b) { return 1.0 - a + a * b; }

float soft_gate_proj_a(float a, float _b) { return a; }

float soft_gate_cimp(float a, float b) { return 1.0f - b + a * b; }

float soft_gate_or(float a, float b) { return a + b - a * b; }

float soft_gate_one(float _a, float _b) { return 1.0f; }

typedef float SoftGateFunc(float, float);

const SoftGateFunc *soft_gates[16] = {
    soft_gate_zero,   soft_gate_nor,   soft_gate_lnt,    soft_gate_not_a,
    soft_gate_rnt,    soft_gate_not_b, soft_gate_xor,    soft_gate_nand,
    soft_gate_and,    soft_gate_xnor,  soft_gate_proj_b, soft_gate_imp,
    soft_gate_proj_a, soft_gate_cimp,  soft_gate_or,     soft_gate_one,
};

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
  int cell_len = 0;
  for (int i = 1; i < shape_len - 1; ++i) {
    cell_len += shape[i];
  }
  result->cells = malloc(cell_len * sizeof(Cell));
  if (result->cells == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  cells_allocated = true;
  result->cell_len = cell_len;
  int connection_len = 0;
  for (int i = 1; i < shape_len; ++i) {
    int num_pre_gates = shape[i - 1];
    int num_gates = shape[i];
    connection_len += (num_pre_gates * num_gates) << 1;
  }
  result->connection_probabilities =
      malloc(connection_len * sizeof(*result->connection_probabilities));
  if (result->connection_probabilities == NULL) {
    memory_cleanup();
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  connection_probabilities_allocated = true;
  result->connection_probabilities_len = connection_len;
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
  free(circuit);
}

EMSCRIPTEN_KEEPALIVE
void circuit_set_input(Circuit *circuit, bool input[]) {}

EMSCRIPTEN_KEEPALIVE
void circuit_read_output(Circuit *circuit, float output[]) {}

EMSCRIPTEN_KEEPALIVE
void circuit_feed_forward(Circuit *circuit) {}

EMSCRIPTEN_KEEPALIVE
void circuit_back_propergate(Circuit *circuit, float error[], float learning_rate) {}

EMSCRIPTEN_KEEPALIVE
void func() { printf("Hello from C\n"); }
