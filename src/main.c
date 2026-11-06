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
void func() { printf("Hello from C\n"); }
