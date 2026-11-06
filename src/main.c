#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <stdio.h>
#include <stdlib.h>

// Utility for preventing clang-format from mis-parsing a member varriable as a
// keyword.
#define ID(x) x

typedef struct {
  union {
    float data[16];
    struct {
      float zero;
      float nor;
      float lnt;
      float not_a;
      float rnt;
      float not_b;
      float ID(xor); // <-- ID() prevents clang-format from getting confused.
      float nand;
      float and;
      float xnor;
      float proj_b;
      float imp;
      float proj_a;
      float cimp;
      float ID(or); // <-- ID() prevents clang-format from getting confused.
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

inline void clear_error() { _last_error = ERR_NONE; }

inline void set_error(int error_code) { _last_error = error_code; }

EMSCRIPTEN_KEEPALIVE
int get_error() { return _last_error; }

EMSCRIPTEN_KEEPALIVE
Circuit *create_circuit(int shape[], int shape_size) {
  clear_error();
  if (shape_size <= 0) {
    set_error(ERR_INVALID_SHAPE);
    return NULL;
  }
  for (int i = 0; i < shape_size; ++i) {
    if (shape[i] <= 0) {
      set_error(ERR_INVALID_SHAPE);
      return NULL;
    }
  }
  Circuit *result = malloc(sizeof(Circuit));
  if (result == NULL) {
    set_error(ERR_OUT_OF_MEMORY);
    return NULL;
  }
  return result;
}

EMSCRIPTEN_KEEPALIVE
void destroy_circuit(Circuit *circuit) { free(circuit); }

EMSCRIPTEN_KEEPALIVE
void func() { printf("Hello from C\n"); }
