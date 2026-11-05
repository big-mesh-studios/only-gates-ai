#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <stdio.h>

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
      float xor;
      float nand;
      float and;
      float xnor;
      float proj_b;
      float imp;
      float proj_a;
      float cimp;
      float or;
      float one;
    };
  };
} GateProbabilities;

EMSCRIPTEN_KEEPALIVE
void func() { printf("Hello from C\n"); }
