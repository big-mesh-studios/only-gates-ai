#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <stdio.h>

typedef struct GateWeights {
} GateWeights;

EMSCRIPTEN_KEEPALIVE
void func() { printf("Hello from C\n"); }
