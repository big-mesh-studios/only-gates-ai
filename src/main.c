#include <emscripten.h> // Required for EMSCRIPTEN_KEEPALIVE
#include <stdio.h>

EMSCRIPTEN_KEEPALIVE
void func() {
  printf("Hello from C\n");
}

