#ifndef TESTS_EMSCRIPTEN_H
#define TESTS_EMSCRIPTEN_H

// Native tests include src/main.c directly, so they only need the export marker
// to be a no-op.
#define EMSCRIPTEN_KEEPALIVE

#endif
