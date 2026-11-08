#ifndef ONLY_GATES_AI_TINY_GPT_QUANTIZE_H
#define ONLY_GATES_AI_TINY_GPT_QUANTIZE_H

#include "tiny_gpt.h"

/*
 * Converts a packed float32 checkpoint into the integer model. This is the
 * only tiny-GPT translation unit that touches floating point; inference is in
 * tiny_gpt_infer.c and uses integers exclusively.
 *
 * Packed weight order:
 *   tokenEmb (vocab * dim)
 *   posEmb   (context * dim)
 *   gFinal   (dim)
 *   per layer: g1 (dim), wqkv (dim * 3 * dim), wo (dim * dim),
 *              g2 (dim), w1 (dim * mlp), w2 (mlp * dim)
 */
TinyGpt *tiny_gpt_create(int vocab_size, int context, int dim, int layers,
                         int heads, int mlp, const float *packed_weights);

#endif
