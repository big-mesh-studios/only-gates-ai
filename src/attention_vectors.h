#ifndef ONLY_GATES_AI_ATTENTION_VECTORS_H
#define ONLY_GATES_AI_ATTENTION_VECTORS_H

#include <stdint.h>

/*
 * Small loaders for public binary word-vector formats.
 *
 * The experiment in src/attention.c only needs bit vectors; this file keeps
 * the file parsing out of the attention math so a real vocabulary can be
 * swapped in without touching training or inference.
 */

typedef struct {
  int count;
  int bits;
  char **words;
  uint8_t *vectors; /* count x bits, row-major, values 0/1 */
} BinaryVectors;

/*
 * Tissier et al., "Near-Lossless Binarization of Word Embeddings".
 *
 * Format:
 *   <count> <bits>
 *   WORD <uint64> <uint64> ...
 *
 * Each integer contributes 64 bits, least-significant bit first.
 * This is the output of github.com/tca19/near-lossless-binarization.
 */
BinaryVectors *binary_vectors_load_tissier(const char *path, int max_count);

/*
 * GloVe / fastText / Word2Bits text format:
 *   <count> <dim>
 *   WORD <float> <float> ...
 *
 * Each value is thresholded at zero, so 1-bit Word2Bits files work directly
 * and ordinary GloVe vectors are binarized by sign. `max_bits` truncates the
 * loaded vector; pass 0 to keep the full dimension.
 */
BinaryVectors *binary_vectors_load_text(const char *path, int max_bits,
                                        int max_count);

void binary_vectors_destroy(BinaryVectors *vectors);

#endif
