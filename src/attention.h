#ifndef ONLY_GATES_AI_ATTENTION_H
#define ONLY_GATES_AI_ATTENTION_H

#include <stdint.h>

/*
 * Experimental integer-only binary attention.
 *
 * This module is deliberately independent of the differentiable gate circuit in
 * src/main.c. It is a small test bed for one question:
 *
 *   can attention be trained with floats and then executed with integer /
 *   popcount operations only?
 *
 * The data path is:
 *
 *   query bits -> ternary projection -> query bits
 *   key bits   -> ternary projection -> key bits
 *   score      = popcount(xnor(query, key))   (integer)
 *   weights    = fixed-point softmax(score)   (integer)
 *
 * Training uses real-valued shadow weights with a straight-through estimator
 * (STE) through the ternary quantizer, so gradients are ordinary floats. The
 * inference path uses only int8/uint8/uint64 arithmetic, popcount and a
 * fixed-point exp table. No float, sqrt or exp is used while predicting.
 *
 * The model is intentionally tiny: a single attention head over a fixed set of
 * key vectors. It is a primitive, not a transformer. The intended next step is
 * to feed it pretrained binary word vectors instead of the synthetic random
 * keys generated here.
 */

typedef struct {
  int dim;       /* feature dimension */
  int words;     /* number of 64-bit words per vector: ceil(dim / 64) */
  int key_count; /* number of keys / classes */

  /* Real-valued shadow weights, row-major dim x dim. */
  float *wq;
  float *wk;

  /* Raw key vectors, key_count x dim, values 0/1. */
  uint8_t *keys;

  uint64_t *word_masks; /* words, masks off unused high bits */
  int inv_sqrt_q8;      /* round(256 / sqrt(dim)), used by the int softmax */

  /* Quantized inference weights, filled by attention_quantize(). */
  int8_t *wq_q;
  int8_t *wk_q;
  uint8_t *key_bits;   /* projected keys, key_count x dim */
  uint64_t *key_words; /* projected keys, key_count x words */

  /* Row masks for the ternary query projection, dim x words. */
  uint64_t *wq_pos;
  uint64_t *wq_neg;
  int *wq_delta; /* popcount(pos) - popcount(neg) per row */

  /* Scratch used by the integer forward pass. */
  uint8_t *query_bits;
  uint64_t *query_words;
  int32_t *logits;
  int32_t *weights;

  /* Adam state for training. */
  float *mq;
  float *vq;
  float *mk;
  float *vk;
} AttentionModel;

typedef struct {
  int count;
  int dim;
  uint8_t *queries; /* count x dim, 0/1 */
  int *targets;     /* count */
} AttentionSamples;

/* Creates a model with random keys and identity ternary projections. */
AttentionModel *attention_model_create(int dim, int key_count, uint32_t seed);
void attention_model_destroy(AttentionModel *model);

/* Replaces one key vector. Call attention_quantize() again before int
 * inference. */
void attention_model_set_key(AttentionModel *model, int index,
                             const uint8_t *bits);

/*
 * Builds a synthetic associative-recall task:
 *
 *   - every sample picks a target key;
 *   - the query is the target key passed through a fixed random
 *     permutation/sign transform (the "hidden relation" the model must learn);
 *   - `noise_percent` of the query bits are then flipped.
 *
 * `transform_seed` defines the hidden relation, so train and test sets that
 * should share a relation must use the same transform_seed.
 */
AttentionSamples *attention_samples_create(const AttentionModel *model,
                                           int count, int noise_percent,
                                           uint32_t transform_seed,
                                           uint32_t sample_seed);
void attention_samples_destroy(AttentionSamples *samples);

/* Trains with float STE + Adam. Returns the mean cross-entropy of the last
 * epoch. */
float attention_train(AttentionModel *model, const AttentionSamples *samples,
                      int epochs, float learning_rate, uint32_t shuffle_seed);

/* Converts the shadow weights to ternary and precomputes projected keys. */
void attention_quantize(AttentionModel *model);

/* Reference float path over the same ternary weights (for comparisons). */
int attention_predict_float(const AttentionModel *model, const uint8_t *query);

/*
 * Integer-only path. Returns the argmax key and, if `alpha_q15` is not NULL,
 * writes key_count fixed-point attention weights (Q15, sum ~= 32768).
 * attention_quantize() must have been called first.
 */
int attention_forward_int(AttentionModel *model, const uint8_t *query,
                          int32_t *alpha_q15);
int attention_predict_int(AttentionModel *model, const uint8_t *query);

float attention_accuracy_float(const AttentionModel *model,
                               const AttentionSamples *samples);
float attention_accuracy_int(AttentionModel *model,
                             const AttentionSamples *samples);

#endif
