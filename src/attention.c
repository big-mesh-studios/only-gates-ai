#include "attention.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

/*
 * Fixed-point helpers.
 *
 * The inference path must not use floating point. The only floating point in
 * this file is in training (and in attention_quantize(), which converts the
 * trained shadow weights once).
 */

/* round(32768 * 2^(-k/32)), k = 0..32. */
static const int32_t TWO_POW_NEG_Q15[33] = {
    32768, 32066, 31379, 30706, 30048, 29405, 28774, 28158, 27554, 26964, 26386,
    25821, 25268, 24726, 24196, 23678, 23170, 22674, 22188, 21713, 21247, 20792,
    20347, 19911, 19484, 19066, 18658, 18258, 17867, 17484, 17109, 16743, 16384,
};

/*
 * exp(-x) in Q15 for x >= 0 in Q8. Uses 2^(-x*log2(e)) with an interpolated
 * 33-entry table for the fractional power. Integer-only.
 */
static int32_t exp_neg_q15(int32_t x_q8) {
  if (x_q8 <= 0) {
    return 32768;
  }
  if (x_q8 > 2048) {
    return 0;
  }
  const int32_t y_q8 = (x_q8 * 369) >> 8; /* 369/256 ~= log2(e) */
  const int32_t integer_part = y_q8 >> 8;
  const int32_t fraction = y_q8 & 255;
  const int32_t index = fraction >> 3;
  const int32_t remainder = fraction & 7;
  const int32_t a = TWO_POW_NEG_Q15[index];
  const int32_t b = TWO_POW_NEG_Q15[index + 1];
  /* The table is decreasing, so interpolate downward to keep the shift on a
   * non-negative value. */
  const int32_t value = a - (((a - b) * remainder) >> 3);
  return value >> integer_part;
}

static uint32_t next_u32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static uint32_t seed_or_default(uint32_t seed) {
  return seed == 0 ? 0x9e3779b9u : seed;
}

static int ternary_from_float(float value) {
  if (value > 0.5f) {
    return 1;
  }
  if (value < -0.5f) {
    return -1;
  }
  return 0;
}

static void quantize_weights(const float *weights, int8_t *out, int length) {
  for (int i = 0; i < length; ++i) {
    out[i] = (int8_t)ternary_from_float(weights[i]);
  }
}

/*
 * Projects `rows` output bits from the ternary matrix `weights` and the binary
 * input `x`. Integer-only; used by training for the forward pass and by
 * attention_quantize().
 */
static void project_ternary_bits(const int8_t *weights, const uint8_t *x,
                                 int rows, int cols, uint8_t *out) {
  for (int i = 0; i < rows; ++i) {
    int dot = 0;
    const int8_t *row = weights + i * cols;
    for (int j = 0; j < cols; ++j) {
      dot += row[j] * (x[j] ? 1 : -1);
    }
    out[i] = dot >= 0 ? 1 : 0;
  }
}

static void build_row_masks(const int8_t *weights, int rows, int cols,
                            int words, uint64_t *pos, uint64_t *neg,
                            int *delta) {
  for (int i = 0; i < rows; ++i) {
    int total_pos = 0;
    int total_neg = 0;
    const int8_t *row = weights + i * cols;
    for (int word = 0; word < words; ++word) {
      uint64_t p = 0;
      uint64_t n = 0;
      const int start = word * 64;
      int end = start + 64;
      if (end > cols) {
        end = cols;
      }
      for (int j = start; j < end; ++j) {
        const int bit = j - start;
        if (row[j] > 0) {
          p |= (uint64_t)1 << bit;
        } else if (row[j] < 0) {
          n |= (uint64_t)1 << bit;
        }
      }
      pos[(size_t)i * words + word] = p;
      neg[(size_t)i * words + word] = n;
      total_pos += __builtin_popcountll(p);
      total_neg += __builtin_popcountll(n);
    }
    delta[i] = total_pos - total_neg;
  }
}

static void bits_to_words(const uint8_t *bits, int length, int words,
                          uint64_t *out) {
  memset(out, 0, (size_t)words * sizeof(*out));
  for (int i = 0; i < length; ++i) {
    if (bits[i]) {
      out[i >> 6] |= (uint64_t)1 << (i & 63);
    }
  }
}

static uint64_t word_mask(int length, int word) {
  const int remaining = length - word * 64;
  if (remaining >= 64) {
    return UINT64_MAX;
  }
  return ((uint64_t)1 << remaining) - 1;
}

AttentionModel *attention_model_create(int dim, int key_count, uint32_t seed) {
  if (dim <= 0 || dim > 1024 || key_count <= 0) {
    return NULL;
  }

  AttentionModel *model = calloc(1, sizeof(*model));
  if (model == NULL) {
    return NULL;
  }
  const int words = (dim + 63) / 64;
  model->dim = dim;
  model->words = words;
  model->key_count = key_count;
  model->inv_sqrt_q8 = (int)(256.0f / sqrtf((float)dim) + 0.5f);

  model->wq = calloc((size_t)dim * dim, sizeof(*model->wq));
  model->wk = calloc((size_t)dim * dim, sizeof(*model->wk));
  model->keys = calloc((size_t)key_count * dim, sizeof(*model->keys));
  model->wq_q = calloc((size_t)dim * dim, sizeof(*model->wq_q));
  model->wk_q = calloc((size_t)dim * dim, sizeof(*model->wk_q));
  model->key_bits = calloc((size_t)key_count * dim, sizeof(*model->key_bits));
  model->key_words =
      calloc((size_t)key_count * words, sizeof(*model->key_words));
  model->word_masks = calloc((size_t)words, sizeof(*model->word_masks));
  model->wq_pos = calloc((size_t)dim * words, sizeof(*model->wq_pos));
  model->wq_neg = calloc((size_t)dim * words, sizeof(*model->wq_neg));
  model->wq_delta = calloc((size_t)dim, sizeof(*model->wq_delta));
  model->query_bits = calloc((size_t)dim, sizeof(*model->query_bits));
  model->query_words = calloc((size_t)words, sizeof(*model->query_words));
  model->logits = calloc((size_t)key_count, sizeof(*model->logits));
  model->weights = calloc((size_t)key_count, sizeof(*model->weights));
  model->mq = calloc((size_t)dim * dim, sizeof(*model->mq));
  model->vq = calloc((size_t)dim * dim, sizeof(*model->vq));
  model->mk = calloc((size_t)dim * dim, sizeof(*model->mk));
  model->vk = calloc((size_t)dim * dim, sizeof(*model->vk));

  if (model->wq == NULL || model->wk == NULL || model->keys == NULL ||
      model->wq_q == NULL || model->wk_q == NULL || model->key_bits == NULL ||
      model->key_words == NULL || model->word_masks == NULL ||
      model->wq_pos == NULL || model->wq_neg == NULL ||
      model->wq_delta == NULL || model->query_bits == NULL ||
      model->query_words == NULL || model->logits == NULL ||
      model->weights == NULL || model->mq == NULL || model->vq == NULL ||
      model->mk == NULL || model->vk == NULL) {
    attention_model_destroy(model);
    return NULL;
  }

  for (int word = 0; word < words; ++word) {
    model->word_masks[word] = word_mask(dim, word);
  }

  /* Identity ternary projections; the synthetic task has to learn its hidden
   * permutation from this starting point. */
  for (int i = 0; i < dim; ++i) {
    model->wq[i * dim + i] = 1.0f;
    model->wk[i * dim + i] = 1.0f;
  }

  uint32_t state = seed_or_default(seed);
  for (int i = 0; i < key_count * dim; ++i) {
    model->keys[i] = (uint8_t)(next_u32(&state) & 1u);
  }

  return model;
}

void attention_model_destroy(AttentionModel *model) {
  if (model == NULL) {
    return;
  }
  free(model->wq);
  free(model->wk);
  free(model->keys);
  free(model->wq_q);
  free(model->wk_q);
  free(model->key_bits);
  free(model->key_words);
  free(model->word_masks);
  free(model->wq_pos);
  free(model->wq_neg);
  free(model->wq_delta);
  free(model->query_bits);
  free(model->query_words);
  free(model->logits);
  free(model->weights);
  free(model->mq);
  free(model->vq);
  free(model->mk);
  free(model->vk);
  free(model);
}

void attention_model_set_key(AttentionModel *model, int index,
                             const uint8_t *bits) {
  if (model == NULL || bits == NULL || index < 0 || index >= model->key_count) {
    return;
  }
  memcpy(model->keys + (size_t)index * model->dim, bits,
         (size_t)model->dim * sizeof(*bits));
}

AttentionSamples *attention_samples_create(const AttentionModel *model,
                                           int count, int noise_percent,
                                           uint32_t transform_seed,
                                           uint32_t sample_seed) {
  if (model == NULL || count <= 0) {
    return NULL;
  }
  if (noise_percent < 0) {
    noise_percent = 0;
  }
  if (noise_percent > 100) {
    noise_percent = 100;
  }

  AttentionSamples *samples = calloc(1, sizeof(*samples));
  if (samples == NULL) {
    return NULL;
  }
  samples->count = count;
  samples->dim = model->dim;
  samples->queries =
      calloc((size_t)count * model->dim, sizeof(*samples->queries));
  samples->targets = calloc((size_t)count, sizeof(*samples->targets));
  if (samples->queries == NULL || samples->targets == NULL) {
    attention_samples_destroy(samples);
    return NULL;
  }

  int *permutation = malloc((size_t)model->dim * sizeof(*permutation));
  uint8_t *flip = malloc((size_t)model->dim * sizeof(*flip));
  if (permutation == NULL || flip == NULL) {
    free(permutation);
    free(flip);
    attention_samples_destroy(samples);
    return NULL;
  }

  uint32_t transform_state = seed_or_default(transform_seed);
  for (int i = 0; i < model->dim; ++i) {
    permutation[i] = i;
    flip[i] = (uint8_t)(next_u32(&transform_state) & 1u);
  }
  for (int i = model->dim - 1; i > 0; --i) {
    const int j = (int)(next_u32(&transform_state) % (uint32_t)(i + 1));
    const int tmp = permutation[i];
    permutation[i] = permutation[j];
    permutation[j] = tmp;
  }

  uint32_t sample_state = seed_or_default(sample_seed);
  for (int sample = 0; sample < count; ++sample) {
    const int target =
        (int)(next_u32(&sample_state) % (uint32_t)model->key_count);
    samples->targets[sample] = target;
    uint8_t *query = samples->queries + (size_t)sample * model->dim;
    for (int i = 0; i < model->dim; ++i) {
      uint8_t bit =
          model->keys[(size_t)target * model->dim + permutation[i]] ^ flip[i];
      if ((int)(next_u32(&sample_state) % 100u) < noise_percent) {
        bit ^= 1u;
      }
      query[i] = bit;
    }
  }

  free(permutation);
  free(flip);
  return samples;
}

void attention_samples_destroy(AttentionSamples *samples) {
  if (samples == NULL) {
    return;
  }
  free(samples->queries);
  free(samples->targets);
  free(samples);
}

void attention_quantize(AttentionModel *model) {
  if (model == NULL) {
    return;
  }
  quantize_weights(model->wq, model->wq_q, model->dim * model->dim);
  quantize_weights(model->wk, model->wk_q, model->dim * model->dim);
  build_row_masks(model->wq_q, model->dim, model->dim, model->words,
                  model->wq_pos, model->wq_neg, model->wq_delta);
  for (int key = 0; key < model->key_count; ++key) {
    project_ternary_bits(model->wk_q, model->keys + (size_t)key * model->dim,
                         model->dim, model->dim,
                         model->key_bits + (size_t)key * model->dim);
    bits_to_words(model->key_bits + (size_t)key * model->dim, model->dim,
                  model->words, model->key_words + (size_t)key * model->words);
  }
}

int attention_predict_float(const AttentionModel *model, const uint8_t *query) {
  if (model == NULL || query == NULL) {
    return -1;
  }

  const int dim = model->dim;
  const int key_count = model->key_count;

  int8_t *wq = malloc((size_t)dim * dim * sizeof(*wq));
  int8_t *wk = malloc((size_t)dim * dim * sizeof(*wk));
  int8_t *qsign = malloc((size_t)dim * sizeof(*qsign));
  if (wq == NULL || wk == NULL || qsign == NULL) {
    free(wq);
    free(wk);
    free(qsign);
    return -1;
  }

  quantize_weights(model->wq, wq, dim * dim);
  quantize_weights(model->wk, wk, dim * dim);

  for (int i = 0; i < dim; ++i) {
    int dot = 0;
    for (int j = 0; j < dim; ++j) {
      dot += wq[i * dim + j] * (query[j] ? 1 : -1);
    }
    qsign[i] = dot >= 0 ? 1 : -1;
  }

  const float inv_sqrt = 1.0f / sqrtf((float)dim);
  int best = 0;
  float best_score = -1e30f;
  for (int key = 0; key < key_count; ++key) {
    int dot = 0;
    for (int i = 0; i < dim; ++i) {
      int value = 0;
      for (int j = 0; j < dim; ++j) {
        value +=
            wk[i * dim + j] * (model->keys[(size_t)key * dim + j] ? 1 : -1);
      }
      const int8_t k = value >= 0 ? 1 : -1;
      dot += qsign[i] * k;
    }
    const float score = (float)dot * inv_sqrt;
    if (score > best_score) {
      best_score = score;
      best = key;
    }
  }

  free(wq);
  free(wk);
  free(qsign);
  return best;
}

int attention_forward_int(AttentionModel *model, const uint8_t *query,
                          int32_t *alpha_q15) {
  if (model == NULL || query == NULL || model->wq_pos == NULL) {
    return -1;
  }

  const int dim = model->dim;
  const int key_count = model->key_count;
  const int words = model->words;

  /* Integer query projection: dot = 2*(popcount(pos&q) - popcount(neg&q))
   * - (popcount(pos) - popcount(neg)). */
  bits_to_words(query, dim, words, model->query_words);
  for (int i = 0; i < dim; ++i) {
    int p = 0;
    int n = 0;
    for (int word = 0; word < words; ++word) {
      p += __builtin_popcountll(model->wq_pos[(size_t)i * words + word] &
                                model->query_words[word]);
      n += __builtin_popcountll(model->wq_neg[(size_t)i * words + word] &
                                model->query_words[word]);
    }
    const int dot = 2 * (p - n) - model->wq_delta[i];
    model->query_bits[i] = dot >= 0 ? 1 : 0;
  }
  bits_to_words(model->query_bits, dim, words, model->query_words);

  int best = 0;
  int32_t best_logit = INT32_MIN;
  for (int key = 0; key < key_count; ++key) {
    int matches = 0;
    for (int word = 0; word < words; ++word) {
      const uint64_t difference = model->query_words[word] ^
                                  model->key_words[(size_t)key * words + word];
      matches += __builtin_popcountll((~difference) & model->word_masks[word]);
    }
    const int dot = 2 * matches - dim;
    const int32_t logit = dot * model->inv_sqrt_q8;
    model->logits[key] = logit;
    if (logit > best_logit) {
      best_logit = logit;
      best = key;
    }
  }

  int64_t sum = 0;
  for (int key = 0; key < key_count; ++key) {
    const int32_t value = exp_neg_q15(best_logit - model->logits[key]);
    model->weights[key] = value;
    sum += value;
  }
  if (sum <= 0) {
    sum = 1;
  }
  for (int key = 0; key < key_count; ++key) {
    const int32_t alpha =
        (int32_t)(((int64_t)model->weights[key] * 32768 + sum / 2) / sum);
    model->weights[key] = alpha;
    if (alpha_q15 != NULL) {
      alpha_q15[key] = alpha;
    }
  }

  /* Argmax of the fixed-point weights is the argmax of the logits. */
  return best;
}

int attention_predict_int(AttentionModel *model, const uint8_t *query) {
  return attention_forward_int(model, query, NULL);
}

float attention_train(AttentionModel *model, const AttentionSamples *samples,
                      int epochs, float learning_rate, uint32_t shuffle_seed) {
  if (model == NULL || samples == NULL || epochs <= 0 || samples->count <= 0 ||
      samples->dim != model->dim) {
    return INFINITY;
  }

  const int dim = model->dim;
  const int key_count = model->key_count;
  const int count = samples->count;
  const float inv_sqrt = 1.0f / sqrtf((float)dim);

  int8_t *wq_b = malloc((size_t)dim * dim * sizeof(*wq_b));
  int8_t *wk_b = malloc((size_t)dim * dim * sizeof(*wk_b));
  uint8_t *qbits = malloc((size_t)dim * sizeof(*qbits));
  uint8_t *kbits = malloc((size_t)key_count * dim * sizeof(*kbits));
  int8_t *qsign = malloc((size_t)dim * sizeof(*qsign));
  int8_t *ksign = malloc((size_t)key_count * dim * sizeof(*ksign));
  float *scores = malloc((size_t)key_count * sizeof(*scores));
  float *alpha = malloc((size_t)key_count * sizeof(*alpha));
  float *dscore = malloc((size_t)key_count * sizeof(*dscore));
  float *dq = malloc((size_t)dim * sizeof(*dq));
  float *dk = malloc((size_t)key_count * dim * sizeof(*dk));
  float *dWq = malloc((size_t)dim * dim * sizeof(*dWq));
  float *dWk = malloc((size_t)dim * dim * sizeof(*dWk));
  int *order = malloc((size_t)count * sizeof(*order));
  if (wq_b == NULL || wk_b == NULL || qbits == NULL || kbits == NULL ||
      qsign == NULL || ksign == NULL || scores == NULL || alpha == NULL ||
      dscore == NULL || dq == NULL || dk == NULL || dWq == NULL ||
      dWk == NULL || order == NULL) {
    free(wq_b);
    free(wk_b);
    free(qbits);
    free(kbits);
    free(qsign);
    free(ksign);
    free(scores);
    free(alpha);
    free(dscore);
    free(dq);
    free(dk);
    free(dWq);
    free(dWk);
    free(order);
    return INFINITY;
  }

  memset(model->mq, 0, (size_t)dim * dim * sizeof(*model->mq));
  memset(model->vq, 0, (size_t)dim * dim * sizeof(*model->vq));
  memset(model->mk, 0, (size_t)dim * dim * sizeof(*model->mk));
  memset(model->vk, 0, (size_t)dim * dim * sizeof(*model->vk));

  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float epsilon = 1e-8f;
  float beta1_pow = 1.0f;
  float beta2_pow = 1.0f;

  uint32_t shuffle_state = seed_or_default(shuffle_seed);
  for (int i = 0; i < count; ++i) {
    order[i] = i;
  }

  float epoch_loss = 0.0f;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    for (int i = count - 1; i > 0; --i) {
      const int j = (int)(next_u32(&shuffle_state) % (uint32_t)(i + 1));
      const int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    epoch_loss = 0.0f;
    for (int sample_index = 0; sample_index < count; ++sample_index) {
      const int sample = order[sample_index];
      const uint8_t *query = samples->queries + (size_t)sample * dim;
      const int target = samples->targets[sample];

      quantize_weights(model->wq, wq_b, dim * dim);
      quantize_weights(model->wk, wk_b, dim * dim);
      project_ternary_bits(wq_b, query, dim, dim, qbits);
      for (int key = 0; key < key_count; ++key) {
        project_ternary_bits(wk_b, model->keys + (size_t)key * dim, dim, dim,
                             kbits + (size_t)key * dim);
      }

      for (int i = 0; i < dim; ++i) {
        qsign[i] = qbits[i] ? 1 : -1;
      }
      for (int key = 0; key < key_count; ++key) {
        for (int i = 0; i < dim; ++i) {
          ksign[key * dim + i] = kbits[key * dim + i] ? 1 : -1;
        }
      }

      float max_score = -1e30f;
      for (int key = 0; key < key_count; ++key) {
        int dot = 0;
        for (int i = 0; i < dim; ++i) {
          dot += qsign[i] * ksign[key * dim + i];
        }
        scores[key] = (float)dot * inv_sqrt;
        if (scores[key] > max_score) {
          max_score = scores[key];
        }
      }
      float sum_exp = 0.0f;
      for (int key = 0; key < key_count; ++key) {
        alpha[key] = expf(scores[key] - max_score);
        sum_exp += alpha[key];
      }
      for (int key = 0; key < key_count; ++key) {
        alpha[key] /= sum_exp;
      }
      epoch_loss += -logf(alpha[target] + 1e-9f);

      for (int key = 0; key < key_count; ++key) {
        dscore[key] = alpha[key] - (key == target ? 1.0f : 0.0f);
      }

      for (int i = 0; i < dim; ++i) {
        float value = 0.0f;
        for (int key = 0; key < key_count; ++key) {
          value += dscore[key] * (float)ksign[key * dim + i];
        }
        dq[i] = value * inv_sqrt;
      }

      for (int key = 0; key < key_count; ++key) {
        for (int i = 0; i < dim; ++i) {
          dk[key * dim + i] = dscore[key] * (float)qsign[i] * inv_sqrt;
        }
      }

      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          dWq[i * dim + j] = dq[i] * (float)(query[j] ? 1 : -1) * inv_sqrt;
        }
      }

      memset(dWk, 0, (size_t)dim * dim * sizeof(*dWk));
      for (int key = 0; key < key_count; ++key) {
        for (int i = 0; i < dim; ++i) {
          const float value = dk[key * dim + i] * inv_sqrt;
          for (int j = 0; j < dim; ++j) {
            const float key_sign =
                model->keys[(size_t)key * dim + j] ? 1.0f : -1.0f;
            dWk[i * dim + j] += value * key_sign;
          }
        }
      }

      beta1_pow *= beta1;
      beta2_pow *= beta2;
      const float bias1 = 1.0f - beta1_pow;
      const float bias2 = 1.0f - beta2_pow;

      for (int index = 0; index < dim * dim; ++index) {
        float g = dWq[index];
        if (g > 2.0f) {
          g = 2.0f;
        } else if (g < -2.0f) {
          g = -2.0f;
        }
        model->mq[index] = beta1 * model->mq[index] + (1.0f - beta1) * g;
        model->vq[index] = beta2 * model->vq[index] + (1.0f - beta2) * g * g;
        const float m = model->mq[index] / bias1;
        const float v = model->vq[index] / bias2;
        model->wq[index] -= learning_rate * m / (sqrtf(v) + epsilon);

        g = dWk[index];
        if (g > 2.0f) {
          g = 2.0f;
        } else if (g < -2.0f) {
          g = -2.0f;
        }
        model->mk[index] = beta1 * model->mk[index] + (1.0f - beta1) * g;
        model->vk[index] = beta2 * model->vk[index] + (1.0f - beta2) * g * g;
        const float m_k = model->mk[index] / bias1;
        const float v_k = model->vk[index] / bias2;
        model->wk[index] -= learning_rate * m_k / (sqrtf(v_k) + epsilon);
      }
    }
    epoch_loss /= (float)count;
  }

  free(wq_b);
  free(wk_b);
  free(qbits);
  free(kbits);
  free(qsign);
  free(ksign);
  free(scores);
  free(alpha);
  free(dscore);
  free(dq);
  free(dk);
  free(dWq);
  free(dWk);
  free(order);
  return epoch_loss;
}

float attention_accuracy_float(const AttentionModel *model,
                               const AttentionSamples *samples) {
  if (model == NULL || samples == NULL || samples->count <= 0) {
    return 0.0f;
  }
  int correct = 0;
  for (int sample = 0; sample < samples->count; ++sample) {
    const int predicted = attention_predict_float(
        model, samples->queries + (size_t)sample * model->dim);
    if (predicted == samples->targets[sample]) {
      ++correct;
    }
  }
  return (float)correct / (float)samples->count;
}

float attention_accuracy_int(AttentionModel *model,
                             const AttentionSamples *samples) {
  if (model == NULL || samples == NULL || samples->count <= 0) {
    return 0.0f;
  }
  int correct = 0;
  for (int sample = 0; sample < samples->count; ++sample) {
    const int predicted = attention_predict_int(
        model, samples->queries + (size_t)sample * model->dim);
    if (predicted == samples->targets[sample]) {
      ++correct;
    }
  }
  return (float)correct / (float)samples->count;
}
