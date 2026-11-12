#include "tiny_gpt.h"

#include "int_math.h"
#include "tiny_gpt_dot_simd.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Integer-only forward pass for the tiny GPT.
 *
 * Everything below uses int8/int32/int64 arithmetic. The float checkpoint is
 * converted once in tiny_gpt_quantize.c; no floating-point type, literal or
 * math.h function appears in this file.
 */

int tiny_gpt_error_code = TINY_GPT_OK;

static uint32_t next_random(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

/*
 * Dynamically quantizes one activation row (Q16) to int8. The returned scale
 * is in Q16 units, so `value ~= q8 * scale / 65536`.
 */
static void quantize_activation_row(const int32_t *src, int count, int8_t *dst,
                                    int32_t *scale_out) {
  int32_t max_abs = 0;
  for (int i = 0; i < count; ++i) {
    const int32_t value = src[i];
    const int32_t magnitude = value < 0 ? -value : value;
    if (magnitude > max_abs) {
      max_abs = magnitude;
    }
  }

  if (max_abs == 0) {
    *scale_out = 0;
    memset(dst, 0, (size_t)count);
    return;
  }
  if (max_abs <= 127) {
    *scale_out = 1;
    for (int i = 0; i < count; ++i) {
      dst[i] = (int8_t)src[i];
    }
    return;
  }

  *scale_out = (max_abs + 63) / 127;
  for (int i = 0; i < count; ++i) {
    int64_t quantized = int_div_round((int64_t)src[i] * 127, max_abs);
    if (quantized > 127) {
      quantized = 127;
    } else if (quantized < -127) {
      quantized = -127;
    }
    dst[i] = (int8_t)quantized;
  }
}

/*
 * DI-MatMul: per-token int8 activations, int8 weights with per-column Q16
 * scales, int32 accumulation, single int64 requantization to Q16 output.
 *
 * Weights are stored one contiguous row per output column, so the inner loop
 * is a sequential dot product over two int8 rows.
 */
static void matmul_q16(const int32_t *x, int rows, int cols,
                       const int8_t *weights, const int32_t *weight_scale,
                       int out_cols, int8_t *scratch, int32_t *out) {
  for (int row = 0; row < rows; ++row) {
    int32_t activation_scale = 0;
    quantize_activation_row(x + (size_t)row * cols, cols, scratch,
                            &activation_scale);
    int32_t *target = out + (size_t)row * out_cols;
    for (int j = 0; j < out_cols; ++j) {
      const int8_t *weight_row = weights + (size_t)j * cols;
      const int32_t accumulator = tiny_gpt_dot_i8(scratch, weight_row, cols);
      const int64_t product =
          (int64_t)accumulator * activation_scale * weight_scale[j];
      target[j] = (int32_t)int_shift_round(product, 16);
    }
  }
}

/* DI-Norm: int64 sum of squares, integer sqrt, Q16 reciprocal and gain. */
static void rmsnorm_q16(const int32_t *x, int rows, int cols,
                        const int32_t *gain, int32_t *out) {
  /* round(1e-5 * 2^32), matching the float path's epsilon. */
  const int64_t epsilon_q32 = 42950;
  for (int row = 0; row < rows; ++row) {
    const int32_t *source = x + (size_t)row * cols;
    int32_t *target = out + (size_t)row * cols;
    int64_t sum_squares = 0;
    for (int i = 0; i < cols; ++i) {
      const int64_t value = source[i];
      sum_squares += value * value;
    }
    const int64_t mean_q32 = sum_squares / cols + epsilon_q32;
    int64_t rms_q16 = (int64_t)int_isqrt_u64((uint64_t)mean_q32);
    if (rms_q16 == 0) {
      rms_q16 = 1;
    }
    for (int i = 0; i < cols; ++i) {
      const int64_t scaled = (int64_t)source[i] * gain[i];
      target[i] = (int32_t)int_div_round(scaled, rms_q16);
    }
  }
}

/* Integer GELU: Q16 lookup table with linear interpolation. */
static int32_t gelu_q16(const TinyGpt *model, int32_t x) {
  if (x <= TINY_GPT_GELU_MIN_Q16) {
    return 0;
  }
  if (x >= TINY_GPT_GELU_MAX_Q16) {
    return x;
  }
  const int32_t offset = x - TINY_GPT_GELU_MIN_Q16;
  const int index = offset / TINY_GPT_GELU_STEP_Q16;
  const int32_t remainder = offset & (TINY_GPT_GELU_STEP_Q16 - 1);
  const int32_t a = model->gelu_lut[index];
  const int32_t b = model->gelu_lut[index + 1];
  /* STEP_Q16 is 2048 = 1 << 11, so the interpolation shift is 11. */
  return a + (int32_t)(((int64_t)(b - a) * remainder) >> 11);
}

/*
 * Integer attention. Q/K/V are quantized per head and per token, scores use
 * int8 dot products, DI-Exp with clipping produces Q15 probabilities, and the
 * value aggregation uses the corrected P * V indexing (sum over keys).
 *
 * `qkv` points at the first row of the computed range; local row `t`
 * corresponds to absolute position `start + t`. Keys and values for those
 * positions are written into the per-layer cache, and the attention window
 * reads cached rows 0..position, so an incremental decode computes one row and
 * reuses everything already cached.
 */
static void attention_q16(TinyGpt *model, int layer, const int32_t *qkv,
                          int start, int count) {
  const TinyGptConfig *config = &model->config;
  const int context = config->context;
  const int dim = config->dim;
  const int heads = config->heads;
  const int head_dim = dim / heads;
  const int qkv_stride = dim * 3;
  const size_t layer_base = (size_t)layer * context;

  for (int head = 0; head < heads; ++head) {
    const int head_offset = head * head_dim;

    /* Quantize and cache the keys and values for the new positions. */
    for (int t = 0; t < count; ++t) {
      const int position = start + t;
      const size_t cache_row = (layer_base + (size_t)position) * dim +
                               (size_t)head_offset;
      const size_t scale_index =
          (layer_base + (size_t)position) * heads + (size_t)head;
      const int32_t *key =
          qkv + (size_t)t * qkv_stride + dim + head_offset;
      const int32_t *value =
          qkv + (size_t)t * qkv_stride + dim * 2 + head_offset;
      quantize_activation_row(key, head_dim, model->kv_k + cache_row,
                              &model->kv_k_scale[scale_index]);
      quantize_activation_row(value, head_dim, model->kv_v + cache_row,
                              &model->kv_v_scale[scale_index]);
    }

    for (int t = 0; t < count; ++t) {
      const int position = start + t;
      const int32_t *query = qkv + (size_t)t * qkv_stride + head_offset;
      int32_t query_scale = 0;
      quantize_activation_row(query, head_dim, model->q8, &query_scale);

      int32_t row_max = -2147483647;
      for (int j = 0; j <= position; ++j) {
        const size_t cache_row =
            (layer_base + (size_t)j) * dim + (size_t)head_offset;
        const size_t scale_index =
            (layer_base + (size_t)j) * heads + (size_t)head;
        const int8_t *key8 = model->kv_k + cache_row;
        const int32_t dot = tiny_gpt_dot_i8(model->q8, key8, head_dim);
        /*
         * product = dot * qscale * kscale * inv_sqrt_q16. With qscale and
         * kscale in Q16, the real score is product / 2^48. Shift by 36 to
         * land in Q12 (1/4096 nat), which keeps the softmax error well below
         * the Q8 variant's ~1e-3.
         */
        const int64_t product = (int64_t)dot * query_scale *
                                model->kv_k_scale[scale_index] *
                                model->inv_sqrt_head_dim_q16;
        int64_t score = int_shift_round(product, 36);
        if (score > TINY_GPT_SOFTMAX_CLIP_Q12) {
          score = TINY_GPT_SOFTMAX_CLIP_Q12;
        } else if (score < -TINY_GPT_SOFTMAX_CLIP_Q12) {
          score = -TINY_GPT_SOFTMAX_CLIP_Q12;
        }
        model->scores[j] = (int32_t)score;
        if (model->scores[j] > row_max) {
          row_max = model->scores[j];
        }
      }

      int32_t total = 0;
      for (int j = 0; j <= position; ++j) {
        int32_t delta = row_max - model->scores[j];
        if (delta < 0) {
          delta = 0;
        } else if (delta > TINY_GPT_SOFTMAX_CLIP_Q12) {
          delta = TINY_GPT_SOFTMAX_CLIP_Q12;
        }
        const int32_t weight = int_exp_neg_q15_q12(delta);
        model->attn_prob[j] = weight;
        total += weight;
      }
      for (int j = position + 1; j < context; ++j) {
        model->attn_prob[j] = 0;
      }

      if (total > 0) {
        for (int j = 0; j <= position; ++j) {
          model->attn_prob[j] = (int32_t)int_div_round(
              (int64_t)model->attn_prob[j] * INT_Q15_ONE, total);
        }
      } else {
        const int32_t uniform = INT_Q15_ONE / (position + 1);
        for (int j = 0; j <= position; ++j) {
          model->attn_prob[j] = uniform;
        }
      }

      for (int d = 0; d < head_dim; ++d) {
        int64_t accumulator = 0;
        for (int j = 0; j <= position; ++j) {
          const size_t cache_row =
              (layer_base + (size_t)j) * dim + (size_t)head_offset;
          const size_t scale_index =
              (layer_base + (size_t)j) * heads + (size_t)head;
          accumulator += (int64_t)model->attn_prob[j] *
                         (int64_t)model->kv_v[cache_row + d] *
                         model->kv_v_scale[scale_index];
        }
        model->attn[(size_t)position * dim + head_offset + d] =
            (int32_t)int_shift_round(accumulator, 15);
      }
    }
  }
}

/* Integer softmax over the last position. Writes vocab Q15 probabilities. */
static void softmax_last_q15(TinyGpt *model, int32_t temperature_q16,
                             int32_t *out_q15) {
  const int vocab = model->config.vocab_size;
  const int offset = (model->config.context - 1) * vocab;
  const int32_t *logits = model->logits + offset;

  if (temperature_q16 <= 0) {
    temperature_q16 = 655; /* 0.01 */
  } else if (temperature_q16 < 655) {
    temperature_q16 = 655;
  }

  int32_t maximum = -2147483647;
  for (int i = 0; i < vocab; ++i) {
    const int64_t scaled =
        int_div_round((int64_t)logits[i] * INT_Q16_ONE, temperature_q16);
    /* Q16 (logit / temperature) -> Q12 nats for DI-Exp. */
    const int32_t score = (int32_t)(scaled >> 4);
    model->probs[i] = score;
    if (score > maximum) {
      maximum = score;
    }
  }

  int32_t total = 0;
  for (int i = 0; i < vocab; ++i) {
    int32_t delta = maximum - model->probs[i];
    if (delta < 0) {
      delta = 0;
    } else if (delta > TINY_GPT_SOFTMAX_CLIP_Q12) {
      delta = TINY_GPT_SOFTMAX_CLIP_Q12;
    }
    const int32_t weight = int_exp_neg_q15_q12(delta);
    model->probs[i] = weight;
    total += weight;
  }

  if (total <= 0) {
    const int32_t uniform = INT_Q15_ONE / vocab;
    for (int i = 0; i < vocab; ++i) {
      out_q15[i] = uniform;
    }
    return;
  }

  for (int i = 0; i < vocab; ++i) {
    out_q15[i] = (int32_t)int_div_round(
        (int64_t)model->probs[i] * INT_Q15_ONE, total);
  }
}

/*
 * Shared forward pass over the absolute positions [start, start + count).
 * Every stage is row-independent, so computing a subset produces exactly the
 * rows a full pass would produce; attention reads cached K/V for all earlier
 * positions.
 */
static void forward_range_impl(TinyGpt *model, const int32_t *tokens,
                               int start, int count) {
  const TinyGptConfig *config = &model->config;
  const int context = config->context;
  const int dim = config->dim;
  const int mlp = config->mlp;
  const int vocab = config->vocab_size;
  const size_t row_offset = (size_t)start * dim;
  const size_t qkv_offset = (size_t)start * dim * 3;
  const size_t mlp_offset = (size_t)start * mlp;

  /* Embeddings: token and position tables are int8 with Q16 row scales. */
  for (int t = start; t < start + count; ++t) {
    int token = tokens[t];
    if (token < 0 || token >= vocab) {
      token = 0;
    }
    const int8_t *token_row = model->token_emb + (size_t)token * dim;
    const int32_t token_scale = model->token_emb_scale[token];
    const int8_t *position_row = model->pos_emb + (size_t)t * dim;
    const int32_t position_scale = model->pos_emb_scale[t];
    for (int d = 0; d < dim; ++d) {
      int64_t value = (int64_t)token_row[d] * token_scale +
                      (int64_t)position_row[d] * position_scale;
      if (value > 2147483647) {
        value = 2147483647;
      } else if (value < -2147483647) {
        value = -2147483647;
      }
      model->x[(size_t)t * dim + d] = (int32_t)value;
    }
  }

  for (int layer = 0; layer < config->layers; ++layer) {
    const TinyGptLayer *weights = &model->layers[layer];

    rmsnorm_q16(model->x + row_offset, count, dim, weights->g1,
                model->h + row_offset);
    matmul_q16(model->h + row_offset, count, dim, weights->wqkv,
               weights->wqkv_scale, dim * 3, model->q8,
               model->qkv + qkv_offset);
    attention_q16(model, layer, model->qkv + qkv_offset, start, count);
    matmul_q16(model->attn + row_offset, count, dim, weights->wo,
               weights->wo_scale, dim, model->q8, model->y + row_offset);

    for (int i = 0; i < count * dim; ++i) {
      const size_t index = row_offset + (size_t)i;
      model->x1[index] = model->x[index] + model->y[index];
    }

    rmsnorm_q16(model->x1 + row_offset, count, dim, weights->g2,
                model->h + row_offset);
    matmul_q16(model->h + row_offset, count, dim, weights->w1,
               weights->w1_scale, mlp, model->q8, model->z + mlp_offset);

    for (int i = 0; i < count * mlp; ++i) {
      const size_t index = mlp_offset + (size_t)i;
      model->gz[index] = gelu_q16(model, model->z[index]);
    }

    matmul_q16(model->gz + mlp_offset, count, mlp, weights->w2,
               weights->w2_scale, dim, model->q8, model->m + row_offset);

    for (int i = 0; i < count * dim; ++i) {
      const size_t index = row_offset + (size_t)i;
      model->x[index] = model->x1[index] + model->m[index];
    }
  }

  rmsnorm_q16(model->x + row_offset, count, dim, model->g_final,
              model->h + row_offset);
  matmul_q16(model->h + row_offset, count, dim, model->head,
             model->head_scale, vocab, model->q8, model->logits + (size_t)start * vocab);

  tiny_gpt_error_code = TINY_GPT_OK;
}

void tiny_gpt_forward_range(TinyGpt *model, const int32_t *tokens, int start,
                            int count) {
  if (model == NULL || tokens == NULL || start < 0 || count < 0 ||
      start > model->config.context ||
      count > model->config.context - start) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return;
  }
  if (count == 0) {
    tiny_gpt_error_code = TINY_GPT_OK;
    return;
  }
  forward_range_impl(model, tokens, start, count);
}

void tiny_gpt_forward(TinyGpt *model, const int32_t *tokens) {
  if (model == NULL || tokens == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return;
  }
  forward_range_impl(model, tokens, 0, model->config.context);
}

void tiny_gpt_reset_cache(TinyGpt *model) {
  if (model == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return;
  }
  const size_t entries =
      (size_t)model->config.layers * (size_t)model->config.context;
  memset(model->kv_k, 0, entries * (size_t)model->config.dim);
  memset(model->kv_v, 0, entries * (size_t)model->config.dim);
  memset(model->kv_k_scale, 0, entries * (size_t)model->config.heads *
                                  sizeof(int32_t));
  memset(model->kv_v_scale, 0, entries * (size_t)model->config.heads *
                                  sizeof(int32_t));
  tiny_gpt_error_code = TINY_GPT_OK;
}

const int32_t *tiny_gpt_logits_row(TinyGpt *model, int position) {
  if (model == NULL || position < 0 || position >= model->config.context) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return NULL;
  }
  return model->logits + (size_t)position * model->config.vocab_size;
}

const int32_t *tiny_gpt_logits(TinyGpt *model) {
  return model == NULL ? NULL : model->logits;
}

const int32_t *tiny_gpt_buffer(TinyGpt *model, int which) {
  if (model == NULL) {
    return NULL;
  }
  switch (which) {
    case TINY_GPT_BUFFER_X:
      return model->x;
    case TINY_GPT_BUFFER_X1:
      return model->x1;
    case TINY_GPT_BUFFER_H:
      return model->h;
    case TINY_GPT_BUFFER_QKV:
      return model->qkv;
    case TINY_GPT_BUFFER_ATTN:
      return model->attn;
    case TINY_GPT_BUFFER_Y:
      return model->y;
    case TINY_GPT_BUFFER_Z:
      return model->z;
    case TINY_GPT_BUFFER_GZ:
      return model->gz;
    case TINY_GPT_BUFFER_M:
      return model->m;
    case TINY_GPT_BUFFER_LOGITS:
      return model->logits;
    default:
      return NULL;
  }
}

void tiny_gpt_probabilities(TinyGpt *model, int32_t temperature_q16,
                            int32_t *out) {
  if (model == NULL || out == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return;
  }
  softmax_last_q15(model, temperature_q16, model->probs);
  for (int i = 0; i < model->config.vocab_size; ++i) {
    out[i] = model->probs[i] * 2;
  }
}

void tiny_gpt_set_seed(TinyGpt *model, uint32_t seed) {
  if (model == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return;
  }
  model->rng = seed == 0 ? 0x9e3779b9u : seed;
}

int32_t tiny_gpt_sample(TinyGpt *model, int32_t temperature_q16,
                        int32_t top_k) {
  if (model == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ARGUMENT;
    return -1;
  }

  const int vocab = model->config.vocab_size;
  softmax_last_q15(model, temperature_q16, model->probs);

  if (top_k < 1) {
    top_k = 1;
  } else if (top_k > vocab) {
    top_k = vocab;
  }

  /*
   * Partial insertion sort: keep the top-k probabilities in descending order.
   * k is small (typically 40), so O(vocab * k) is fine.
   */
  int count = 0;
  for (int id = 0; id < vocab; ++id) {
    const int32_t value = model->probs[id];
    if (value <= 0) {
      continue;
    }
    if (count < top_k) {
      int position = count;
      while (position > 0 && model->top_vals[position - 1] < value) {
        model->top_vals[position] = model->top_vals[position - 1];
        model->top_ids[position] = model->top_ids[position - 1];
        --position;
      }
      model->top_vals[position] = value;
      model->top_ids[position] = id;
      ++count;
    } else if (value > model->top_vals[top_k - 1]) {
      int position = top_k - 1;
      while (position > 0 && model->top_vals[position - 1] < value) {
        model->top_vals[position] = model->top_vals[position - 1];
        model->top_ids[position] = model->top_ids[position - 1];
        --position;
      }
      model->top_vals[position] = value;
      model->top_ids[position] = id;
    }
  }

  if (count <= 0) {
    return 0;
  }

  int64_t total = 0;
  for (int i = 0; i < count; ++i) {
    total += model->top_vals[i];
  }
  if (total <= 0) {
    return model->top_ids[0];
  }

  const uint32_t random = next_random(&model->rng);
  const uint64_t target = ((uint64_t)random * (uint64_t)total) >> 32;
  uint64_t cumulative = 0;
  for (int i = 0; i < count; ++i) {
    cumulative += (uint64_t)model->top_vals[i];
    if (target < cumulative) {
      return model->top_ids[i];
    }
  }
  return model->top_ids[count - 1];
}

int tiny_gpt_vocab_size(const TinyGpt *model) {
  return model == NULL ? 0 : model->config.vocab_size;
}

int tiny_gpt_context(const TinyGpt *model) {
  return model == NULL ? 0 : model->config.context;
}

int tiny_gpt_last_error(void) {
  return tiny_gpt_error_code;
}

void tiny_gpt_destroy(TinyGpt *model) {
  if (model == NULL) {
    return;
  }
  if (model->layers != NULL) {
    for (int layer = 0; layer < model->config.layers; ++layer) {
      TinyGptLayer *weights = &model->layers[layer];
      free(weights->wqkv);
      free(weights->wqkv_scale);
      free(weights->wo);
      free(weights->wo_scale);
      free(weights->w1);
      free(weights->w1_scale);
      free(weights->w2);
      free(weights->w2_scale);
      free(weights->g1);
      free(weights->g2);
    }
    free(model->layers);
  }
  free(model->token_emb);
  free(model->token_emb_scale);
  free(model->pos_emb);
  free(model->pos_emb_scale);
  free(model->head);
  free(model->head_scale);
  free(model->g_final);
  free(model->x);
  free(model->x1);
  free(model->h);
  free(model->qkv);
  free(model->attn);
  free(model->y);
  free(model->z);
  free(model->gz);
  free(model->m);
  free(model->logits);
  free(model->probs);
  free(model->q8);
  free(model->kv_k);
  free(model->kv_v);
  free(model->kv_k_scale);
  free(model->kv_v_scale);
  free(model->scores);
  free(model->attn_prob);
  free(model->top_ids);
  free(model->top_vals);
  free(model->gelu_lut);
  free(model);
}
