#include "tiny_gpt_quantize.h"

#include "int_math.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Load-time checkpoint conversion. Floating point is confined to this file:
 * it turns the trained float32 weights into int8 matrices with Q16 scales,
 * Q16 norm gains and a Q16 GELU lookup table. tiny_gpt_infer.c only sees the
 * resulting integers.
 */

static int32_t q16_from_float(float value) {
  const float scaled = roundf(value * 65536.0f);
  if (scaled > 2147483647.0f) {
    return 2147483647;
  }
  if (scaled < -2147483648.0f) {
    return (-2147483647 - 1);
  }
  return (int32_t)scaled;
}

static int32_t scale_q16_from_float(float scale) {
  if (!(scale > 0.0f)) {
    return 0;
  }
  const float scaled = roundf(scale * 65536.0f);
  if (scaled < 1.0f) {
    return 1;
  }
  if (scaled > 2147483647.0f) {
    return 2147483647;
  }
  return (int32_t)scaled;
}

static int8_t quantize_value(float value, float inverse) {
  float scaled = roundf(value * inverse);
  if (scaled > 127.0f) {
    scaled = 127.0f;
  } else if (scaled < -127.0f) {
    scaled = -127.0f;
  }
  return (int8_t)scaled;
}

/* One int8 scale per row (embedding lookup tables). */
static void quantize_rows(const float *src, int rows, int cols, int8_t *dst,
                          int32_t *scales) {
  for (int row = 0; row < rows; ++row) {
    const float *source = src + (size_t)row * cols;
    int8_t *target = dst + (size_t)row * cols;
    float max_abs = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float magnitude = fabsf(source[col]);
      if (magnitude > max_abs) {
        max_abs = magnitude;
      }
    }
    if (max_abs <= 0.0f) {
      memset(target, 0, (size_t)cols);
      scales[row] = 0;
      continue;
    }
    scales[row] = scale_q16_from_float(max_abs / 127.0f);
    const float inverse = 127.0f / max_abs;
    for (int col = 0; col < cols; ++col) {
      target[col] = quantize_value(source[col], inverse);
    }
  }
}

/* One int8 scale per output column (linear projections). */
static void quantize_columns(const float *src, int rows, int cols, int8_t *dst,
                             int32_t *scales) {
  for (int col = 0; col < cols; ++col) {
    float max_abs = 0.0f;
    for (int row = 0; row < rows; ++row) {
      const float magnitude = fabsf(src[(size_t)row * cols + col]);
      if (magnitude > max_abs) {
        max_abs = magnitude;
      }
    }
    if (max_abs <= 0.0f) {
      scales[col] = 0;
      for (int row = 0; row < rows; ++row) {
        dst[(size_t)row * cols + col] = 0;
      }
      continue;
    }
    scales[col] = scale_q16_from_float(max_abs / 127.0f);
    const float inverse = 127.0f / max_abs;
    for (int row = 0; row < rows; ++row) {
      const size_t index = (size_t)row * cols + col;
      dst[index] = quantize_value(src[index], inverse);
    }
  }
}

static void quantize_gain(const float *src, int length, int32_t *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = q16_from_float(src[i]);
  }
}

static void build_gelu_lut(int32_t *lut) {
  for (int i = 0; i < TINY_GPT_GELU_POINTS; ++i) {
    const float x = -16.0f + (float)i / 32.0f;
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    const float value = 0.5f * x * (1.0f + tanhf(inner));
    lut[i] = q16_from_float(value);
  }
}

static void *allocate_zero(size_t count, size_t size) {
  return calloc(count == 0 ? 1 : count, size);
}

static int32_t inverse_sqrt_q16(int value) {
  if (value <= 0) {
    return INT_Q16_ONE;
  }
  const uint64_t root = int_isqrt_u64(((uint64_t)value) << 32);
  if (root == 0) {
    return INT_Q16_ONE;
  }
  return (int32_t)(((uint64_t)INT_Q16_ONE * INT_Q16_ONE) / root);
}

static int allocate_layer_weights(TinyGptLayer *layer, int dim, int mlp,
                                  const float **cursor) {
  const float *g1 = *cursor;
  *cursor += dim;
  const float *wqkv = *cursor;
  *cursor += (size_t)dim * dim * 3;
  const float *wo = *cursor;
  *cursor += (size_t)dim * dim;
  const float *g2 = *cursor;
  *cursor += dim;
  const float *w1 = *cursor;
  *cursor += (size_t)dim * mlp;
  const float *w2 = *cursor;
  *cursor += (size_t)mlp * dim;

  layer->wqkv = allocate_zero((size_t)dim * dim * 3, sizeof(int8_t));
  layer->wqkv_scale = allocate_zero((size_t)dim * 3, sizeof(int32_t));
  layer->wo = allocate_zero((size_t)dim * dim, sizeof(int8_t));
  layer->wo_scale = allocate_zero((size_t)dim, sizeof(int32_t));
  layer->w1 = allocate_zero((size_t)dim * mlp, sizeof(int8_t));
  layer->w1_scale = allocate_zero((size_t)mlp, sizeof(int32_t));
  layer->w2 = allocate_zero((size_t)mlp * dim, sizeof(int8_t));
  layer->w2_scale = allocate_zero((size_t)dim, sizeof(int32_t));
  layer->g1 = allocate_zero((size_t)dim, sizeof(int32_t));
  layer->g2 = allocate_zero((size_t)dim, sizeof(int32_t));

  if (layer->wqkv == NULL || layer->wqkv_scale == NULL || layer->wo == NULL ||
      layer->wo_scale == NULL || layer->w1 == NULL || layer->w1_scale == NULL ||
      layer->w2 == NULL || layer->w2_scale == NULL || layer->g1 == NULL ||
      layer->g2 == NULL) {
    return 0;
  }

  quantize_columns(wqkv, dim, dim * 3, layer->wqkv, layer->wqkv_scale);
  quantize_columns(wo, dim, dim, layer->wo, layer->wo_scale);
  quantize_columns(w1, dim, mlp, layer->w1, layer->w1_scale);
  quantize_columns(w2, mlp, dim, layer->w2, layer->w2_scale);
  quantize_gain(g1, dim, layer->g1);
  quantize_gain(g2, dim, layer->g2);
  return 1;
}

static int allocate_scratch(TinyGpt *model) {
  const int context = model->config.context;
  const int dim = model->config.dim;
  const int mlp = model->config.mlp;
  const int vocab = model->config.vocab_size;
  const int head_dim = dim / model->config.heads;

  model->x = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->x1 = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->h = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->qkv = allocate_zero((size_t)context * dim * 3, sizeof(int32_t));
  model->attn = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->y = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->z = allocate_zero((size_t)context * mlp, sizeof(int32_t));
  model->gz = allocate_zero((size_t)context * mlp, sizeof(int32_t));
  model->m = allocate_zero((size_t)context * dim, sizeof(int32_t));
  model->logits = allocate_zero((size_t)context * vocab, sizeof(int32_t));
  model->probs = allocate_zero((size_t)vocab, sizeof(int32_t));
  model->q8 = allocate_zero((size_t)dim * 3, sizeof(int8_t));
  model->k8 = allocate_zero((size_t)context * head_dim, sizeof(int8_t));
  model->v8 = allocate_zero((size_t)context * head_dim, sizeof(int8_t));
  model->k_scale = allocate_zero((size_t)context, sizeof(int32_t));
  model->v_scale = allocate_zero((size_t)context, sizeof(int32_t));
  model->scores = allocate_zero((size_t)context, sizeof(int32_t));
  model->attn_prob = allocate_zero((size_t)context, sizeof(int32_t));
  model->top_ids = allocate_zero((size_t)vocab, sizeof(int32_t));
  model->top_vals = allocate_zero((size_t)vocab, sizeof(int32_t));
  model->gelu_lut =
      allocate_zero(TINY_GPT_GELU_POINTS, sizeof(int32_t));

  return model->x != NULL && model->x1 != NULL && model->h != NULL &&
         model->qkv != NULL && model->attn != NULL && model->y != NULL &&
         model->z != NULL && model->gz != NULL && model->m != NULL &&
         model->logits != NULL && model->probs != NULL && model->q8 != NULL &&
         model->k8 != NULL && model->v8 != NULL && model->k_scale != NULL &&
         model->v_scale != NULL && model->scores != NULL &&
         model->attn_prob != NULL && model->top_ids != NULL &&
         model->top_vals != NULL && model->gelu_lut != NULL;
}

TinyGpt *tiny_gpt_create(int vocab_size, int context, int dim, int layers,
                         int heads, int mlp, const float *packed_weights) {
  tiny_gpt_error_code = TINY_GPT_OK;
  if (vocab_size <= 0 || context <= 0 || dim <= 0 || layers <= 0 ||
      heads <= 0 || mlp <= 0 || packed_weights == NULL || dim % heads != 0) {
    tiny_gpt_error_code = TINY_GPT_ERROR_CONFIG;
    return NULL;
  }

  TinyGpt *model = calloc(1, sizeof(*model));
  if (model == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ALLOC;
    return NULL;
  }

  model->config.vocab_size = vocab_size;
  model->config.context = context;
  model->config.dim = dim;
  model->config.layers = layers;
  model->config.heads = heads;
  model->config.mlp = mlp;
  model->inv_sqrt_head_dim_q16 = inverse_sqrt_q16(dim / heads);
  model->rng = 0x9e3779b9u;

  const float *cursor = packed_weights;
  const float *token_emb = cursor;
  cursor += (size_t)vocab_size * dim;
  const float *pos_emb = cursor;
  cursor += (size_t)context * dim;
  const float *g_final = cursor;
  cursor += dim;

  model->token_emb = allocate_zero((size_t)vocab_size * dim, sizeof(int8_t));
  model->token_emb_scale = allocate_zero((size_t)vocab_size, sizeof(int32_t));
  model->pos_emb = allocate_zero((size_t)context * dim, sizeof(int8_t));
  model->pos_emb_scale = allocate_zero((size_t)context, sizeof(int32_t));
  model->head = allocate_zero((size_t)dim * vocab_size, sizeof(int8_t));
  model->head_scale = allocate_zero((size_t)vocab_size, sizeof(int32_t));
  model->g_final = allocate_zero((size_t)dim, sizeof(int32_t));
  model->layers = allocate_zero((size_t)layers, sizeof(TinyGptLayer));

  if (model->token_emb == NULL || model->token_emb_scale == NULL ||
      model->pos_emb == NULL || model->pos_emb_scale == NULL ||
      model->head == NULL || model->head_scale == NULL ||
      model->g_final == NULL || model->layers == NULL) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ALLOC;
    tiny_gpt_destroy(model);
    return NULL;
  }

  quantize_rows(token_emb, vocab_size, dim, model->token_emb,
                model->token_emb_scale);
  quantize_rows(pos_emb, context, dim, model->pos_emb, model->pos_emb_scale);
  quantize_gain(g_final, dim, model->g_final);

  /*
   * The output head is the tied token embedding, transposed to dim x vocab.
   * Reuse the row-quantized values and scales so the two paths stay identical.
   */
  for (int token = 0; token < vocab_size; ++token) {
    model->head_scale[token] = model->token_emb_scale[token];
    for (int d = 0; d < dim; ++d) {
      model->head[(size_t)d * vocab_size + token] =
          model->token_emb[(size_t)token * dim + d];
    }
  }

  for (int layer = 0; layer < layers; ++layer) {
    if (!allocate_layer_weights(&model->layers[layer], dim, mlp, &cursor)) {
      tiny_gpt_error_code = TINY_GPT_ERROR_ALLOC;
      tiny_gpt_destroy(model);
      return NULL;
    }
  }

  if (!allocate_scratch(model)) {
    tiny_gpt_error_code = TINY_GPT_ERROR_ALLOC;
    tiny_gpt_destroy(model);
    return NULL;
  }

  build_gelu_lut(model->gelu_lut);

  return model;
}
