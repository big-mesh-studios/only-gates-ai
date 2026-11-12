#ifndef ONLY_GATES_AI_TINY_GPT_H
#define ONLY_GATES_AI_TINY_GPT_H

#include <stdint.h>

/*
 * Integer-only inference for the tiny GPT demo.
 *
 * Weight and activation storage:
 *   - matrices are int8 with one Q16 dequantization scale per output column
 *     (weight value = q8 * scale_q16 / 65536);
 *   - the residual / activation stream is Q16 int32 (value * 65536);
 *   - attention and output probabilities are Q15 int32 (value * 32768).
 *
 * This header deliberately mentions no floating-point type. The conversion
 * from the float32 checkpoint lives in tiny_gpt_quantize.c; the forward pass
 * in tiny_gpt_infer.c is integer-only.
 */

/* GELU lookup table shared by the quantizer and the integer forward pass. */
#define TINY_GPT_GELU_POINTS 1025
#define TINY_GPT_GELU_MIN_Q16 (-1048576) /* -16.0 */
#define TINY_GPT_GELU_MAX_Q16 1048576    /* 16.0 */
#define TINY_GPT_GELU_STEP_Q16 2048      /* 1/32 */

/* DI-Exp clipping range: exp arguments are clamped to 20 nats. */
#define TINY_GPT_SOFTMAX_CLIP_Q12 81920

typedef struct {
  int vocab_size;
  int context;
  int dim;
  int layers;
  int heads;
  int mlp;
} TinyGptConfig;

typedef struct {
  /*
   * Matrices are stored transposed, one contiguous row of `k` per output
   * column: weights[out_col * k] .. weights[out_col * k + k - 1]. The dot
   * product then streams both operands sequentially, which is the shape a
   * MAC array or a SIMD lane expects.
   */
  int8_t *wqkv;         /* (3 * dim) x dim, row-major */
  int32_t *wqkv_scale;  /* 3 * dim */
  int8_t *wo;           /* dim x dim */
  int32_t *wo_scale;    /* dim */
  int8_t *w1;           /* mlp x dim */
  int32_t *w1_scale;    /* mlp */
  int8_t *w2;           /* dim x mlp */
  int32_t *w2_scale;    /* dim */
  int32_t *g1;          /* dim, Q16 */
  int32_t *g2;          /* dim, Q16 */
} TinyGptLayer;

typedef struct {
  TinyGptConfig config;

  int8_t *token_emb;        /* vocab x dim, one scale per row */
  int32_t *token_emb_scale; /* vocab, Q16 */
  int8_t *pos_emb;          /* context x dim, one scale per row */
  int32_t *pos_emb_scale;   /* context, Q16 */
  int8_t *head;             /* vocab x dim, tied to token_emb (same layout) */
  int32_t *head_scale;      /* vocab, Q16 */
  int32_t *g_final;         /* dim, Q16 */
  TinyGptLayer *layers;

  int32_t inv_sqrt_head_dim_q16;

  /*
   * Decode cache. Keys and values are quantized once when their position is
   * computed and reused by later positions, exactly as a hardware accelerator
   * would keep the KV context in on-chip memory:
   *
   *   kv_k[(layer * context + position) * dim + head * head_dim + d]
   *   kv_k_scale[(layer * context + position) * heads + head]
   *
   * The full-context reference path fills every entry; the incremental path
   * fills only the positions it computes.
   */
  int8_t *kv_k;        /* layers * context * dim */
  int8_t *kv_v;        /* layers * context * dim */
  int32_t *kv_k_scale; /* layers * context * heads */
  int32_t *kv_v_scale; /* layers * context * heads */

  /* Scratch, sized once at creation. */
  int32_t *x;         /* context * dim */
  int32_t *x1;        /* context * dim */
  int32_t *h;         /* context * dim */
  int32_t *qkv;       /* context * dim * 3 */
  int32_t *attn;      /* context * dim */
  int32_t *y;         /* context * dim */
  int32_t *z;         /* context * mlp */
  int32_t *gz;        /* context * mlp */
  int32_t *m;         /* context * dim */
  int32_t *logits;    /* context * vocab, Q16 */
  int32_t *probs;     /* vocab, Q15 (also used as score scratch) */
  int8_t *q8;         /* max column count */
  int32_t *scores;    /* context, Q12 nats */
  int32_t *attn_prob; /* context, Q15 */
  int32_t *top_ids;   /* vocab */
  int32_t *top_vals;  /* vocab */
  int32_t *gelu_lut;  /* fixed-point GELU table */
  uint32_t rng;
} TinyGpt;

/* Error codes shared by the creation and inference entry points. */
enum {
  TINY_GPT_OK = 0,
  TINY_GPT_ERROR_ARGUMENT = 1,
  TINY_GPT_ERROR_ALLOC = 2,
  TINY_GPT_ERROR_CONFIG = 3,
};

extern int tiny_gpt_error_code;

/*
 * Runs the integer forward pass over the full context. This is the reference
 * lane: it recomputes every position and must stay available for teaching and
 * for validating faster schedules.
 */
void tiny_gpt_forward(TinyGpt *model, const int32_t *tokens);

/*
 * Runs the integer forward pass over the absolute positions
 * [start, start + count) only, reusing the quantized keys and values cached for
 * positions before `start`. `tokens` is indexed by absolute position, so it
 * must stay valid for the whole window.
 *
 * This is the incremental decode schedule: prefill once with (0, prompt_len),
 * then call (position, 1) per generated token. The result is bit-identical to
 * a full pass over the same window because every operation is row-independent
 * and attention is causal.
 */
void tiny_gpt_forward_range(TinyGpt *model, const int32_t *tokens, int start,
                            int count);

/* Clears the KV cache. Not required before a full `tiny_gpt_forward`. */
void tiny_gpt_reset_cache(TinyGpt *model);

/* Q16 logits for all context positions, row-major context x vocab. */
const int32_t *tiny_gpt_logits(TinyGpt *model);

/* Q16 logits for one absolute position, or NULL if out of range. */
const int32_t *tiny_gpt_logits_row(TinyGpt *model, int position);

/* Debug/tests: exposes an internal Q16 buffer after the last forward pass. */
enum {
  TINY_GPT_BUFFER_X = 0,
  TINY_GPT_BUFFER_X1,
  TINY_GPT_BUFFER_H,
  TINY_GPT_BUFFER_QKV,
  TINY_GPT_BUFFER_ATTN,
  TINY_GPT_BUFFER_Y,
  TINY_GPT_BUFFER_Z,
  TINY_GPT_BUFFER_GZ,
  TINY_GPT_BUFFER_M,
  TINY_GPT_BUFFER_LOGITS,
};
const int32_t *tiny_gpt_buffer(TinyGpt *model, int which);

/*
 * Integer softmax over the last position's logits, divided by a Q16
 * temperature. Writes `vocab_size` Q16 probabilities (sum ~= 65536).
 */
void tiny_gpt_probabilities(TinyGpt *model, int32_t temperature_q16,
                            int32_t *out);

/*
 * Same softmax for one absolute position. This is the incremental-decode
 * variant: after `tiny_gpt_forward_range(model, tokens, position, 1)` the
 * newest logits are at `position`, not necessarily at `context - 1`.
 */
void tiny_gpt_probabilities_at(TinyGpt *model, int position,
                               int32_t temperature_q16, int32_t *out);

/* Seeds the xorshift32 sampler. A zero seed is replaced by a default. */
void tiny_gpt_set_seed(TinyGpt *model, uint32_t seed);

/* Samples from the integer distribution over the top-k tokens. */
int32_t tiny_gpt_sample(TinyGpt *model, int32_t temperature_q16, int32_t top_k);

/* Samples using the logits at `position` (incremental decode hot path). */
int32_t tiny_gpt_sample_at(TinyGpt *model, int position,
                           int32_t temperature_q16, int32_t top_k);

/*
 * Position-aware sampler with an optional repetition penalty. `recent` points
 * to `recent_count` token ids; each distinct id is penalized by `penalty_q16`
 * (Q16; 65536 means no penalty). `recent_count` may be zero and `recent` may
 * be NULL.
 *
 * Penalizing inside the C sampler keeps the per-token decode path free of
 * JavaScript sorting and exponentiation over the whole vocabulary.
 */
int32_t tiny_gpt_sample_at_penalized(
    TinyGpt *model, int position, int32_t temperature_q16, int32_t top_k,
    const int32_t *recent, int recent_count, int32_t penalty_q16);

int tiny_gpt_vocab_size(const TinyGpt *model);
int tiny_gpt_context(const TinyGpt *model);
int tiny_gpt_last_error(void);
void tiny_gpt_destroy(TinyGpt *model);

#endif
