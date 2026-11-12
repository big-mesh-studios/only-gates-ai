#include "test_common.h"

#include <stdlib.h>

#include "../src/tiny_gpt_infer.c"
#include "../src/tiny_gpt_quantize.c"

#define TEST_VOCAB 512
#define TEST_CONTEXT 16
#define TEST_DIM 48
#define TEST_LAYERS 2
#define TEST_HEADS 2
#define TEST_MLP 96

static size_t synthetic_weight_count(void) {
  return (size_t)TEST_VOCAB * TEST_DIM + (size_t)TEST_CONTEXT * TEST_DIM +
         TEST_DIM +
         (size_t)TEST_LAYERS *
             (TEST_DIM + (size_t)TEST_DIM * TEST_DIM * 3 +
              (size_t)TEST_DIM * TEST_DIM + TEST_DIM +
              (size_t)TEST_DIM * TEST_MLP + (size_t)TEST_MLP * TEST_DIM);
}

static float *synthetic_weights(void) {
  const size_t count = synthetic_weight_count();
  float *weights = malloc(count * sizeof(float));
  if (weights == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < count; ++i) {
    weights[i] = (float)((int)(i % 17) - 8) * 0.01f;
  }
  return weights;
}

static void test_int_math(void) {
  expect_true("isqrt(144) == 12", int_isqrt_u64(144) == 12);
  expect_true("isqrt(145) == 12", int_isqrt_u64(145) == 12);
  expect_true("isqrt(0) == 0", int_isqrt_u64(0) == 0);
  expect_true("isqrt(1) == 1", int_isqrt_u64(1) == 1);

  expect_true("div_round(7, 2) == 4", int_div_round(7, 2) == 4);
  expect_true("div_round(-7, 2) == -4", int_div_round(-7, 2) == -4);
  expect_true("div_round(5, 2) == 3", int_div_round(5, 2) == 3);

  expect_true("shift_round(8, 2) == 2", int_shift_round(8, 2) == 2);
  expect_true("shift_round(6, 2) == 2", int_shift_round(6, 2) == 2);
  expect_true("shift_round(10, 2) == 3", int_shift_round(10, 2) == 3);

  /* Q12 DI-Exp: e^0 = 1, e^-1 ~= 0.3679, and monotone decreasing. */
  expect_near("exp(0)", (float)int_exp_neg_q15_q12(0) / 32768.0f, 1.0f,
              1e-4f);
  expect_near("exp(-1)", (float)int_exp_neg_q15_q12(4096) / 32768.0f, 0.3679f,
              2e-3f);
  expect_near("exp(-2)", (float)int_exp_neg_q15_q12(8192) / 32768.0f, 0.1353f,
              2e-3f);
  expect_true("exp monotone",
              int_exp_neg_q15_q12(2048) > int_exp_neg_q15_q12(4096));
  expect_true("exp underflow", int_exp_neg_q15_q12(20 * 4096) <= 1);
}

static void test_create_forward(void) {
  float *weights = synthetic_weights();
  expect_true("allocate synthetic weights", weights != NULL);
  if (weights == NULL) {
    return;
  }

  TinyGpt *model =
      tiny_gpt_create(TEST_VOCAB, TEST_CONTEXT, TEST_DIM, TEST_LAYERS,
                      TEST_HEADS, TEST_MLP, weights);
  expect_true("create model", model != NULL);
  expect_true("create error is ok", tiny_gpt_last_error() == TINY_GPT_OK);
  if (model == NULL) {
    free(weights);
    return;
  }

  expect_true("vocab size", tiny_gpt_vocab_size(model) == TEST_VOCAB);
  expect_true("context", tiny_gpt_context(model) == TEST_CONTEXT);

  int32_t tokens[TEST_CONTEXT];
  for (int i = 0; i < TEST_CONTEXT; ++i) {
    tokens[i] = i;
  }
  tiny_gpt_forward(model, tokens);
  expect_true("forward error is ok", tiny_gpt_last_error() == TINY_GPT_OK);

  const int32_t *logits = tiny_gpt_logits(model);
  expect_true("logits available", logits != NULL);
  if (logits != NULL) {
    int32_t minimum = logits[0];
    int32_t maximum = logits[0];
    int nonzero = 0;
    const int total = TEST_CONTEXT * TEST_VOCAB;
    for (int i = 0; i < total; ++i) {
      if (logits[i] < minimum) {
        minimum = logits[i];
      }
      if (logits[i] > maximum) {
        maximum = logits[i];
      }
      if (logits[i] != 0) {
        ++nonzero;
      }
    }
    expect_true("logits are non-degenerate", maximum > minimum);
    expect_true("logits are mostly nonzero", nonzero > total / 2);
    /* A Q16 logit should stay far inside int32 for this checkpoint. */
    expect_true("logits fit int32", maximum < 1000000 && minimum > -1000000);
  }

  int32_t probabilities[TEST_VOCAB];
  tiny_gpt_probabilities(model, INT_Q16_ONE, probabilities);
  expect_true("probabilities error is ok", tiny_gpt_last_error() == TINY_GPT_OK);

  int64_t sum = 0;
  int32_t max_probability = 0;
  int max_index = -1;
  for (int i = 0; i < TEST_VOCAB; ++i) {
    expect_true("probability is non-negative", probabilities[i] >= 0);
    sum += probabilities[i];
    if (probabilities[i] > max_probability) {
      max_probability = probabilities[i];
      max_index = i;
    }
  }
  /* Q16 probabilities sum to 65536 up to rounding. */
  expect_true("probabilities sum to Q16 one",
              sum > 65536 - TEST_VOCAB && sum < 65536 + TEST_VOCAB);
  expect_true("probabilities peak above uniform",
              max_probability >= INT_Q16_ONE / TEST_VOCAB);
  expect_true("argmax within vocab", max_index >= 0 && max_index < TEST_VOCAB);

  /* Lower temperature concentrates mass; higher temperature flattens it. */
  int32_t cold[TEST_VOCAB];
  int32_t hot[TEST_VOCAB];
  tiny_gpt_probabilities(model, INT_Q16_ONE / 2, cold);
  tiny_gpt_probabilities(model, INT_Q16_ONE * 2, hot);
  expect_true("cold peak >= hot peak", cold[max_index] >= hot[max_index]);

  /* The sampler is deterministic for a fixed seed and stays in range. */
  tiny_gpt_set_seed(model, 12345);
  const int32_t first = tiny_gpt_sample(model, (int32_t)(0.8 * 65536), 40);
  tiny_gpt_set_seed(model, 12345);
  const int32_t second = tiny_gpt_sample(model, (int32_t)(0.8 * 65536), 40);
  expect_true("sample is reproducible", first == second);
  expect_true("sample within vocab", first >= 0 && first < TEST_VOCAB);

  /* Debug buffers are wired up and hold the forward pass activations. */
  const int32_t *x = tiny_gpt_buffer(model, TINY_GPT_BUFFER_X);
  const int32_t *qkv = tiny_gpt_buffer(model, TINY_GPT_BUFFER_QKV);
  const int32_t *attn = tiny_gpt_buffer(model, TINY_GPT_BUFFER_ATTN);
  expect_true("x buffer available", x != NULL);
  expect_true("qkv buffer available", qkv != NULL);
  expect_true("attention buffer available", attn != NULL);
  if (qkv != NULL) {
    int nonzero = 0;
    for (int i = 0; i < TEST_CONTEXT * TEST_DIM * 3; ++i) {
      if (qkv[i] != 0) {
        ++nonzero;
      }
    }
    expect_true("qkv is nonzero", nonzero > 0);
  }
  if (attn != NULL) {
    int nonzero = 0;
    for (int i = 0; i < TEST_CONTEXT * TEST_DIM; ++i) {
      if (attn[i] != 0) {
        ++nonzero;
      }
    }
    expect_true("attention output is nonzero", nonzero > 0);
  }
  expect_true("unknown buffer is null",
              tiny_gpt_buffer(model, 12345) == NULL);

  tiny_gpt_destroy(model);
  free(weights);
}

/*
 * Regression: the MLP up/down projections reuse the same int8 activation
 * scratch as the QKV projection. A model with mlp > dim * 3 used to write past
 * the end of that scratch buffer, corrupting the heap.
 */
static void test_mlp_larger_than_qkv_scratch(void) {
  const int vocab = 64, context = 8, dim = 32, layers = 1, heads = 2, mlp = 128;
  const size_t count =
      (size_t)vocab * dim + (size_t)context * dim + dim +
      (size_t)layers *
          (dim + (size_t)dim * dim * 3 + (size_t)dim * dim + dim +
           (size_t)dim * mlp + (size_t)mlp * dim);
  float *weights = calloc(count, sizeof(float));
  expect_true("allocate large-mlp weights", weights != NULL);
  if (weights == NULL) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    weights[i] = (float)((int)(i % 13) - 6) * 0.02f;
  }

  TinyGpt *model =
      tiny_gpt_create(vocab, context, dim, layers, heads, mlp, weights);
  expect_true("create large-mlp model", model != NULL);
  if (model != NULL) {
    int32_t tokens[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    tiny_gpt_forward(model, tokens);
    expect_true("large-mlp forward error is ok",
                tiny_gpt_last_error() == TINY_GPT_OK);
    const int32_t *logits = tiny_gpt_logits(model);
    expect_true("large-mlp logits available", logits != NULL);
    if (logits != NULL) {
      int nonzero = 0;
      for (int i = 0; i < context * vocab; ++i) {
        if (logits[i] != 0) {
          ++nonzero;
        }
      }
      expect_true("large-mlp logits are nonzero", nonzero > 0);
    }
    tiny_gpt_destroy(model);
  }
  free(weights);
}

/*
 * The reference lane recomputes every position; the decode lane computes a
 * range and reuses cached keys/values. Same arithmetic, different schedule, so
 * every Q16 logit must match exactly.
 */
static void test_incremental_matches_full(void) {
  float *weights = synthetic_weights();
  expect_true("incremental: allocate weights", weights != NULL);
  if (weights == NULL) {
    return;
  }
  TinyGpt *model =
      tiny_gpt_create(TEST_VOCAB, TEST_CONTEXT, TEST_DIM, TEST_LAYERS,
                      TEST_HEADS, TEST_MLP, weights);
  expect_true("incremental: create model", model != NULL);
  if (model == NULL) {
    free(weights);
    return;
  }

  int32_t tokens[TEST_CONTEXT];
  for (int i = 0; i < TEST_CONTEXT; ++i) {
    tokens[i] = (i * 7 + 3) % TEST_VOCAB;
  }
  const int total = TEST_CONTEXT * TEST_VOCAB;

  tiny_gpt_forward(model, tokens);
  int32_t *reference = malloc((size_t)total * sizeof(*reference));
  expect_true("incremental: allocate reference", reference != NULL);
  if (reference == NULL) {
    tiny_gpt_destroy(model);
    free(weights);
    return;
  }
  memcpy(reference, tiny_gpt_logits(model), (size_t)total * sizeof(int32_t));

  int mismatches = 0;
  tiny_gpt_reset_cache(model);
  tiny_gpt_forward_range(model, tokens, 0, 5);
  tiny_gpt_forward_range(model, tokens, 5, TEST_CONTEXT - 5);
  const int32_t *logits = tiny_gpt_logits(model);
  for (int i = 0; i < total; ++i) {
    if (logits[i] != reference[i]) {
      ++mismatches;
    }
  }
  expect_true("incremental: prefill + range matches full", mismatches == 0);

  mismatches = 0;
  tiny_gpt_reset_cache(model);
  for (int position = 0; position < TEST_CONTEXT; ++position) {
    tiny_gpt_forward_range(model, tokens, position, 1);
  }
  logits = tiny_gpt_logits(model);
  for (int i = 0; i < total; ++i) {
    if (logits[i] != reference[i]) {
      ++mismatches;
    }
  }
  expect_true("incremental: one row per step matches full", mismatches == 0);

  const int32_t *row = tiny_gpt_logits_row(model, 7);
  expect_true("incremental: logits_row available", row != NULL);
  if (row != NULL) {
    mismatches = 0;
    for (int i = 0; i < TEST_VOCAB; ++i) {
      if (row[i] != reference[7 * TEST_VOCAB + i]) {
        ++mismatches;
      }
    }
    expect_true("incremental: logits_row matches buffer", mismatches == 0);
  }

  expect_true("incremental: range rejects overflow",
              (tiny_gpt_forward_range(model, tokens, TEST_CONTEXT - 1, 2),
               tiny_gpt_last_error() == TINY_GPT_ERROR_ARGUMENT));
  expect_true("incremental: logits_row rejects out of range",
              tiny_gpt_logits_row(model, TEST_CONTEXT) == NULL);

  free(reference);
  tiny_gpt_destroy(model);
  free(weights);
}

static void test_invalid_config(void) {
  float weights[16] = {0.0f};
  expect_true("reject zero vocab",
              tiny_gpt_create(0, 16, 48, 2, 2, 96, weights) == NULL);
  expect_true("config error code",
              tiny_gpt_last_error() == TINY_GPT_ERROR_CONFIG);
  expect_true("reject null weights",
              tiny_gpt_create(512, 16, 48, 2, 2, 96, NULL) == NULL);
  expect_true("reject indivisible heads",
              tiny_gpt_create(512, 16, 48, 2, 5, 96, weights) == NULL);

  /* NULL model calls must not crash and must report an argument error. */
  tiny_gpt_forward(NULL, NULL);
  expect_true("null forward error",
              tiny_gpt_last_error() == TINY_GPT_ERROR_ARGUMENT);
  expect_true("null logits", tiny_gpt_logits(NULL) == NULL);
  expect_true("null sample", tiny_gpt_sample(NULL, INT_Q16_ONE, 40) == -1);
  tiny_gpt_destroy(NULL);
}

int main(void) {
  test_int_math();
  test_create_forward();
  test_incremental_matches_full();
  test_mlp_larger_than_qkv_scratch();
  test_invalid_config();
  return test_report();
}
