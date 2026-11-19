#include "test_common.h"

#include <stdlib.h>
#include <string.h>

#include "../src/masked_diffusion.c"
#include "../src/tiny_gpt_infer.c"
#include "../src/tiny_gpt_quantize.c"

#define TEST_VOCAB 64
#define TEST_CONTEXT 16
#define TEST_DIM 32
#define TEST_LAYERS 1
#define TEST_HEADS 2
#define TEST_MLP 64

/* The C sampler never emits pad (0) or the mask id. */
#define TEST_PAD 0
#define TEST_MASK (TEST_VOCAB - 1)

static size_t synthetic_weight_count(void) {
  return (size_t)TEST_VOCAB * TEST_DIM + (size_t)TEST_CONTEXT * TEST_DIM +
         TEST_DIM +
         (size_t)TEST_LAYERS *
             (TEST_DIM + (size_t)TEST_DIM * TEST_DIM * 3 +
              (size_t)TEST_DIM * TEST_DIM + TEST_DIM +
              (size_t)TEST_DIM * TEST_MLP + (size_t)TEST_MLP * TEST_DIM);
}

static TinyGpt *create_model(void) {
  const size_t count = synthetic_weight_count();
  float *weights = malloc(count * sizeof(float));
  if (weights == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < count; ++i) {
    weights[i] = (float)((int)(i % 17) - 8) * 0.01f;
  }
  TinyGpt *model =
      tiny_gpt_create(TEST_VOCAB, TEST_CONTEXT, TEST_DIM, TEST_LAYERS,
                      TEST_HEADS, TEST_MLP, weights);
  free(weights);
  return model;
}

/*
 * A causal model cannot see the last token from position 0; a bidirectional
 * model must. This pins down the difference between the two forward paths.
 */
static void test_bidirectional_is_not_causal(TinyGpt *model) {
  int32_t a[TEST_CONTEXT];
  int32_t b[TEST_CONTEXT];
  for (int i = 0; i < TEST_CONTEXT; ++i) {
    a[i] = (i * 5 + 1) % TEST_VOCAB;
    b[i] = a[i];
  }
  /* Change only the final token. */
  b[TEST_CONTEXT - 1] = (a[TEST_CONTEXT - 1] + 7) % TEST_VOCAB;
  if (b[TEST_CONTEXT - 1] == a[TEST_CONTEXT - 1]) {
    b[TEST_CONTEXT - 1] = (a[TEST_CONTEXT - 1] + 1) % TEST_VOCAB;
  }

  int32_t causal_a[TEST_VOCAB];
  int32_t causal_b[TEST_VOCAB];

  tiny_gpt_forward(model, a);
  memcpy(causal_a, tiny_gpt_logits_row(model, 0),
         sizeof(causal_a));
  tiny_gpt_forward(model, b);
  memcpy(causal_b, tiny_gpt_logits_row(model, 0),
         sizeof(causal_b));

  int causal_differences = 0;
  for (int i = 0; i < TEST_VOCAB; ++i) {
    if (causal_a[i] != causal_b[i]) {
      ++causal_differences;
    }
  }
  expect_true("causal position 0 ignores later tokens",
              causal_differences == 0);

  int32_t bidirectional_a[TEST_VOCAB];
  int32_t bidirectional_b[TEST_VOCAB];

  tiny_gpt_set_bidirectional(model, 1);
  expect_true("model records bidirectional flag", model->bidirectional != 0);
  tiny_gpt_forward_bidirectional(model, a, TEST_CONTEXT);
  expect_true("bidirectional forward error is ok",
              tiny_gpt_last_error() == TINY_GPT_OK);
  memcpy(bidirectional_a, tiny_gpt_logits_row(model, 0),
         sizeof(bidirectional_a));
  tiny_gpt_forward_bidirectional(model, b, TEST_CONTEXT);
  memcpy(bidirectional_b, tiny_gpt_logits_row(model, 0),
         sizeof(bidirectional_b));

  int bidirectional_differences = 0;
  for (int i = 0; i < TEST_VOCAB; ++i) {
    if (bidirectional_a[i] != bidirectional_b[i]) {
      ++bidirectional_differences;
    }
  }
  expect_true("bidirectional position 0 sees later tokens",
              bidirectional_differences > 0);
  expect_true("bidirectional flag can be cleared",
              (tiny_gpt_set_bidirectional(model, 0), model->bidirectional == 0));
}

/*
 * The denoiser should unmask ceil(remaining / steps_left) positions per call,
 * never emit pad or mask, and finish with a fully filled answer region.
 */
static void test_masked_step(TinyGpt *model) {
  const int prompt_length = 3;
  const int length = 10;
  const int total_steps = 4;
  int32_t tokens[TEST_CONTEXT];
  for (int i = 0; i < length; ++i) {
    tokens[i] = i < prompt_length ? (i + 1) : TEST_MASK;
  }
  int32_t prompt_copy[3];
  memcpy(prompt_copy, tokens, sizeof(prompt_copy));

  tiny_gpt_set_seed(model, 2024);

  const int expected_remaining[4] = {5, 3, 1, 0};
  int remaining = length - prompt_length;
  for (int step = 0; step < total_steps; ++step) {
    remaining = tiny_masked_diffusion_step(model, tokens, length,
                                           prompt_length, TEST_MASK,
                                           INT_Q16_ONE, step, total_steps);
    expect_true("masked step returns a remaining count", remaining >= 0);
    expect_true("masked step follows the unmask schedule",
                remaining == expected_remaining[step]);
    if (remaining < 0) {
      return;
    }
  }

  expect_true("masked step leaves no masks", remaining == 0);
  expect_true("masked step preserves the prompt",
              memcmp(prompt_copy, tokens, sizeof(prompt_copy)) == 0);

  int masks = 0;
  int pads = 0;
  for (int i = prompt_length; i < length; ++i) {
    if (tokens[i] == TEST_MASK) {
      ++masks;
    }
    if (tokens[i] == TEST_PAD) {
      ++pads;
    }
    expect_true("answer token stays within vocab",
                tokens[i] >= 0 && tokens[i] < TEST_VOCAB);
  }
  expect_true("answer has no mask tokens", masks == 0);
  expect_true("answer has no pad tokens", pads == 0);
}

static void test_masked_step_arguments(TinyGpt *model) {
  int32_t tokens[TEST_CONTEXT] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  expect_true("null model rejected",
              tiny_masked_diffusion_step(NULL, tokens, 10, 3, TEST_MASK,
                                         INT_Q16_ONE, 0, 4) == -1);
  expect_true("null tokens rejected",
              tiny_masked_diffusion_step(model, NULL, 10, 3, TEST_MASK,
                                         INT_Q16_ONE, 0, 4) == -1);
  expect_true("empty answer rejected",
              tiny_masked_diffusion_step(model, tokens, 10, 10, TEST_MASK,
                                         INT_Q16_ONE, 0, 4) == -1);
  expect_true("negative step rejected",
              tiny_masked_diffusion_step(model, tokens, 10, 3, TEST_MASK,
                                         INT_Q16_ONE, -1, 4) == -1);
  expect_true("zero total steps rejected",
              tiny_masked_diffusion_step(model, tokens, 10, 3, TEST_MASK,
                                         INT_Q16_ONE, 0, 0) == -1);

  /* A fully filled answer is a no-op. */
  expect_true("fully filled answer returns zero",
              tiny_masked_diffusion_step(model, tokens, 10, 3, TEST_MASK,
                                         INT_Q16_ONE, 0, 4) == 0);
}

static void test_masked_determinism(void) {
  TinyGpt *model = create_model();
  expect_true("determinism: create model", model != NULL);
  if (model == NULL) {
    return;
  }

  int32_t first[TEST_CONTEXT];
  int32_t second[TEST_CONTEXT];
  for (int i = 0; i < TEST_CONTEXT; ++i) {
    first[i] = i < 4 ? (i + 2) : TEST_MASK;
    second[i] = first[i];
  }

  tiny_gpt_set_seed(model, 99);
  for (int step = 0; step < 6; ++step) {
    tiny_masked_diffusion_step(model, first, TEST_CONTEXT, 4, TEST_MASK,
                               INT_Q16_ONE, step, 6);
  }
  tiny_gpt_set_seed(model, 99);
  for (int step = 0; step < 6; ++step) {
    tiny_masked_diffusion_step(model, second, TEST_CONTEXT, 4, TEST_MASK,
                               INT_Q16_ONE, step, 6);
  }

  expect_true("same seed reproduces the denoising trajectory",
              memcmp(first, second, sizeof(first)) == 0);

  tiny_gpt_destroy(model);
}

int main(void) {
  TinyGpt *model = create_model();
  expect_true("create model", model != NULL);
  if (model != NULL) {
    test_bidirectional_is_not_causal(model);
    test_masked_step(model);
    test_masked_step_arguments(model);
    tiny_gpt_destroy(model);
  }
  test_masked_determinism();
  return test_report();
}
