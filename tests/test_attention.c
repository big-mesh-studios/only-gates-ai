#include "test_common.h"

#include <stdlib.h>

#include "../src/attention.c"

static void test_multi_word_integer_path(void) {
  const int dim = 128;
  const int key_count = 5;
  AttentionModel *model = attention_model_create(dim, key_count, 0x55u);
  AttentionSamples *samples =
      attention_samples_create(model, 16, 10, 0x99u, 0x77u);
  if (model == NULL || samples == NULL) {
    expect_true("create 128-bit model", 0);
    attention_samples_destroy(samples);
    attention_model_destroy(model);
    return;
  }

  attention_quantize(model);
  int mismatches = 0;
  int32_t alpha[5];
  for (int sample = 0; sample < samples->count; ++sample) {
    const uint8_t *query = samples->queries + (size_t)sample * dim;
    const int reference = attention_predict_float(model, query);
    const int integer = attention_forward_int(model, query, alpha);
    if (reference != integer) {
      ++mismatches;
    }
    int32_t sum = 0;
    for (int key = 0; key < key_count; ++key) {
      sum += alpha[key];
    }
    expect_true("128-bit softmax normalizes", sum >= 32700 && sum <= 32800);
  }
  expect_true("128-bit integer matches float", mismatches == 0);

  attention_samples_destroy(samples);
  attention_model_destroy(model);
}

int main(void) {
  /* Fixed-point exp sanity checks (Q15 values). */
  expect_true("exp_neg_q15(0)", exp_neg_q15(0) == 32768);
  expect_near("exp_neg_q15(1.0)", (float)exp_neg_q15(256), 12055.0f, 20.0f);
  expect_near("exp_neg_q15(2.0)", (float)exp_neg_q15(512), 4435.0f, 10.0f);
  expect_near("exp_neg_q15(4.0)", (float)exp_neg_q15(1024), 600.0f, 5.0f);

  const int dim = 24;
  const int key_count = 6;

  AttentionModel *model = attention_model_create(dim, key_count, 0x12345678u);
  if (model == NULL) {
    expect_true("create attention model", 0);
    return test_report();
  }

  AttentionSamples *train =
      attention_samples_create(model, 128, 10, 0xabcdu, 0x1111u);
  AttentionSamples *test =
      attention_samples_create(model, 64, 15, 0xabcdu, 0x2222u);
  if (train == NULL || test == NULL) {
    expect_true("create attention samples", 0);
    attention_samples_destroy(train);
    attention_samples_destroy(test);
    attention_model_destroy(model);
    return test_report();
  }

  const float before = attention_accuracy_float(model, test);
  const float loss = attention_train(model, train, 120, 0.02f, 0x3333u);
  const float after = attention_accuracy_float(model, test);

  attention_quantize(model);
  const float integer_accuracy = attention_accuracy_int(model, test);

  int mismatches = 0;
  int bad_softmax = 0;
  int32_t *alpha = malloc((size_t)key_count * sizeof(*alpha));
  if (alpha == NULL) {
    expect_true("allocate alpha", 0);
  } else {
    for (int sample = 0; sample < test->count; ++sample) {
      const uint8_t *query = test->queries + (size_t)sample * test->dim;
      const int reference = attention_predict_float(model, query);
      const int integer = attention_forward_int(model, query, alpha);
      if (reference != integer) {
        ++mismatches;
      }
      int32_t sum = 0;
      for (int key = 0; key < key_count; ++key) {
        sum += alpha[key];
      }
      if (sum < 32700 || sum > 32800) {
        ++bad_softmax;
      }
    }
    free(alpha);
  }

  printf("loss=%.4f accuracy before=%.3f float=%.3f int=%.3f "
         "mismatches=%d bad_softmax=%d\n",
         loss, before, after, integer_accuracy, mismatches, bad_softmax);

  expect_true("training reduces loss", loss < 1.0f);
  expect_true("float test accuracy", after > 0.90f);
  expect_true("training improves accuracy", after > before);
  expect_true("integer accuracy matches float", integer_accuracy == after);
  expect_true("integer predictions match float", mismatches == 0);
  expect_true("fixed-point softmax normalizes", bad_softmax == 0);

  attention_samples_destroy(train);
  attention_samples_destroy(test);
  attention_model_destroy(model);

  test_multi_word_integer_path();
  return test_report();
}
