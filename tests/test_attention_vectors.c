#include "test_common.h"

#include <stdlib.h>
#include <string.h>

#include "../src/attention.c"
#include "../src/attention_vectors.c"

static void test_tissier_loader(void) {
  const char *path = "test_attention_vectors_tissier.tmp";
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    expect_true("write tissier fixture", 0);
    return;
  }
  fprintf(file, "3 64\n");
  fprintf(file, "all_ones %llu\n", (unsigned long long)UINT64_MAX);
  fprintf(file, "all_zeros 0\n");
  fprintf(file, "sparse 12345\n");
  fclose(file);

  BinaryVectors *vectors = binary_vectors_load_tissier(path, 0);
  remove(path);
  if (vectors == NULL) {
    expect_true("load tissier fixture", 0);
    return;
  }

  expect_true("tissier count", vectors->count == 3);
  expect_true("tissier bits", vectors->bits == 64);
  expect_true("tissier word 0", strcmp(vectors->words[0], "all_ones") == 0);
  expect_true("tissier word 2", strcmp(vectors->words[2], "sparse") == 0);
  for (int bit = 0; bit < 64; ++bit) {
    expect_true("tissier all ones", vectors->vectors[bit] == 1);
    expect_true("tissier all zeros", vectors->vectors[64 + bit] == 0);
  }
  const unsigned long long sparse = 12345;
  for (int bit = 0; bit < 64; ++bit) {
    const uint8_t expected = (uint8_t)((sparse >> bit) & 1u);
    expect_true("tissier sparse", vectors->vectors[128 + bit] == expected);
  }

  binary_vectors_destroy(vectors);
}

static void test_text_loader(void) {
  const char *path = "test_attention_vectors_text.tmp";
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    expect_true("write text fixture", 0);
    return;
  }
  fprintf(file, "3 4\n");
  fprintf(file, "a 0.3 -0.3 0.3 -0.3\n");
  fprintf(file, "b -0.1 0.2 -0.3 0.4\n");
  fprintf(file, "c 0 0 0 0\n");
  fclose(file);

  BinaryVectors *vectors = binary_vectors_load_text(path, 2, 0);
  remove(path);
  if (vectors == NULL) {
    expect_true("load text fixture", 0);
    return;
  }

  expect_true("text count", vectors->count == 3);
  expect_true("text truncated bits", vectors->bits == 2);
  expect_true("text rows are sign bits",
              vectors->vectors[0] == 1 && vectors->vectors[1] == 0);
  expect_true("text second row",
              vectors->vectors[2] == 0 && vectors->vectors[3] == 1);
  expect_true("text zero row",
              vectors->vectors[4] == 0 && vectors->vectors[5] == 0);

  binary_vectors_destroy(vectors);
}

static void test_keys_from_vectors(void) {
  const char *path = "test_attention_vectors_model.tmp";
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    expect_true("write model fixture", 0);
    return;
  }
  fprintf(file, "3 32\n");
  fprintf(file, "alpha 2863311530\n"); /* alternating bits */
  fprintf(file, "beta 1431655765\n");
  fprintf(file, "gamma 0\n");
  fclose(file);

  BinaryVectors *vectors = binary_vectors_load_tissier(path, 0);
  remove(path);
  if (vectors == NULL || vectors->bits != 32) {
    expect_true("load model fixture", 0);
    binary_vectors_destroy(vectors);
    return;
  }

  AttentionModel *model =
      attention_model_create(vectors->bits, vectors->count, 0x1234u);
  if (model == NULL) {
    expect_true("create model from vectors", 0);
    binary_vectors_destroy(vectors);
    return;
  }
  for (int key = 0; key < vectors->count; ++key) {
    attention_model_set_key(model, key,
                            vectors->vectors + (size_t)key * vectors->bits);
  }
  attention_quantize(model);

  /* Query is the first key with a couple of bits flipped; identity
   * projections should still select it. */
  uint8_t query[32];
  memcpy(query, vectors->vectors, sizeof(query));
  query[0] ^= 1;
  query[7] ^= 1;

  const int float_prediction = attention_predict_float(model, query);
  const int int_prediction = attention_predict_int(model, query);
  expect_true("identity attention selects key 0", float_prediction == 0);
  expect_true("integer path agrees on loaded keys",
              int_prediction == float_prediction);

  attention_model_destroy(model);
  binary_vectors_destroy(vectors);
}

int main(void) {
  test_tissier_loader();
  test_text_loader();
  test_keys_from_vectors();
  return test_report();
}
