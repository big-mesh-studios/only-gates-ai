/*
 * Standalone harness for the shipping integer GPT engine.
 *
 * This compiles the real src/tiny_gpt_infer.c and src/tiny_gpt_quantize.c; it
 * is not a second implementation. The harness itself may use stdio, malloc and
 * doubles for argument parsing and display. The model arithmetic stays integer.
 */

#include "tiny_gpt.h"
#include "tiny_gpt_quantize.h"
#include "int_math.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SELFTEST_VOCAB 512
#define SELFTEST_CONTEXT 16
#define SELFTEST_DIM 48
#define SELFTEST_LAYERS 2
#define SELFTEST_HEADS 2
#define SELFTEST_MLP 96

typedef struct {
  int vocab;
  int context;
  int dim;
  int layers;
  int heads;
  int mlp;
} Config;

typedef struct {
  const char *weights_path;
  Config config;
  int have_config;
  int32_t *tokens;
  int token_count;
  int steps;
  double temperature;
  int top_k;
  uint32_t seed;
  const char *dump_buffer;
  const char *dump_vectors;
} Options;

static void usage(const char *program) {
  fprintf(stderr,
          "usage: %s --selftest\n"
          "       %s --weights FILE --config V,T,D,L,H,M [options]\n"
          "\n"
          "options:\n"
          "  --tokens 1,2,3       token ids; the last T ids are used\n"
          "  --steps N            generate N tokens (default 0)\n"
          "  --temperature F      sampling temperature (default 0.8)\n"
          "  --top-k N            keep the N best tokens (default 40)\n"
          "  --seed N             xorshift32 seed (default 12345)\n"
          "  --dump-buffer NAME   x, x1, h, qkv, attn, y, z, gz, m, logits\n"
          "  --dump-vectors FILE  write an FPGA conformance vector (see\n"
          "                       docs/FPGA-CONFORMANCE.md)\n",
          program, program);
}

static size_t parameter_count(Config c) {
  const size_t v = (size_t)c.vocab;
  const size_t t = (size_t)c.context;
  const size_t d = (size_t)c.dim;
  const size_t l = (size_t)c.layers;
  const size_t m = (size_t)c.mlp;
  return v * d + t * d + d + l * (2 * d + 4 * d * d + 2 * d * m);
}

static int parse_config(const char *text, Config *out) {
  Config c;
  if (sscanf(text, "%d,%d,%d,%d,%d,%d", &c.vocab, &c.context, &c.dim,
             &c.layers, &c.heads, &c.mlp) != 6) {
    return 0;
  }
  if (c.vocab <= 0 || c.context <= 0 || c.dim <= 0 || c.layers <= 0 ||
      c.heads <= 0 || c.mlp <= 0 || c.dim % c.heads != 0) {
    return 0;
  }
  *out = c;
  return 1;
}

static int32_t *parse_tokens(const char *text, int *count_out) {
  size_t capacity = strlen(text) + 1;
  int32_t *ids = malloc(capacity * sizeof(*ids));
  if (ids == NULL) {
    return NULL;
  }
  int count = 0;
  const char *cursor = text;
  while (*cursor != '\0') {
    char *end = NULL;
    long value = strtol(cursor, &end, 10);
    if (end == cursor) {
      free(ids);
      return NULL;
    }
    if (value < 0 || value > 2147483647L) {
      free(ids);
      return NULL;
    }
    ids[count++] = (int32_t)value;
    cursor = end;
    while (*cursor == ',' || *cursor == ' ') {
      ++cursor;
    }
  }
  *count_out = count;
  return ids;
}

static float *read_weights(const char *path, size_t *count_out) {
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "could not open %s\n", path);
    return NULL;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    return NULL;
  }
  long size = ftell(file);
  if (size <= 0 || size % 4 != 0) {
    fprintf(stderr, "%s is not a float32 buffer\n", path);
    fclose(file);
    return NULL;
  }
  rewind(file);

  const size_t count = (size_t)size / 4;
  uint8_t *bytes = malloc((size_t)size);
  float *weights = malloc(count * sizeof(*weights));
  if (bytes == NULL || weights == NULL) {
    free(bytes);
    free(weights);
    fclose(file);
    return NULL;
  }
  if (fread(bytes, 1, (size_t)size, file) != (size_t)size) {
    fprintf(stderr, "short read from %s\n", path);
    free(bytes);
    free(weights);
    fclose(file);
    return NULL;
  }
  fclose(file);

  /* The checkpoint is little-endian float32. Decode explicitly so the harness
     also works on a big-endian host. */
  for (size_t i = 0; i < count; ++i) {
    const uint8_t *p = bytes + i * 4;
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    memcpy(&weights[i], &bits, sizeof(bits));
  }
  free(bytes);
  *count_out = count;
  return weights;
}

static float *synthetic_weights(size_t count) {
  float *weights = malloc(count * sizeof(*weights));
  if (weights == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < count; ++i) {
    weights[i] = (float)((int)(i % 17) - 8) * 0.01f;
  }
  return weights;
}

static int32_t temperature_q16(double temperature) {
  if (temperature < 0.01) {
    temperature = 0.01;
  }
  return (int32_t)(temperature * 65536.0 + 0.5);
}

static void fill_window(const int32_t *ids, int count, int context,
                        int32_t *window) {
  int start = count > context ? count - context : 0;
  int used = count - start;
  for (int i = 0; i < context - used; ++i) {
    window[i] = 0;
  }
  for (int i = 0; i < used; ++i) {
    window[context - used + i] = ids[start + i];
  }
}

static void print_top_logits(const int32_t *logits, int vocab, int k) {
  if (k > 16) {
    k = 16;
  }
  if (k > vocab) {
    k = vocab;
  }
  int32_t values[16];
  int ids[16];
  int count = 0;
  for (int id = 0; id < vocab; ++id) {
    const int32_t value = logits[id];
    if (count < k) {
      int pos = count;
      while (pos > 0 && values[pos - 1] < value) {
        values[pos] = values[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
      }
      values[pos] = value;
      ids[pos] = id;
      ++count;
    } else if (k > 0 && value > values[k - 1]) {
      int pos = k - 1;
      while (pos > 0 && values[pos - 1] < value) {
        values[pos] = values[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
      }
      values[pos] = value;
      ids[pos] = id;
    }
  }
  for (int i = 0; i < count; ++i) {
    printf("  token %6d  logit %10d  real %10.5f\n", ids[i], values[i],
           (double)values[i] / 65536.0);
  }
}

static void print_top_probabilities(const int32_t *probs, int vocab, int k) {
  if (k > 16) {
    k = 16;
  }
  if (k > vocab) {
    k = vocab;
  }
  int32_t values[16];
  int ids[16];
  int count = 0;
  for (int id = 0; id < vocab; ++id) {
    const int32_t value = probs[id];
    if (count < k) {
      int pos = count;
      while (pos > 0 && values[pos - 1] < value) {
        values[pos] = values[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
      }
      values[pos] = value;
      ids[pos] = id;
      ++count;
    } else if (k > 0 && value > values[k - 1]) {
      int pos = k - 1;
      while (pos > 0 && values[pos - 1] < value) {
        values[pos] = values[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
      }
      values[pos] = value;
      ids[pos] = id;
    }
  }
  for (int i = 0; i < count; ++i) {
    printf("  token %6d  p %8.5f  q16 %6d\n", ids[i],
           (double)values[i] / 65536.0, values[i]);
  }
}

static const int32_t *named_buffer(TinyGpt *model, const char *name) {
  if (strcmp(name, "x") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_X);
  if (strcmp(name, "x1") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_X1);
  if (strcmp(name, "h") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_H);
  if (strcmp(name, "qkv") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_QKV);
  if (strcmp(name, "attn") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_ATTN);
  if (strcmp(name, "y") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_Y);
  if (strcmp(name, "z") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_Z);
  if (strcmp(name, "gz") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_GZ);
  if (strcmp(name, "m") == 0) return tiny_gpt_buffer(model, TINY_GPT_BUFFER_M);
  if (strcmp(name, "logits") == 0)
    return tiny_gpt_buffer(model, TINY_GPT_BUFFER_LOGITS);
  return NULL;
}

static void dump_buffer(TinyGpt *model, const char *name, int count) {
  const int32_t *buffer = named_buffer(model, name);
  if (buffer == NULL) {
    fprintf(stderr, "unknown buffer: %s\n", name);
    return;
  }
  const int limit = count < 32 ? count : 32;
  printf("buffer %s, first %d values (Q16):\n", name, limit);
  for (int i = 0; i < limit; ++i) {
    printf("  [%2d] %10d  %10.5f\n", i, buffer[i],
           (double)buffer[i] / 65536.0);
  }
}

/* FNV-1a over the packed float32 checkpoint, used to tie a vector file to a
 * specific weights.bin. */
static uint64_t hash_weights(const float *weights, size_t count) {
  const uint8_t *bytes = (const uint8_t *)weights;
  const size_t length = count * sizeof(float);
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

/*
 * Writes a deterministic conformance vector for an RTL testbench: the config,
 * the weights hash, the absolute token window, and the Q16 logits at one
 * position. The format is deliberately plain text so a testbench can parse it
 * without a JSON dependency.
 */
static int write_vectors(const char *path, const Config *config,
                         const float *weights, size_t weight_count,
                         const int32_t *tokens, int position,
                         const int32_t *logits) {
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    fprintf(stderr, "could not write %s\n", path);
    return 0;
  }
  fprintf(file, "# only-gates-ai conformance vectors v1\n");
  fprintf(file, "config %d %d %d %d %d %d\n", config->vocab, config->context,
          config->dim, config->layers, config->heads, config->mlp);
  fprintf(file, "weights_fnv1a64 %016llx\n",
          (unsigned long long)hash_weights(weights, weight_count));
  fprintf(file, "tokens %d", position + 1);
  for (int i = 0; i <= position; ++i) {
    fprintf(file, " %d", tokens[i]);
  }
  fprintf(file, "\nposition %d\n", position);
  fprintf(file, "logits %d", config->vocab);
  for (int i = 0; i < config->vocab; ++i) {
    fprintf(file, " %d", logits[i]);
  }
  fprintf(file, "\n");
  fclose(file);
  return 1;
}

static int selftest_check(const char *name, int condition, int *failures) {
  if (!condition) {
    printf("FAIL %s\n", name);
    ++*failures;
  }
  return condition;
}

static int run_selftest(void) {
  int failures = 0;
  selftest_check("isqrt(144) == 12", int_isqrt_u64(144) == 12, &failures);
  selftest_check("isqrt(145) == 12", int_isqrt_u64(145) == 12, &failures);
  selftest_check("div_round(-7, 2) == -4", int_div_round(-7, 2) == -4,
                 &failures);
  selftest_check("shift_round(-6, 2) == -1", int_shift_round(-6, 2) == -1,
                 &failures);

  const int32_t exp_1 = int_exp_neg_q15_q12(4096);
  selftest_check("exp(-1) is near 12051 (Q15)", exp_1 > 11950 && exp_1 < 12150,
                 &failures);
  selftest_check("exp(0) == Q15 one", int_exp_neg_q15_q12(0) == INT_Q15_ONE,
                 &failures);
  selftest_check("exp underflows after 20 nats",
                 int_exp_neg_q15_q12(20 * 4096) <= 1, &failures);

  const Config config = {SELFTEST_VOCAB, SELFTEST_CONTEXT, SELFTEST_DIM,
                         SELFTEST_LAYERS, SELFTEST_HEADS, SELFTEST_MLP};
  const size_t count = parameter_count(config);
  float *weights = synthetic_weights(count);
  selftest_check("allocate synthetic weights", weights != NULL, &failures);
  if (weights == NULL) {
    return 1;
  }

  TinyGpt *model = tiny_gpt_create(config.vocab, config.context, config.dim,
                                   config.layers, config.heads, config.mlp,
                                   weights);
  selftest_check("create model", model != NULL, &failures);
  if (model == NULL) {
    free(weights);
    return 1;
  }
  selftest_check("create reports OK",
                 tiny_gpt_last_error() == TINY_GPT_OK, &failures);
  selftest_check("vocab size", tiny_gpt_vocab_size(model) == config.vocab,
                 &failures);
  selftest_check("context", tiny_gpt_context(model) == config.context,
                 &failures);

  int32_t tokens[SELFTEST_CONTEXT];
  for (int i = 0; i < SELFTEST_CONTEXT; ++i) {
    tokens[i] = i;
  }
  tiny_gpt_forward(model, tokens);
  selftest_check("forward reports OK", tiny_gpt_last_error() == TINY_GPT_OK,
                 &failures);

  const int32_t *logits = tiny_gpt_logits(model);
  selftest_check("logits are available", logits != NULL, &failures);
  if (logits != NULL) {
    const int total = config.context * config.vocab;
    int32_t minimum = logits[0];
    int32_t maximum = logits[0];
    int nonzero = 0;
    for (int i = 0; i < total; ++i) {
      if (logits[i] < minimum) minimum = logits[i];
      if (logits[i] > maximum) maximum = logits[i];
      if (logits[i] != 0) ++nonzero;
    }
    selftest_check("logits are non-degenerate", maximum > minimum, &failures);
    selftest_check("logits are mostly nonzero", nonzero > total / 2, &failures);
    selftest_check("logits fit comfortably in int32",
                   maximum < 1000000 && minimum > -1000000, &failures);
  }

  int32_t probabilities[SELFTEST_VOCAB];
  tiny_gpt_probabilities(model, INT_Q16_ONE, probabilities);
  selftest_check("probabilities report OK",
                 tiny_gpt_last_error() == TINY_GPT_OK, &failures);
  int64_t sum = 0;
  int32_t peak = 0;
  int peak_index = -1;
  for (int i = 0; i < SELFTEST_VOCAB; ++i) {
    if (probabilities[i] < 0) {
      selftest_check("probability is non-negative", 0, &failures);
    }
    sum += probabilities[i];
    if (probabilities[i] > peak) {
      peak = probabilities[i];
      peak_index = i;
    }
  }
  selftest_check("probabilities sum to Q16 one",
                 sum > 65536 - SELFTEST_VOCAB && sum < 65536 + SELFTEST_VOCAB,
                 &failures);
  selftest_check("argmax is in range", peak_index >= 0, &failures);

  int32_t cold[SELFTEST_VOCAB];
  int32_t hot[SELFTEST_VOCAB];
  tiny_gpt_probabilities(model, INT_Q16_ONE / 2, cold);
  tiny_gpt_probabilities(model, INT_Q16_ONE * 2, hot);
  if (peak_index >= 0) {
    selftest_check("lower temperature concentrates mass",
                   cold[peak_index] >= hot[peak_index], &failures);
  }

  tiny_gpt_set_seed(model, 12345);
  const int32_t first = tiny_gpt_sample(model, temperature_q16(0.8), 40);
  tiny_gpt_set_seed(model, 12345);
  const int32_t second = tiny_gpt_sample(model, temperature_q16(0.8), 40);
  selftest_check("sampler is reproducible", first == second, &failures);
  selftest_check("sample is in range",
                 first >= 0 && first < SELFTEST_VOCAB, &failures);

  int32_t *first_logits = malloc((size_t)config.context * config.vocab *
                                 sizeof(*first_logits));
  selftest_check("allocate logits copy", first_logits != NULL, &failures);
  if (first_logits != NULL && logits != NULL) {
    memcpy(first_logits, logits,
           (size_t)config.context * config.vocab * sizeof(*first_logits));
    tiny_gpt_forward(model, tokens);
    selftest_check("same tokens produce identical logits",
                   memcmp(first_logits, tiny_gpt_logits(model),
                          (size_t)config.context * config.vocab *
                              sizeof(*first_logits)) == 0,
                   &failures);
    free(first_logits);
  }

  selftest_check("qkv buffer is wired",
                 tiny_gpt_buffer(model, TINY_GPT_BUFFER_QKV) != NULL,
                 &failures);
  selftest_check("unknown buffer is rejected",
                 tiny_gpt_buffer(model, 12345) == NULL, &failures);

  tiny_gpt_destroy(model);
  free(weights);

  if (failures == 0) {
    printf("all selftests passed\n");
    return 0;
  }
  printf("%d selftest(s) failed\n", failures);
  return 1;
}

static int parse_options(int argc, char **argv, Options *options) {
  options->temperature = 0.8;
  options->top_k = 40;
  options->seed = 12345;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--selftest") == 0) {
      return 2;
    } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
      options->weights_path = argv[++i];
    } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      if (!parse_config(argv[++i], &options->config)) {
        fprintf(stderr, "invalid --config; expected V,T,D,L,H,M\n");
        return 0;
      }
      options->have_config = 1;
    } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
      options->tokens = parse_tokens(argv[++i], &options->token_count);
      if (options->tokens == NULL) {
        fprintf(stderr, "invalid --tokens; expected 1,2,3\n");
        return 0;
      }
    } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
      options->steps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      options->temperature = atof(argv[++i]);
    } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      options->top_k = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      options->seed = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--dump-buffer") == 0 && i + 1 < argc) {
      options->dump_buffer = argv[++i];
    } else if (strcmp(argv[i], "--dump-vectors") == 0 && i + 1 < argc) {
      options->dump_vectors = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0 ||
               strcmp(argv[i], "-h") == 0) {
      return 0;
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 0;
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  Options options;
  memset(&options, 0, sizeof(options));
  const int parsed = parse_options(argc, argv, &options);
  if (parsed == 2) {
    return run_selftest();
  }
  if (parsed == 0) {
    usage(argv[0]);
    return 1;
  }
  if (options.weights_path == NULL || !options.have_config) {
    usage(argv[0]);
    return 1;
  }

  size_t weight_count = 0;
  float *weights = read_weights(options.weights_path, &weight_count);
  if (weights == NULL) {
    return 1;
  }
  const size_t expected = parameter_count(options.config);
  if (weight_count != expected) {
    fprintf(stderr,
            "weight count mismatch: file has %zu values, config needs %zu\n",
            weight_count, expected);
    free(weights);
    return 1;
  }

  TinyGpt *model =
      tiny_gpt_create(options.config.vocab, options.config.context,
                      options.config.dim, options.config.layers,
                      options.config.heads, options.config.mlp, weights);
  if (model == NULL) {
    fprintf(stderr, "tiny_gpt_create failed with error %d\n",
            tiny_gpt_last_error());
    free(weights);
    return 1;
  }

  int32_t *window = malloc((size_t)options.config.context * sizeof(*window));
  int32_t *probabilities =
      malloc((size_t)options.config.vocab * sizeof(*probabilities));
  if (window == NULL || probabilities == NULL) {
    fprintf(stderr, "allocation failed\n");
    free(options.tokens);
    free(window);
    free(probabilities);
    tiny_gpt_destroy(model);
    free(weights);
    return 1;
  }

  if (options.tokens == NULL) {
    options.tokens = calloc(1, sizeof(*options.tokens));
    options.token_count = options.tokens == NULL ? 0 : 1;
  }
  fill_window(options.tokens, options.token_count, options.config.context,
              window);

  printf("config: vocab=%d context=%d dim=%d layers=%d heads=%d mlp=%d\n",
         options.config.vocab, options.config.context, options.config.dim,
         options.config.layers, options.config.heads, options.config.mlp);
  printf("weights: %s (%zu float32 values)\n", options.weights_path,
         weight_count);
  printf("window: ");
  if (options.config.context <= 32) {
    for (int i = 0; i < options.config.context; ++i) {
      printf("%d%s", window[i],
             i + 1 == options.config.context ? "\n" : " ");
    }
  } else {
    printf("%d ids, last 8:", options.config.context);
    for (int i = options.config.context - 8; i < options.config.context; ++i) {
      printf(" %d", window[i]);
    }
    printf("\n");
  }

  tiny_gpt_forward(model, window);
  if (tiny_gpt_last_error() != TINY_GPT_OK) {
    fprintf(stderr, "forward failed with error %d\n", tiny_gpt_last_error());
    free(options.tokens);
    free(window);
    free(probabilities);
    tiny_gpt_destroy(model);
    free(weights);
    return 1;
  }

  if (options.dump_buffer != NULL) {
    dump_buffer(model, options.dump_buffer,
                options.config.context * options.config.vocab);
  }

  const int32_t *logits = tiny_gpt_logits(model);
  const int last = (options.config.context - 1) * options.config.vocab;
  if (options.dump_vectors != NULL) {
    if (!write_vectors(options.dump_vectors, &options.config, weights,
                       weight_count, window, options.config.context - 1,
                       logits + last)) {
      free(options.tokens);
      free(window);
      free(probabilities);
      tiny_gpt_destroy(model);
      free(weights);
      return 1;
    }
    printf("wrote conformance vectors: %s\n", options.dump_vectors);
  }
  printf("next-token logits (Q16, last position):\n");
  print_top_logits(logits + last, options.config.vocab, 10);

  tiny_gpt_probabilities(model, temperature_q16(options.temperature),
                         probabilities);
  printf("probabilities at temperature %.3f (Q16, sum should be ~65536):\n",
         options.temperature);
  print_top_probabilities(probabilities, options.config.vocab, 10);

  if (options.steps > 0) {
    const int capacity = options.token_count + options.steps;
    int32_t *ids = malloc((size_t)capacity * sizeof(*ids));
    if (ids == NULL) {
      fprintf(stderr, "allocation failed\n");
      free(options.tokens);
      free(window);
      free(probabilities);
      tiny_gpt_destroy(model);
      free(weights);
      return 1;
    }
    memcpy(ids, options.tokens, (size_t)options.token_count * sizeof(*ids));
    int count = options.token_count;

    tiny_gpt_set_seed(model, options.seed);
    printf("generation (temperature %.3f, top-k %d, seed %u):\n",
           options.temperature, options.top_k, options.seed);
    for (int step = 0; step < options.steps; ++step) {
      fill_window(ids, count, options.config.context, window);
      tiny_gpt_forward(model, window);
      if (tiny_gpt_last_error() != TINY_GPT_OK) {
        fprintf(stderr, "forward failed at step %d\n", step);
        break;
      }
      const int32_t token =
          tiny_gpt_sample(model, temperature_q16(options.temperature),
                          options.top_k);
      printf("  %3d  token %d\n", step + 1, token);
      ids[count++] = token;
    }

    printf("ids:");
    for (int i = 0; i < count; ++i) {
      printf(" %d", ids[i]);
    }
    printf("\n");
    free(ids);
  }

  free(options.tokens);
  free(window);
  free(probabilities);
  tiny_gpt_destroy(model);
  free(weights);
  return 0;
}
