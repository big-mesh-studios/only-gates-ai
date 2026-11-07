#include "attention_vectors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VECTOR_LINE_MAX 8192

static char *duplicate_string(const char *text) {
  const size_t length = strlen(text) + 1;
  char *copy = malloc(length);
  if (copy != NULL) {
    memcpy(copy, text, length);
  }
  return copy;
}

static void free_vectors(BinaryVectors *vectors, int loaded) {
  if (vectors == NULL) {
    return;
  }
  if (vectors->words != NULL) {
    for (int i = 0; i < loaded; ++i) {
      free(vectors->words[i]);
    }
  }
  free(vectors->words);
  free(vectors->vectors);
  free(vectors);
}

static BinaryVectors *allocate_vectors(int count, int bits) {
  BinaryVectors *vectors = calloc(1, sizeof(*vectors));
  if (vectors == NULL) {
    return NULL;
  }
  vectors->count = count;
  vectors->bits = bits;
  vectors->words = calloc((size_t)count, sizeof(*vectors->words));
  vectors->vectors = calloc((size_t)count * bits, sizeof(*vectors->vectors));
  if (vectors->words == NULL || vectors->vectors == NULL) {
    free_vectors(vectors, 0);
    return NULL;
  }
  return vectors;
}

BinaryVectors *binary_vectors_load_tissier(const char *path, int max_count) {
  if (path == NULL) {
    return NULL;
  }
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    return NULL;
  }

  char line[VECTOR_LINE_MAX];
  if (fgets(line, sizeof(line), file) == NULL) {
    fclose(file);
    return NULL;
  }
  int count = 0;
  int bits = 0;
  if (sscanf(line, "%d %d", &count, &bits) != 2 || count <= 0 || bits <= 0) {
    fclose(file);
    return NULL;
  }
  int loaded = count;
  if (max_count > 0 && max_count < loaded) {
    loaded = max_count;
  }

  BinaryVectors *vectors = allocate_vectors(loaded, bits);
  if (vectors == NULL) {
    fclose(file);
    return NULL;
  }

  const int groups = (bits + 63) / 64;
  int row = 0;
  while (row < loaded && fgets(line, sizeof(line), file) != NULL) {
    char *token = strtok(line, " \t\r\n");
    if (token == NULL) {
      continue;
    }
    vectors->words[row] = duplicate_string(token);
    if (vectors->words[row] == NULL) {
      free_vectors(vectors, row + 1);
      fclose(file);
      return NULL;
    }
    for (int group = 0; group < groups; ++group) {
      token = strtok(NULL, " \t\r\n");
      if (token == NULL) {
        free_vectors(vectors, row + 1);
        fclose(file);
        return NULL;
      }
      const unsigned long long value = strtoull(token, NULL, 10);
      const int start = group * 64;
      for (int bit = 0; bit < 64 && start + bit < bits; ++bit) {
        vectors->vectors[(size_t)row * bits + start + bit] =
            (uint8_t)((value >> bit) & 1u);
      }
    }
    ++row;
  }
  fclose(file);

  if (row != loaded) {
    free_vectors(vectors, row);
    return NULL;
  }
  return vectors;
}

BinaryVectors *binary_vectors_load_text(const char *path, int max_bits,
                                        int max_count) {
  if (path == NULL) {
    return NULL;
  }
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    return NULL;
  }

  char line[VECTOR_LINE_MAX];
  if (fgets(line, sizeof(line), file) == NULL) {
    fclose(file);
    return NULL;
  }
  int count = 0;
  int dimension = 0;
  if (sscanf(line, "%d %d", &count, &dimension) != 2 || count <= 0 ||
      dimension <= 0) {
    fclose(file);
    return NULL;
  }
  int bits = dimension;
  if (max_bits > 0 && max_bits < bits) {
    bits = max_bits;
  }
  int loaded = count;
  if (max_count > 0 && max_count < loaded) {
    loaded = max_count;
  }

  BinaryVectors *vectors = allocate_vectors(loaded, bits);
  if (vectors == NULL) {
    fclose(file);
    return NULL;
  }

  int row = 0;
  while (row < loaded && fgets(line, sizeof(line), file) != NULL) {
    char *token = strtok(line, " \t\r\n");
    if (token == NULL) {
      continue;
    }
    vectors->words[row] = duplicate_string(token);
    if (vectors->words[row] == NULL) {
      free_vectors(vectors, row + 1);
      fclose(file);
      return NULL;
    }
    for (int bit = 0; bit < bits; ++bit) {
      token = strtok(NULL, " \t\r\n");
      if (token == NULL) {
        free_vectors(vectors, row + 1);
        fclose(file);
        return NULL;
      }
      vectors->vectors[(size_t)row * bits + bit] =
          strtof(token, NULL) > 0.0f ? 1 : 0;
    }
    ++row;
  }
  fclose(file);

  if (row != loaded) {
    free_vectors(vectors, row);
    return NULL;
  }
  return vectors;
}

void binary_vectors_destroy(BinaryVectors *vectors) {
  if (vectors == NULL) {
    return;
  }
  for (int i = 0; i < vectors->count; ++i) {
    free(vectors->words[i]);
  }
  free(vectors->words);
  free(vectors->vectors);
  free(vectors);
}
