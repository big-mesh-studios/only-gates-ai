#include "masked_diffusion.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Integer-only masked-diffusion sampler.
 *
 * The transformer forward pass lives in tiny_gpt_infer.c. This file keeps the
 * unmasking policy next to the model so the hot path never crosses into JS for
 * per-position softmax or sorting. It uses only public TinyGpt entry points.
 *
 * Unmasking is left to right. The denoiser is trained on random mask patterns,
 * so any order is valid, but filling a contiguous prefix keeps variable names
 * consistent: a later `print(i)` sees the `i` chosen by the earlier `for`.
 * Confidence-ordered unmasking looked more "diffusion-like" but regularly
 * mixed names like `for offset ... scores[position]`, which reads as a bug.
 */

/* Top-k candidates retained per position for sampling. */
#define MASKED_TOP_K 32

typedef struct {
  int32_t ids[MASKED_TOP_K];
  int32_t values[MASKED_TOP_K];
  int count;
  int64_t total;
} MaskCandidate;

static uint32_t masked_next_random(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

/*
 * Keeps the highest `MASKED_TOP_K` probabilities in descending order. The
 * mask and pad ids are never emitted: a denoiser that predicts its own mask
 * would not make progress, and pad is a layout token, not answer text.
 */
static void collect_candidate(const int32_t *probabilities, int vocab,
                              int mask_id, MaskCandidate *out) {
  out->count = 0;
  out->total = 0;

  for (int id = 0; id < vocab; ++id) {
    const int32_t value = probabilities[id];
    if (value <= 0 || id == mask_id || id == 0) {
      continue;
    }
    if (out->count < MASKED_TOP_K) {
      int slot = out->count;
      while (slot > 0 && out->values[slot - 1] < value) {
        out->values[slot] = out->values[slot - 1];
        out->ids[slot] = out->ids[slot - 1];
        --slot;
      }
      out->values[slot] = value;
      out->ids[slot] = id;
      ++out->count;
    } else if (value > out->values[MASKED_TOP_K - 1]) {
      int slot = MASKED_TOP_K - 1;
      while (slot > 0 && out->values[slot - 1] < value) {
        out->values[slot] = out->values[slot - 1];
        out->ids[slot] = out->ids[slot - 1];
        --slot;
      }
      out->values[slot] = value;
      out->ids[slot] = id;
    }
  }

  for (int i = 0; i < out->count; ++i) {
    out->total += out->values[i];
  }
}

static int32_t sample_candidate(const MaskCandidate *candidate,
                                uint32_t *rng) {
  if (candidate->count <= 0) {
    return 0;
  }
  if (candidate->total <= 0) {
    return candidate->ids[0];
  }
  const uint32_t random = masked_next_random(rng);
  const uint64_t target =
      ((uint64_t)random * (uint64_t)candidate->total) >> 32;
  uint64_t cumulative = 0;
  for (int i = 0; i < candidate->count; ++i) {
    cumulative += (uint64_t)candidate->values[i];
    if (target < cumulative) {
      return candidate->ids[i];
    }
  }
  return candidate->ids[candidate->count - 1];
}

int tiny_masked_diffusion_step(TinyGpt *model, int32_t *tokens, int length,
                               int prompt_length, int mask_id,
                               int32_t temperature_q16, int step,
                               int total_steps) {
  if (model == NULL || tokens == NULL || length <= 0 ||
      length > model->config.context || prompt_length < 0 ||
      prompt_length >= length) {
    return -1;
  }
  if (step < 0 || total_steps <= 0) {
    return -1;
  }

  int remaining = 0;
  for (int i = prompt_length; i < length; ++i) {
    if (tokens[i] == mask_id) {
      ++remaining;
    }
  }
  if (remaining == 0) {
    return 0;
  }

  tiny_gpt_forward_bidirectional(model, tokens, length);
  if (tiny_gpt_last_error() != TINY_GPT_OK) {
    return -1;
  }

  const int vocab = model->config.vocab_size;
  int32_t *probabilities = malloc((size_t)vocab * sizeof(int32_t));
  if (probabilities == NULL) {
    return -1;
  }

  int steps_left = total_steps - step;
  if (steps_left < 1) {
    steps_left = 1;
  }
  int to_unmask = (remaining + steps_left - 1) / steps_left;
  if (to_unmask < 1) {
    to_unmask = 1;
  } else if (to_unmask > remaining) {
    to_unmask = remaining;
  }

  MaskCandidate candidate;
  int unmasked = 0;
  for (int i = prompt_length; i < length && unmasked < to_unmask; ++i) {
    if (tokens[i] != mask_id) {
      continue;
    }
    tiny_gpt_probabilities_at(model, i, temperature_q16, probabilities);
    if (tiny_gpt_last_error() != TINY_GPT_OK) {
      free(probabilities);
      return -1;
    }
    collect_candidate(probabilities, vocab, mask_id, &candidate);
    const int32_t next = sample_candidate(&candidate, &model->rng);
    tokens[i] = next == mask_id ? 0 : next;
    ++unmasked;
  }

  free(probabilities);
  return remaining - unmasked;
}
