#ifndef ONLY_GATES_AI_MASKED_DIFFUSION_H
#define ONLY_GATES_AI_MASKED_DIFFUSION_H

#include "tiny_gpt.h"

#include <stdint.h>

/*
 * One denoising step for a masked-diffusion language model.
 *
 * The model is a bidirectional Tiny GPT. `tokens` holds `length` ids: the
 * prompt prefix `[0, prompt_length)` is fixed and the answer region
 * `[prompt_length, length)` starts as `mask_id`. The step runs one
 * bidirectional forward pass, scores the still-masked answer positions, and
 * fills the leftmost `ceil(remaining / steps_left)` of them. Filling left to
 * right keeps variable names consistent across the answer; the number per step
 * guarantees the sample is complete after `total_steps` calls.
 *
 * `step` is zero-based. `temperature_q16` uses Q16 (65536 = 1.0). The function
 * writes the updated token ids back into `tokens` and returns the number of
 * masks still remaining, or -1 on an argument/allocation error. A return value
 * of 0 means the answer region is fully unmasked.
 */
int tiny_masked_diffusion_step(TinyGpt *model, int32_t *tokens, int length,
                               int prompt_length, int mask_id,
                               int32_t temperature_q16, int step,
                               int total_steps);

#endif
