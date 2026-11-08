#ifndef ONLY_GATES_AI_INT_MATH_H
#define ONLY_GATES_AI_INT_MATH_H

#include <stdint.h>

/*
 * Integer-only fixed-point helpers shared by the attention test bed and the
 * tiny GPT inference path.
 *
 * The inference code keeps the activation stream in Q16 (value * 65536) and
 * probabilities in Q15 (value * 32768). Products that would overflow 32 bits
 * are accumulated in int64. No floating point is used here.
 */

#define INT_Q16_ONE 65536
#define INT_Q15_ONE 32768
#define INT_Q12_ONE 4096

static inline int32_t int_clamp_i32(int32_t value, int32_t low, int32_t high) {
  if (value < low) {
    return low;
  }
  if (value > high) {
    return high;
  }
  return value;
}

/* Rounds a / b to nearest, with b > 0. */
static inline int64_t int_div_round(int64_t a, int64_t b) {
  if (b <= 0) {
    return 0;
  }
  if (a >= 0) {
    return (a + b / 2) / b;
  }
  return -((-a + b / 2) / b);
}

/* Rounds a / 2^shift to nearest. */
static inline int64_t int_shift_round(int64_t a, int shift) {
  if (shift <= 0) {
    return a;
  }
  return (a + ((int64_t)1 << (shift - 1))) >> shift;
}

/* Floor of sqrt(n), integer-only. */
static inline uint64_t int_isqrt_u64(uint64_t n) {
  uint64_t result = 0;
  uint64_t bit = (uint64_t)1 << 62;
  while (bit > n) {
    bit >>= 2;
  }
  while (bit != 0) {
    if (n >= result + bit) {
      n -= result + bit;
      result = (result >> 1) + bit;
    } else {
      result >>= 1;
    }
    bit >>= 2;
  }
  return result;
}

/* round(32768 * 2^(-k/32)), k = 0..32. */
static const int32_t INT_TWO_POW_NEG_Q15[33] = {
    32768, 32066, 31379, 30706, 30048, 29405, 28774, 28158, 27554, 26964,
    26386, 25821, 25268, 24726, 24196, 23678, 23170, 22674, 22188, 21713,
    21247, 20792, 20347, 19911, 19484, 19066, 18658, 18258, 17867, 17484,
    17109, 16743, 16384,
};

/*
 * exp(-x) in Q15 for x >= 0 in Q12 (1/4096 nat resolution).
 *
 * The exponent is clamped to 20 nats by the callers (the paper's DI-Exp
 * clipping range). Values beyond the Q15 resolution return 0.
 */
static inline int32_t int_exp_neg_q15_q12(int32_t x_q12) {
  if (x_q12 <= 0) {
    return INT_Q15_ONE;
  }
  const int32_t y_q12 = (int32_t)(((int64_t)x_q12 * 5909) >> 12);
  const int32_t integer_part = y_q12 >> 12;
  if (integer_part >= 15) {
    return 0;
  }
  const int32_t fraction = y_q12 & 4095;
  const int32_t index = fraction >> 7;
  const int32_t remainder = fraction & 127;
  const int32_t a = INT_TWO_POW_NEG_Q15[index];
  const int32_t b = INT_TWO_POW_NEG_Q15[index + 1];
  const int32_t value = a - (((a - b) * remainder) >> 7);
  return value >> integer_part;
}

#endif
