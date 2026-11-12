#ifndef ONLY_GATES_AI_TINY_GPT_DOT_SIMD_H
#define ONLY_GATES_AI_TINY_GPT_DOT_SIMD_H

#include <stdint.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

/*
 * int8 dot product used by the integer matmul and attention scores.
 *
 * The scalar path is the reference. The wasm SIMD path widens i8x16 to i16x8
 * and uses i32x4.dot_i16x8_s, so every product is exact and every partial sum
 * is exact in int32. Integer addition is associative, which is why the SIMD
 * lane reproduces the scalar logits bit-for-bit even though the accumulation
 * is grouped differently.
 *
 * The largest possible cell is 127 * 127 * 512 = 8,258,048, well inside int32.
 */
static inline int32_t tiny_gpt_dot_i8(const int8_t *a, const int8_t *b,
                                      int count) {
#if defined(__wasm_simd128__)
  v128_t accumulator = wasm_i32x4_const_splat(0);
  int i = 0;
  for (; i + 16 <= count; i += 16) {
    const v128_t va = wasm_v128_load((const void *)(a + i));
    const v128_t vb = wasm_v128_load((const void *)(b + i));
    const v128_t a_low = wasm_i16x8_extend_low_i8x16(va);
    const v128_t a_high = wasm_i16x8_extend_high_i8x16(va);
    const v128_t b_low = wasm_i16x8_extend_low_i8x16(vb);
    const v128_t b_high = wasm_i16x8_extend_high_i8x16(vb);
    accumulator =
        wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(a_low, b_low));
    accumulator =
        wasm_i32x4_add(accumulator, wasm_i32x4_dot_i16x8(a_high, b_high));
  }
  int32_t sum = wasm_i32x4_extract_lane(accumulator, 0) +
                wasm_i32x4_extract_lane(accumulator, 1) +
                wasm_i32x4_extract_lane(accumulator, 2) +
                wasm_i32x4_extract_lane(accumulator, 3);
#else
  int32_t sum = 0;
  int i = 0;
#endif
  for (; i < count; ++i) {
    sum += (int32_t)a[i] * (int32_t)b[i];
  }
  return sum;
}

#endif
