/** Small deterministic PRNG so seeded initialisation is reproducible. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^=
      value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fills one probability distribution with normalized random values. */
export function fillRandomDistribution(
  heap: Float32Array,
  offset: number,
  length: number,
  random: () => number,
): void {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const value = -Math.log(Math.max(random(), Number.MIN_VALUE));
    heap[offset + i] = value;
    sum += value;
  }
  if (sum > 0) {
    for (let i = 0; i < length; i++) {
      heap[offset + i] = heap[offset + i]! / sum;
    }
  } else {
    const uniform = 1 / length;
    for (let i = 0; i < length; i++) {
      heap[offset + i] = uniform;
    }
  }
}
