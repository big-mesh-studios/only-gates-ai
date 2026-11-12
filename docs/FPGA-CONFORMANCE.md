# Integer GPT conformance vectors

An FPGA bring-up needs a machine-checkable answer to "does my bitstream compute
the same thing as the shipping engine?" The browser/Node engine is the golden
model; `examples/integer-gpt` can export a vector file that an RTL testbench
compares against.

This is the same guarantee the C and WebAssembly tests enforce: the reference
lane, the KV-cache decode lane, and the WASM SIMD lane must produce **identical
Q16 logits**, not merely similar probabilities.

## Producing a vector

From the repository root, build the example with any C compiler (or Emscripten
for a Node-runnable build):

```sh
cc -O2 -Isrc examples/integer-gpt/main.c src/tiny_gpt_infer.c \
   src/tiny_gpt_quantize.c -o integer-gpt

./integer-gpt \
  --weights apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin \
  --config 4096,256,128,4,4,512 \
  --tokens 1,42,17,... \
  --dump-vectors solid2-vectors.txt
```

`--tokens` takes token ids; the last `context` ids form the window. Use the
BPE tokenizer to obtain ids for a real prompt, or a fixed synthetic window for a
first smoke test.

## Format

Plain text, one record per line, `#` comments ignored:

```
# only-gates-ai conformance vectors v1
config 4096 256 128 4 4 512
weights_fnv1a64 8f2a...
tokens 17 1 42 17 9 ...
position 16
logits 4096 -1234 5678 ...
```

- `config` — `vocab context dim layers heads mlp`.
- `weights_fnv1a64` — FNV-1a 64-bit hash over the packed float32 checkpoint
  bytes as loaded by the harness. It pins the vector to a specific
  `weights.bin` and catches a silent checkpoint swap.
- `tokens` — count followed by the absolute token ids `0..position`.
- `position` — absolute context position the logits belong to.
- `logits` — count followed by Q16 `int32` logits, index = token id.

The field order and the absence of floating-point values are deliberate: a
testbench can parse this with a few `read` calls, and there is no ambiguity
about formatting or rounding.

## How a testbench uses it

1. Load the same weights (or their int8/Q16 quantization) into the design.
2. Apply the token ids in order, one position per cycle group as the schedule
   requires.
3. At `position`, read the Q16 logits and compare **exactly** to the file.
4. On a mismatch, report the first differing token id, both values, and the
   delta. The first divergence identifies the stage: embeddings, one layer, the
   head.

The C harness also supports `--dump-buffer x|h|qkv|attn|...` for
intermediate Q16 buffers, which is how you bisect a mismatch stage by stage
rather than guessing at the final logits.

## What is allowed to differ

- Host-side tokenization and sampling are out of scope; only the forward pass
  is compared.
- A design that replaces dynamic per-row activation quantization with static
  scales will **not** match these vectors. That is expected: generate a second
  set of vectors from a build with the same static scales and keep the two lanes
  separate. Do not relabel one as the other.
- Cycle count, latency, and parallelism are free to differ. Only the integer
  values are fixed.
