# Integer GPT: runnable harness

A small command-line program around the **shipping** integer engine. The
Makefile compiles:

```text
../../src/tiny_gpt_infer.c      integer-only forward pass and sampler
../../src/tiny_gpt_quantize.c   float checkpoint -> int8/Q16 model
```

`main.c` is only a harness. It parses arguments, reads a checkpoint, calls the
real API, and prints results. It does not reimplement a kernel.

## Build and run the selftest

```sh
cd examples/integer-gpt
make check
```

`make check` does two things:

1. Searches `src/tiny_gpt_infer.c` for a float/double type or a `math.h`
   include and fails if one appears. Comments are stripped first.
2. Builds the harness and runs a deterministic selftest: integer math, model
   creation, a forward pass, non-degenerate logits, Q16 probability sum,
   temperature behavior, and sampler reproducibility.

Expected final line:

```text
all selftests passed
```

## Inspect a real checkpoint

The repository ships Solid 2 in
`apps/only-gates-ai/public/models/tiny-gpt-solid2/`. From this directory:

```sh
./integer-gpt \
  --weights ../../apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin \
  --config 4096,256,128,4,4,512 \
  --tokens 1,2,3,4,5,6,7,8 \
  --steps 8 --temperature 0.7 --top-k 40 --seed 12345
```

`--config` order is **vocab, context, dim, layers, heads, mlp**. The program
checks the file size against the parameter formula before loading, so a wrong
config fails immediately instead of producing garbage.

Output includes:

- the normalized window passed to the engine;
- the top 10 Q16 logits of the last position, with the real value;
- the top 10 Q16 probabilities at the chosen temperature;
- the sampled token for each generation step;
- the final id sequence.

### Token ids, not text

The engine has no tokenizer. `--tokens` takes integer ids. For a real prompt,
encode it first with the checkpoint's tokenizer:

- Alice uses a word-level vocabulary (`<unk>=0`, `<bos>=1`, `<eos>=2`).
- Solid 2 uses byte-level BPE, with `<|pad|>=0`, `<|user|>=1`,
  `<|assistant|>=2`, `<|end|>=3`, `<|facts|>=4` and a chat template of
  `<|user|>question<|assistant|>`.

`apps/only-gates-ai/src/lib/tinyGptBpe.ts` and `tinyGptSolid2.ts` show the host
side. The command above intentionally uses arbitrary ids to demonstrate the
engine boundary; it is not a meaningful prompt.

The window keeps the **last** `context` ids and right-pads with 0 when the
sequence is shorter. Training also right-pads, so causal attention means the
pads never affect the real tokens. Logits are read at the last real prompt
position rather than at the end of the padded window. The browser decode path
does not pad at all: it prefills the real prompt and then calls
`tiny_gpt_forward_range` one position at a time against the KV cache.

## Export FPGA conformance vectors

```sh
./integer-gpt \
  --weights ../../apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin \
  --config 4096,256,128,4,4,512 \
  --tokens 1,2,3,4,5,6,7,8 \
  --dump-vectors solid2-vectors.txt
```

The file records the config, a hash of the packed weights, the token window,
and the Q16 logits at the sampled position. It is the golden output an RTL
testbench compares against; see `docs/FPGA-CONFORMANCE.md`. A design that
replaces dynamic activation scales with static ones will not match these
vectors and needs its own set.

## Inspect an intermediate buffer

```sh
./integer-gpt \
  --weights ../../apps/only-gates-ai/public/models/tiny-gpt-solid2/weights.bin \
  --config 4096,256,128,4,4,512 \
  --tokens 1,2,3,4,5,6,7,8 \
  --dump-buffer qkv
```

Buffer names: `x`, `x1`, `h`, `qkv`, `attn`, `y`, `z`, `gz`, `m`, `logits`.
The first 32 Q16 values are printed. This is the fastest way to find the layer
where the integer path diverges from a float reference.

## What this harness does not do

- It does not tokenize or detokenize.
- It does not reproduce the browser's repetition penalty, which lives in the
  TypeScript host wrapper.
- It does not cache keys/values in the CLI generation loop: each `--steps` token
  still calls the full-context reference path. The incremental
  `tiny_gpt_forward_range` schedule is exercised by the C tests and the browser;
  the CLI stays on the simplest reference code.
- It is not bit-identical to a float forward pass. Quantization changes the
  distribution; the selftest checks invariants, not float equality.

## Files

```text
examples/integer-gpt/
  README.md    this file
  Makefile     build, float guard, selftest
  main.c       CLI harness around the real engine
```

The tutorial that explains every function is at `/learn/deep-dive/` in the
built site, with full source listings and the numeric contract.
