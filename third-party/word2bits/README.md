# Word2Bits vector subset

The mini language model in the web app bundles a small subset of the pretrained
1-bit word vectors published with **Word2Bits**:

- Project: <https://github.com/agnusmaximus/Word2Bits>
- Paper: Lam, "Word2Bits - Quantized Word Vectors" (2018),
  <https://arxiv.org/abs/1803.05651>
- Data: `w2b_bitlevel1_size800_vocab400K.tar.gz` (1-bit, 800 dimensions,
  400K vocabulary), linked from the project README.
- License: Apache License 2.0. A copy is in `LICENSE` in this directory.

## What is bundled

`apps/only-gates-ai/src/assets/word-vectors.ts` contains:

- the 4,096 most frequent alphabetic tokens from the top of the frequency-sorted
  vector file, lowercased and de-duplicated;
- 64 bits per word, sampled evenly across the 800 dimensions
  (`round(i * 800 / 64)` for `i = 0..63`);
- each value reduced to its sign and packed least-significant bit first.

The subset is derived data. It is not the full Word2Bits distribution, and it is
not intended for evaluating the original paper's results. The app uses it as a
small, fixed key memory for the experimental attention model.

## Regenerating

The source archive is about 86 MB compressed. Streaming the first part of the
archive is enough to produce the subset:

```sh
curl -L "https://drive.usercontent.google.com/download?id=107guTTy93J-y7UCO2ZA2spxRIFpoqhjh&export=download&confirm=t" \
  | tar -xzO \
  | head -n 12001 > w2b.txt
```

Then read the first token of each line after the header, keep alphabetic tokens,
and pack the signs of the selected dimensions. The generated asset records the
same provenance in its header comment.
