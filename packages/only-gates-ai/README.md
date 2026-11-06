# @only-gates-ai/core

TypeScript wrapper for the `only-gates-ai` differentiable logic circuit
WebAssembly module.

## Install

The package is part of this pnpm workspace:

```bash
pnpm install
pnpm --filter @only-gates-ai/core build
```

## Usage

```ts
import { createOnlyGatesAI } from "@only-gates-ai/core";

const og = await createOnlyGatesAI();

// Layers: 2 inputs -> 4 hidden gates -> 1 output.
const circuit = og.createCircuit([2, 4, 1]);
circuit.randomize(42);

circuit.setInput([true, false]);
circuit.feedForward();
console.log(circuit.getOutput());
```

Arrays are copied into WebAssembly memory automatically, so inputs and targets
can be regular JavaScript arrays or typed arrays.

### Inspecting and editing parameters

```ts
const connections = circuit.getConnections(); // Float32Array
circuit.setConnections(connections);

const gates = circuit.getGates(0); // 16 gate probabilities for cell 0
circuit.setGates(0, gates);
```

`getActivations()` returns the activations of every layer in one array.

### Training

```ts
const dataset = {
  inputs: [
    [false, false],
    [false, true],
    [true, false],
    [true, true],
  ],
  outputs: [[0], [1], [1], [1]],
};

const trainer = circuit.createTrainer(dataset, 0.05);

for (let epoch = 0; epoch < 200; epoch++) {
  trainer.trainEpoch();
}

trainer.destroy();
```

`trainer.step()` trains on one sample and returns its loss if you want to drive
the loop yourself. `trainer.reset()` restarts at the first sample.

### Hard circuit

```ts
const hard = circuit.createHardCircuit();
hard.setInput([true, false]);
hard.feedForward();
console.log(hard.getOutput());

console.log(hard.connections); // selected wire indices
console.log(hard.gates); // selected gate indices

hard.destroy();
circuit.destroy();
```

## Building

```bash
pnpm --filter @only-gates-ai/core build
```

This compiles the C module with Emscripten into `wasm/`, then emits JavaScript
and type declarations into `dist/`. The build requires the Emscripten SDK.

## Testing

```bash
pnpm --filter @only-gates-ai/core test
```
