import assert from "node:assert/strict";
import test from "node:test";

import {
  createOnlyGatesAI,
  OnlyGatesError,
  OnlyGatesErrorCode,
} from "../dist/index.js";

async function createCircuit(shape) {
  const og = await createOnlyGatesAI();
  return { og, circuit: og.createCircuit(shape) };
}

test("creates, initializes and runs a circuit", async () => {
  const { circuit } = await createCircuit([2, 3, 1]);
  circuit.randomize(42);

  assert.equal(circuit.inputsLength, 2);
  assert.equal(circuit.outputsLength, 1);
  assert.equal(circuit.cellCount, 4);

  circuit.setInput([true, false]);
  circuit.feedForward();
  const output = circuit.getOutput();
  assert.equal(output.length, 1);
  assert.ok(Number.isFinite(output[0]));
  assert.ok(output[0] >= 0 && output[0] <= 1);

  assert.equal(circuit.getConnections().length, circuit.connectionProbabilitiesLength);
  assert.equal(circuit.getGates(0).length, 16);
  assert.equal(circuit.getActivations().length, 2 + 3 + 1);
  circuit.destroy();
});

test("copies parameter arrays in and out", async () => {
  const { circuit } = await createCircuit([2, 1, 1]);
  const connections = Float32Array.from([0.25, 0.75, 0.5, 0.5, 1, 1]);
  circuit.setConnections(connections);
  assert.deepEqual(Array.from(circuit.getConnections()), Array.from(connections));

  const gates = Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
  circuit.setGates(0, gates);
  assert.deepEqual(Array.from(circuit.getGates(0)), Array.from(gates));

  assert.throws(() => circuit.setConnections([1, 2]), RangeError);
  assert.throws(() => circuit.setGates(0, [1, 2]), RangeError);
  assert.throws(() => circuit.setGates(99, gates), RangeError);

  circuit.destroy();
  assert.throws(() => circuit.feedForward(), /destroyed/);
});

test("back-propagates through a deterministic AND circuit", async () => {
  const { og, circuit } = await createCircuit([2, 1, 1]);
  circuit.setConnections([1, 0, 0, 1, 1, 1]);
  circuit.setGates(
    0,
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  );
  circuit.setGates(
    1,
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  );
  circuit.setInput([true, false]);
  circuit.feedForward();
  assert.equal(circuit.getOutput()[0], 0);

  circuit.backpropagate([-1], 0.1);
  circuit.feedForward();
  assert.ok(circuit.getOutput()[0] > 0);

  const connections = circuit.getConnections();
  assert.ok(Math.abs(connections[0] + connections[1] - 1) < 1e-6);
  assert.ok(Math.abs(connections[2] + connections[3] - 1) < 1e-6);
  const gates = circuit.getGates(0);
  assert.ok(Math.abs(gates.reduce((sum, value) => sum + value, 0) - 1) < 1e-6);

  circuit.destroy();
  void og;
});

test("trains on a dataset with trainer.step and trainEpoch", async () => {
  const { circuit } = await createCircuit([2, 1, 1]);
  circuit.setConnections([0.6, 0.4, 0.4, 0.6, 1, 1]);
  circuit.setGates(0, [
    0, 0.25, 0, 0, 0, 0, 0.25, 0, 0.25, 0, 0, 0, 0, 0, 0.25, 0,
  ]);
  circuit.setGates(1, [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
  ]);

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
  assert.equal(trainer.sampleCount, 4);
  assert.equal(trainer.index, 0);

  const firstLoss = trainer.step();
  assert.ok(Number.isFinite(firstLoss) && firstLoss >= 0);
  assert.equal(trainer.index, 1);

  const initialLoss = trainer.trainEpoch(1);
  const finalLoss = trainer.trainEpoch(500);
  assert.ok(finalLoss < initialLoss, `${finalLoss} should be < ${initialLoss}`);

  trainer.reset();
  assert.equal(trainer.index, 0);
  trainer.learningRate = 0.01;
  assert.equal(trainer.learningRate, 0.01);

  trainer.destroy();
  assert.throws(() => trainer.step(), /destroyed/);
  circuit.destroy();
});

test("creates a hard circuit from the highest-probability choices", async () => {
  const { circuit } = await createCircuit([2, 2, 1]);
  circuit.setConnections([
    0.1, 0.9, // hidden 0 pin a -> input 1
    0.8, 0.2, // hidden 0 pin b -> input 0
    0.3, 0.7, // hidden 1 pin a -> input 1
    0.4, 0.6, // hidden 1 pin b -> input 1
    0.2, 0.8, // output pin a -> hidden 1
    0.9, 0.1, // output pin b -> hidden 0
  ]);
  circuit.setGates(0, [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
  ]);
  circuit.setGates(1, [
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
  ]);
  circuit.setGates(2, [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
  ]);

  const hard = circuit.createHardCircuit();
  assert.deepEqual(Array.from(hard.connections), [1, 0, 1, 1, 1, 0]);
  assert.deepEqual(Array.from(hard.gates), [8, 7, 14]);

  hard.setInput([true, false]);
  hard.feedForward();
  assert.equal(hard.getOutput()[0], 1);

  hard.destroy();
  assert.throws(() => hard.feedForward(), /destroyed/);
  circuit.destroy();
});

test("validates shapes and reports C errors", async () => {
  const og = await createOnlyGatesAI();
  assert.throws(
    () => og.createCircuit([]),
    (error) =>
      error instanceof OnlyGatesError &&
      error.code === OnlyGatesErrorCode.InvalidShape,
  );
  assert.throws(
    () => og.createCircuit([2, 0, 1]),
    (error) =>
      error instanceof OnlyGatesError &&
      error.code === OnlyGatesErrorCode.InvalidShape,
  );
});
