#!/usr/bin/env python3
"""Vote-ensemble exported LUT4 circuits on MNIST, with per-class recall.

Pure Python (no torch) so it runs anywhere the test set is present. Each model
is a `--export-hard` JSON blob; votes are summed across models and the label
with the most votes wins.

    python3 ensemble.py a.json b.json c.json

Add ``--data DIR`` if the MNIST idx files are not in the default cache.
"""

import argparse
import json
import pathlib
import struct
import sys

DEFAULT_DATA = "/data/data/com.termux/files/usr/tmp/opencode/mnist"


def read_idx(path):
    data = pathlib.Path(path).read_bytes()
    magic = struct.unpack(">I", data[:4])[0]
    dims = magic & 0xFF
    sizes = struct.unpack(">" + "I" * dims, data[4 : 4 + 4 * dims])
    return sizes, data[4 + 4 * dims :]


def load_test(data_dir, count):
    (_, rows, cols), images = read_idx(pathlib.Path(data_dir) / "t10k-images-idx3-ubyte")
    _, labels = read_idx(pathlib.Path(data_dir) / "t10k-labels-idx1-ubyte")
    count = min(count, len(labels))
    samples = []
    for index in range(count):
        offset = index * rows * cols
        grid = []
        for y in range(14):
            for x in range(14):
                total = 0
                for dy in range(2):
                    for dx in range(2):
                        total += images[offset + (2 * y + dy) * cols + 2 * x + dx]
                grid.append(1 if total >= 255 else 0)
        samples.append((grid, labels[index]))
    return samples


def compile_model(path):
    model = json.loads(pathlib.Path(path).read_text())
    if model.get("kind") != "lut4":
        raise SystemExit(f"{path}: only lut4 exports are supported")
    layers = [(layer["wires"], layer["tables"]) for layer in model["layers"]]
    return model, layers


def votes(model, layers, bits):
    value = bits
    for wires, tables in layers:
        next_value = [0] * len(tables)
        for cell in range(len(tables)):
            base = cell * 4
            index = (
                (value[wires[base]] << 3)
                | (value[wires[base + 1]] << 2)
                | (value[wires[base + 2]] << 1)
                | value[wires[base + 3]]
            )
            next_value[cell] = (tables[cell] >> index) & 1
        value = next_value
    circuits = model["circuitsPerLabel"]
    totals = [0] * 10
    for cell, bit in enumerate(value):
        if bit:
            totals[cell // circuits] += 1
    return totals


def report(name, predictions, truth):
    correct = [0] * 10
    total = [0] * 10
    for prediction, label in zip(predictions, truth):
        total[label] += 1
        if prediction == label:
            correct[label] += 1
    recall = [correct[i] / total[i] if total[i] else 0.0 for i in range(10)]
    overall = sum(correct) / len(truth)
    print(
        f"{name:<28} overall {overall * 100:6.2f}%  worst {min(recall) * 100:6.2f}%"
    )
    print("  per-class: " + " ".join(f"{i}:{r * 100:.1f}" for i, r in enumerate(recall)))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--count", type=int, default=10000)
    args = parser.parse_args()

    samples = load_test(args.data, args.count)
    truth = [label for _, label in samples]
    compiled = [compile_model(path) for path in args.models]

    summed = [[0] * 10 for _ in samples]
    for path, (model, layers) in zip(args.models, compiled):
        predictions = []
        for index, (bits, _) in enumerate(samples):
            totals = votes(model, layers, bits)
            for label in range(10):
                summed[index][label] += totals[label]
            predictions.append(max(range(10), key=lambda label: totals[label]))
        report(pathlib.Path(path).name, predictions, truth)

    ensemble = [max(range(10), key=lambda label: row[label]) for row in summed]
    report(f"ensemble of {len(compiled)}", ensemble, truth)


if __name__ == "__main__":
    sys.exit(main())
