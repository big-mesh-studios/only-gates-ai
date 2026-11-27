#!/usr/bin/env python3
"""
Gate-wise convolutional detectors on MNIST (PyTorch / Colab edition).

This mirrors experiments/gate-cnn-mnist/run.mjs but runs fast enough to try
several architectures and hardening strategies. The gate semantics are the
same as src/main.c:

  - each cell has two input pins,
  - each pin is a softmax over its inputs (a local patch for convolution),
  - the cell has a softmax over the 16 Boolean gates,
  - soft output = sum_k p(gate=k) * soft_gate_k(a, b).

A convolutional cell shares its wire and gate distributions across every
spatial position. `conv-deep` stacks two such gates per detector before
pooling, which is the "multi-layer per feature detector" variant.

The final number that matters is the *hard* accuracy: argmax every wire and
gate, run the Boolean circuit. `--ste-epochs` fine-tunes with a straight-
through hard forward pass to make that hardened circuit better.

Usage:
  colab run --gpu T4 --timeout 1800 run_colab.py \
    --train 10000 --test 2000 --epochs 20 --ste-epochs 5

  python3 run_colab.py --quick   # local/CPU smoke test
"""

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Soft and hard gate semantics
# ---------------------------------------------------------------------------


def gate_values(a, b):
    one = torch.ones_like(a)
    ab = a * b
    xor = a + b - 2 * ab
    return torch.stack(
        [
            torch.zeros_like(a),  # 0 zero
            one - (a + b - ab),  # 1 nor
            a - ab,  # 2 lnt: a && !b
            one - a,  # 3 not_a
            b - ab,  # 4 rnt: !a && b
            one - b,  # 5 not_b
            xor,  # 6 xor
            one - ab,  # 7 nand
            ab,  # 8 and
            one - xor,  # 9 xnor
            b,  # 10 proj_b
            one - a + ab,  # 11 imp: !a || b
            a,  # 12 proj_a
            one - b + ab,  # 13 cimp: a || !b
            a + b - ab,  # 14 or
            one,  # 15 one
        ],
        dim=-1,
    )


def gate_values_bool(a, b):
    one = torch.ones_like(a, dtype=torch.bool)
    return torch.stack(
        [
            torch.zeros_like(a, dtype=torch.bool),
            ~(a | b),
            a & ~b,
            ~a,
            ~a & b,
            ~b,
            a ^ b,
            ~(a & b),
            a & b,
            ~(a ^ b),
            b,
            (~a) | b,
            a,
            a | (~b),
            a | b,
            one,
        ],
        dim=-1,
    )


def gate_mix(a, b, gate_probabilities):
    values = gate_values(a, b)
    channels = a.shape[1]
    if a.dim() == 4:
        weights = gate_probabilities.view(1, channels, 1, 1, 16)
    else:
        weights = gate_probabilities.view(1, channels, 16)
    return (values * weights).sum(-1).clamp(0, 1)


def gate_mix_bool(a, b, gate_indices):
    values = gate_values_bool(a, b)
    channels = a.shape[1]
    if a.dim() == 4:
        indices = gate_indices.view(1, channels, 1, 1, 1).expand(
            a.shape[0], channels, a.shape[2], a.shape[3], 1
        )
    else:
        indices = gate_indices.view(1, channels, 1).expand(a.shape[0], channels, 1)
    return torch.gather(values, -1, indices).squeeze(-1)


def gate_probabilities(logits, temperature, mask_constants):
    gates = F.softmax(logits / temperature, dim=1)
    if mask_constants:
        gates = gates.clone()
        gates[:, 0] = 0.0
        gates[:, 15] = 0.0
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return gates


def gate_indices(logits, mask_constants):
    if not mask_constants:
        return logits.argmax(dim=1)
    masked = logits.clone()
    masked[:, 0] = -1e9
    masked[:, 15] = -1e9
    return masked.argmax(dim=1)


def straight_through(probabilities):
    index = probabilities.argmax(dim=1, keepdim=True)
    hard = torch.zeros_like(probabilities).scatter_(1, index, 1.0)
    return probabilities + (hard - probabilities).detach()


def random_logits(shape, generator):
    """Dirichlet(1) sample, stored as logits. Matches the C core's init."""
    weights = -torch.log(torch.rand(shape, generator=generator).clamp_min(1e-12))
    weights = weights / weights.sum(dim=1, keepdim=True)
    return nn.Parameter(torch.log(weights))


# ---------------------------------------------------------------------------
# Gate layers
# ---------------------------------------------------------------------------


class GateConv(nn.Module):
    def __init__(self, in_channels, filters, kernel, stride, generator):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.in_channels = in_channels
        self.filters = filters
        self.wire_a = random_logits([filters, kernel * kernel * in_channels], generator)
        self.wire_b = random_logits([filters, kernel * kernel * in_channels], generator)
        self.gates = random_logits([filters, 16], generator)
        self.ste = False
        self.temperature = 1.0
        self.mask_constants = False

    def _kernel(self, weights):
        return weights.view(
            self.filters, self.kernel, self.kernel, self.in_channels
        ).permute(0, 3, 1, 2)

    def forward(self, x):
        wire_a = F.softmax(self.wire_a / self.temperature, dim=1)
        wire_b = F.softmax(self.wire_b / self.temperature, dim=1)
        if self.ste:
            wire_a = straight_through(wire_a)
            wire_b = straight_through(wire_b)
        a = F.conv2d(x, self._kernel(wire_a), stride=self.stride, padding=self.kernel // 2)
        b = F.conv2d(x, self._kernel(wire_b), stride=self.stride, padding=self.kernel // 2)
        gates = gate_probabilities(self.gates, self.temperature, self.mask_constants)
        if self.ste:
            gates = straight_through(gates)
        return gate_mix(a, b, gates)

    def hard_forward(self, x):
        kernel_size = self.kernel * self.kernel * self.in_channels
        kernel_a = torch.zeros(
            self.filters, kernel_size, device=x.device, dtype=torch.float32
        )
        kernel_b = torch.zeros_like(kernel_a)
        kernel_a.scatter_(1, self.wire_a.argmax(dim=1).unsqueeze(1), 1.0)
        kernel_b.scatter_(1, self.wire_b.argmax(dim=1).unsqueeze(1), 1.0)
        a = F.conv2d(
            x.float(),
            self._kernel(kernel_a),
            stride=self.stride,
            padding=self.kernel // 2,
        ) > 0.5
        b = F.conv2d(
            x.float(),
            self._kernel(kernel_b),
            stride=self.stride,
            padding=self.kernel // 2,
        ) > 0.5
        return gate_mix_bool(a, b, gate_indices(self.gates, self.mask_constants))


TRIT_VALUES = (0.0, 0.5, 1.0)


def hat_weights(value):
    """Piecewise-linear basis over the ternary nodes 0, 0.5, 1."""
    weight_0 = torch.clamp(1.0 - 2.0 * value, min=0.0)
    weight_2 = torch.clamp(2.0 * value - 1.0, min=0.0)
    weight_1 = 1.0 - weight_0 - weight_2
    return torch.stack([weight_0, weight_1, weight_2], dim=-1)


class GateDenseTri(nn.Module):
    """Dense cell with ternary signals {0, 0.5, 1} and an arbitrary 3x3 table.

    A ternary gate is exactly its 3x3 table, so 9 three-way softmaxes per cell
    (27 logits) span all 3^9 = 19,683 gates. Soft mode interpolates that table
    bilinearly at the continuous pin values.
    """

    def __init__(self, in_features, out_features, generator):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wire_a = random_logits([out_features, in_features], generator)
        self.wire_b = random_logits([out_features, in_features], generator)
        self.table = random_logits([out_features, 3, 3, 3], generator)
        self.temperature = 1.0
        self.ste = False

    def _table_values(self):
        probabilities = F.softmax(self.table / self.temperature, dim=-1)
        values = torch.tensor(TRIT_VALUES, device=probabilities.device)
        return (probabilities * values).sum(dim=-1)

    def forward(self, x):
        wire_a = F.softmax(self.wire_a / self.temperature, dim=1)
        wire_b = F.softmax(self.wire_b / self.temperature, dim=1)
        a = x @ wire_a.t()
        b = x @ wire_b.t()
        table = self._table_values()
        return torch.einsum(
            "noi,noj,oij->no", hat_weights(a), hat_weights(b), table
        )

    def hard_forward(self, x):
        a = x[:, self.wire_a.argmax(dim=1)].float()
        b = x[:, self.wire_b.argmax(dim=1)].float()
        a_index = (a * 2).round().long().clamp(0, 2)
        b_index = (b * 2).round().long().clamp(0, 2)
        values = torch.tensor(TRIT_VALUES, device=x.device)
        table = values[self.table.argmax(dim=-1)]  # [out,3,3]
        flat = table.reshape(self.out_features, 9)
        index = (a_index * 3 + b_index).unsqueeze(-1)
        gathered = torch.gather(
            flat.unsqueeze(0).expand(x.shape[0], -1, -1), 2, index
        )
        return gathered.squeeze(-1)


class GateDense3(nn.Module):
    """Dense cell with three input pins and an arbitrary 3-input Boolean gate.

    A 3-input gate is exactly its 8-bit truth table, so 8 Bernoulli logits per
    cell span all 2^8 = 256 gates. Soft mode interpolates the expected table
    trilinearly at the continuous pin values.
    """

    def __init__(self, in_features, out_features, generator):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wire_a = random_logits([out_features, in_features], generator)
        self.wire_b = random_logits([out_features, in_features], generator)
        self.wire_c = random_logits([out_features, in_features], generator)
        # Symmetric init: log(weights) would be negative everywhere and make
        # every 8-bit table harden to constant 0.
        self.table = nn.Parameter(
            torch.randn(out_features, 8, generator=generator) * 0.5
        )
        self.temperature = 1.0
        self.ste = False

    def forward(self, x):
        wire_a = F.softmax(self.wire_a / self.temperature, dim=1)
        wire_b = F.softmax(self.wire_b / self.temperature, dim=1)
        wire_c = F.softmax(self.wire_c / self.temperature, dim=1)
        a = x @ wire_a.t()
        b = x @ wire_b.t()
        c = x @ wire_c.t()
        table = torch.sigmoid(self.table / self.temperature).view(
            self.out_features, 2, 2, 2
        )
        weight_a = torch.stack([1.0 - a, a], dim=-1)
        weight_b = torch.stack([1.0 - b, b], dim=-1)
        weight_c = torch.stack([1.0 - c, c], dim=-1)
        return torch.einsum(
            "noi,noj,nok,oijk->no", weight_a, weight_b, weight_c, table
        )

    def hard_forward(self, x):
        a = x[:, self.wire_a.argmax(dim=1)].float()
        b = x[:, self.wire_b.argmax(dim=1)].float()
        c = x[:, self.wire_c.argmax(dim=1)].float()
        index = (a * 4 + b * 2 + c).long().unsqueeze(-1)
        bits = (self.table > 0).float()
        return torch.gather(
            bits.unsqueeze(0).expand(x.shape[0], -1, -1), 2, index
        ).squeeze(-1)


class GateDense4(nn.Module):
    """Dense cell with four input pins and an arbitrary 4-input Boolean gate.

    A 4-input gate is its 16-bit truth table, so 16 Bernoulli logits per cell
    span all 2^16 = 65,536 gates. Soft mode interpolates the expected table
    multilinearly at the continuous pin values.
    """

    def __init__(self, in_features, out_features, generator):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wire_a = random_logits([out_features, in_features], generator)
        self.wire_b = random_logits([out_features, in_features], generator)
        self.wire_c = random_logits([out_features, in_features], generator)
        self.wire_d = random_logits([out_features, in_features], generator)
        self.table = nn.Parameter(
            torch.randn(out_features, 16, generator=generator) * 0.5
        )
        self.temperature = 1.0
        self.ste = False

    def forward(self, x):
        wires = [
            F.softmax(getattr(self, name) / self.temperature, dim=1)
            for name in ("wire_a", "wire_b", "wire_c", "wire_d")
        ]
        values = [x @ wire.t() for wire in wires]
        weights = [
            torch.stack([1.0 - value, value], dim=-1) for value in values
        ]
        weight = (
            weights[0].unsqueeze(3).unsqueeze(4).unsqueeze(5)
            * weights[1].unsqueeze(2).unsqueeze(4).unsqueeze(5)
            * weights[2].unsqueeze(2).unsqueeze(3).unsqueeze(5)
            * weights[3].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        ).reshape(x.shape[0], self.out_features, 16)
        table = torch.sigmoid(self.table / self.temperature)
        return (weight * table.unsqueeze(0)).sum(dim=-1)

    def hard_forward(self, x):
        a = x[:, self.wire_a.argmax(dim=1)].float()
        b = x[:, self.wire_b.argmax(dim=1)].float()
        c = x[:, self.wire_c.argmax(dim=1)].float()
        d = x[:, self.wire_d.argmax(dim=1)].float()
        index = (a * 8 + b * 4 + c * 2 + d).long().unsqueeze(-1)
        bits = (self.table > 0).float()
        return torch.gather(
            bits.unsqueeze(0).expand(x.shape[0], -1, -1), 2, index
        ).squeeze(-1)


class GateDenseNet4(nn.Module):
    def __init__(self, input_size, hidden, output_size, generator):
        super().__init__()
        layers = []
        previous = input_size
        for size in [*hidden, output_size]:
            layers.append(GateDense4(previous, size, generator))
            previous = size
        self.layers = nn.ModuleList(layers)
        self.output_size = output_size

    def forward(self, x):
        value = x.flatten(1)
        for layer in self.layers:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool().flatten(1).float()
        for layer in self.layers:
            value = layer.hard_forward(value)
        return value


class GateDenseNet3(nn.Module):
    def __init__(self, input_size, hidden, output_size, generator):
        super().__init__()
        layers = []
        previous = input_size
        for size in [*hidden, output_size]:
            layers.append(GateDense3(previous, size, generator))
            previous = size
        self.layers = nn.ModuleList(layers)
        self.output_size = output_size

    def forward(self, x):
        value = x.flatten(1)
        for layer in self.layers:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool().flatten(1).float()
        for layer in self.layers:
            value = layer.hard_forward(value)
        return value


class GateDenseNetTri(nn.Module):
    def __init__(self, input_size, hidden, output_size, generator):
        super().__init__()
        layers = []
        previous = input_size
        for size in [*hidden, output_size]:
            layers.append(GateDenseTri(previous, size, generator))
            previous = size
        self.layers = nn.ModuleList(layers)
        self.output_size = output_size

    def forward(self, x):
        value = x.flatten(1)
        for layer in self.layers:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool().flatten(1).float()
        for layer in self.layers:
            value = layer.hard_forward(value)
        return value


class GateDense(nn.Module):
    def __init__(self, in_features, out_features, generator):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wire_a = random_logits([out_features, in_features], generator)
        self.wire_b = random_logits([out_features, in_features], generator)
        self.gates = random_logits([out_features, 16], generator)
        self.ste = False
        self.temperature = 1.0
        self.mask_constants = False

    def forward(self, x):
        wire_a = F.softmax(self.wire_a / self.temperature, dim=1)
        wire_b = F.softmax(self.wire_b / self.temperature, dim=1)
        if self.ste:
            wire_a = straight_through(wire_a)
            wire_b = straight_through(wire_b)
        a = x @ wire_a.t()
        b = x @ wire_b.t()
        gates = gate_probabilities(self.gates, self.temperature, self.mask_constants)
        if self.ste:
            gates = straight_through(gates)
        return gate_mix(a, b, gates)

    def hard_forward(self, x):
        a = x[:, self.wire_a.argmax(dim=1)]
        b = x[:, self.wire_b.argmax(dim=1)]
        return gate_mix_bool(a, b, gate_indices(self.gates, self.mask_constants))


class GateBlock(nn.Module):
    def __init__(self, layers, pool):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.pool = pool


class GateConvNet(nn.Module):
    def __init__(self, input_size, blocks_spec, head_sizes, output_size, generator):
        super().__init__()
        blocks = []
        in_channels = 1
        height = input_size
        width = input_size
        for spec in blocks_spec:
            layers = []
            for layer_spec in spec["layers"]:
                layers.append(
                    GateConv(
                        in_channels,
                        layer_spec["filters"],
                        layer_spec.get("kernel", 3),
                        layer_spec.get("stride", 1),
                        generator,
                    )
                )
                in_channels = layer_spec["filters"]
                height = math.ceil(height / layer_spec.get("stride", 1))
                width = math.ceil(width / layer_spec.get("stride", 1))
            blocks.append(GateBlock(layers, spec["pool"]))
            if spec["pool"] == "max2":
                height = math.ceil(height / 2)
                width = math.ceil(width / 2)
            elif spec["pool"] == "global":
                height = 1
                width = 1
        self.blocks = nn.ModuleList(blocks)

        head_input = height * width * in_channels
        head = []
        for size in [*head_sizes, output_size]:
            head.append(GateDense(head_input, size, generator))
            head_input = size
        self.head = nn.ModuleList(head)
        self.output_size = output_size

    def forward(self, x):
        value = x
        for block in self.blocks:
            for layer in block.layers:
                value = layer(value)
            if block.pool == "max2":
                value = F.max_pool2d(value, 2, 2, ceil_mode=True)
            elif block.pool == "global":
                value = value.amax(dim=(2, 3))
        value = value.flatten(1)
        for layer in self.head:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool()
        for block in self.blocks:
            for layer in block.layers:
                value = layer.hard_forward(value)
            if block.pool == "max2":
                value = F.max_pool2d(value.float(), 2, 2, ceil_mode=True) > 0.5
            elif block.pool == "global":
                value = value.float().amax(dim=(2, 3)) > 0.5
        value = value.flatten(1)
        for layer in self.head:
            value = layer.hard_forward(value)
        return value

class GateDenseNet(nn.Module):
    def __init__(self, input_size, hidden, output_size, generator):
        super().__init__()
        layers = []
        previous = input_size
        for size in [*hidden, output_size]:
            layers.append(GateDense(previous, size, generator))
            previous = size
        self.layers = nn.ModuleList(layers)
        self.output_size = output_size

    def forward(self, x):
        value = x.flatten(1)
        for layer in self.layers:
            value = layer(value)
        return value

    def hard_forward(self, x):
        value = x.bool().flatten(1)
        for layer in self.layers:
            value = layer.hard_forward(value)
        return value


class FloatCNN(nn.Module):
    """Reference point: a normal small CNN, not a gate circuit."""

    def __init__(self, output_size=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2, ceil_mode=True)
        x = F.relu(self.conv2(x))
        x = x.amax(dim=(2, 3))
        return F.softmax(self.fc(x), dim=1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_mnist(size, threshold, train_count, test_count, seed, device, augment=False):
    if 28 % size != 0:
        raise ValueError("size must divide 28")
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )

    def raw(dataset):
        return dataset.data.float().unsqueeze(1) / 255.0

    def prepare(images):
        pooled = F.avg_pool2d(images, 28 // size)
        return (pooled >= threshold).float()

    raw_train = raw(train_dataset)
    raw_test = raw(test_dataset)
    train_x = prepare(raw_train)
    test_x = prepare(raw_test)
    train_y = train_dataset.targets.long()
    test_y = test_dataset.targets.long()

    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(len(train_x), generator=generator)[:train_count]
    test_indices = torch.randperm(len(test_x), generator=generator)[:test_count]

    result = (
        train_x[train_indices].to(device),
        train_y[train_indices].to(device),
        test_x[test_indices].to(device),
        test_y[test_indices].to(device),
    )
    if augment:
        return result + (raw_train[train_indices].to(device),)
    return result


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def set_ste(model, enabled):
    for module in model.modules():
        if isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            module.ste = enabled


def set_temperature(model, value):
    for module in model.modules():
        if isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            module.temperature = value


def set_mask_constants(model, enabled):
    for module in model.modules():
        if isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            module.mask_constants = enabled


def wire_parameters(module):
    for name in ("wire_a", "wire_b", "wire_c", "wire_d"):
        parameter = getattr(module, name, None)
        if parameter is not None:
            yield name, parameter


def freeze_wires(model, generator, first_only=False):
    """Pin wires to one input and stop training them.

    Learning drives each wire softmax onto a favourite input, and the whole
    layer then collapses onto a handful of pixels. Covering the fan-in with a
    shuffled round-robin keeps the hard circuit's input coverage high. With
    ``first_only`` just layer 0 is pinned, which guarantees every input is
    reachable while leaving deeper wires free to learn.
    """
    seen = 0
    for module in model.modules():
        if not isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            continue
        if first_only and seen > 0:
            break
        seen += 1
        for _, parameter in wire_parameters(module):
            count, fan_in = parameter.shape
            slots = []
            while len(slots) < count:
                slots.extend(torch.randperm(fan_in, generator=generator).tolist())
            slots = torch.tensor(slots[:count], device=parameter.device)
            with torch.no_grad():
                parameter.zero_()
                parameter[torch.arange(count, device=parameter.device), slots] = 12.0
            parameter.requires_grad_(False)


def stacked_readout(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    kind="gate",
    hidden=(32,),
    steps=20,
    lr=0.05,
    generator=None,
):
    """Train a second-level combiner on the hardened forest bits.

    The shipped readout is an unweighted vote: it can only count. A second
    gate layer over the 10*K bits can learn non-additive combinations (AND of
    two circuits, vetoes, etc.) while staying Boolean and tiny.
    """
    train_bits = hard_outputs(model, train_x)
    test_bits = hard_outputs(model, test_x)
    features = train_bits.shape[1]

    if kind == "linear":
        head = torch.nn.Linear(features, 10).to(train_bits.device)
        optimiser = torch.optim.Adam(head.parameters(), lr=0.02, weight_decay=1e-4)
        for _ in range(400):
            loss = F.cross_entropy(head(train_bits), train_y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        accuracy = (head(test_bits).argmax(dim=1) == test_y).float().mean().item()
        return {"kind": "linear", "accuracy": accuracy, "cells": 0}

    head = GateDenseNet(features, list(hidden), 10, generator).to(train_bits.device)
    optimiser = torch.optim.Adam(head.parameters(), lr=lr)
    for epoch in range(steps):
        temperature = 1.0 * (0.25 ** (epoch / max(1, steps - 1)))
        for module in head.modules():
            if isinstance(module, (GateDense, GateDenseTri, GateDense3, GateDense4)):
                module.temperature = temperature
        loss = F.cross_entropy(head(train_bits), train_y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    soft = (head(test_bits).argmax(dim=1) == test_y).float().mean().item()
    hard = (
        (head.hard_forward(test_bits > 0.5).float().argmax(dim=1) == test_y)
        .float()
        .mean()
        .item()
    )
    return {
        "kind": "gate",
        "soft": soft,
        "hard": hard,
        "cells": sum(layer.out_features for layer in head.layers),
    }


def class_scores(values, circuits_per_label):
    """Sum the K circuit votes for each label, like the shipped forest readout."""
    if circuits_per_label <= 1:
        return values
    return values.reshape(values.shape[0], -1, circuits_per_label).sum(dim=2)


def soft_accuracy(model, x, y, circuits_per_label=1, batch=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output = class_scores(
                model(x[start : start + batch]), circuits_per_label
            )
            correct += (output.argmax(dim=1) == y[start : start + batch]).sum().item()
    return correct / len(x)


def hard_accuracy(
    model, x, y, circuits_per_label=1, batch=512, threshold=None
):
    model.eval()
    correct = 0
    all_zero = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            output = model.hard_forward(x[start : start + batch]).float()
            any_set = output.any(dim=1)
            if threshold is not None:
                output = (output >= threshold).float()
            votes = class_scores(output, circuits_per_label)
            prediction = votes.argmax(dim=1)
            target = y[start : start + batch]
            correct += (prediction == target).sum().item()
            all_zero += (~any_set).sum().item()
    return correct / len(x), all_zero / len(x)


def forest_loss(output, labels, circuits_per_label, scale=8.0):
    """Cross-entropy over the per-label mean vote.

    A plain BCE over 10*K outputs is dominated by the 9*K negatives, so the
    loss drops while the actual vote decision stays weak. Training the summed
    scores directly matches the readout, but the sum's range grows with K, so
    the loss would get sharper just by growing the forest. Dividing by K keeps
    the logit range (and therefore the optimisation problem) fixed, while
    leaving the vote ordering — and hence the readout — unchanged.
    """
    scores = class_scores(output, circuits_per_label) / circuits_per_label
    return F.cross_entropy(scores * scale, labels)


def hard_outputs(model, x, batch=512):
    """Collect the Boolean output bits for an entire dataset."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunks.append(model.hard_forward(x[start : start + batch]).float())
    return torch.cat(chunks, 0)


def pruned_hard_accuracy(model, train_x, test_x, test_y, circuits_per_label):
    """Drop circuits that are constant on the training set, then re-vote.

    A circuit that always fires adds a fixed vote to its label, which biases
    the forest; one that never fires is dead weight. Removing both leaves the
    discriminating circuits and costs nothing at inference time.
    """
    train_bits = hard_outputs(model, train_x)
    active = (train_bits.mean(dim=0) > 0) & (train_bits.mean(dim=0) < 1)
    bits = hard_outputs(model, test_x)
    shaped = bits.reshape(bits.shape[0], 10, circuits_per_label)
    active_shaped = active.reshape(10, circuits_per_label)
    votes = (shaped * active_shaped.unsqueeze(0)).sum(dim=2)
    prediction = votes.argmax(dim=1)
    accuracy = (prediction == test_y).sum().item() / len(test_y)
    return accuracy, active.float().mean().item()


def pin_wires(model, generator, fraction):
    """Pin a fraction of each layer's wires to a shuffled round-robin cover.

    Fully learned wires cover only part of the fan-in; fully random wires keep
    coverage but lose feature quality. Pinning a fraction keeps every layer's
    reachable set large while the remaining wires still learn. Pinned rows are
    re-imposed after every optimiser step (see ``reimpose_pinned``).
    """
    for module in model.modules():
        if not isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            continue
        for index, (_, parameter) in enumerate(wire_parameters(module)):
            name = f"pin_{index}"
            count, fan_in = parameter.shape
            keep = int(round(count * fraction))
            if keep <= 0:
                continue
            rows = torch.randperm(count, generator=generator)[:keep]
            slots = []
            while len(slots) < keep:
                slots.extend(torch.randperm(fan_in, generator=generator).tolist())
            slots = torch.tensor(slots[:keep], device=parameter.device)
            setattr(module, name, (rows, slots))
            with torch.no_grad():
                parameter[rows] = 0.0
                parameter[rows, slots] = 12.0


def reimpose_pinned(model):
    """Restore pinned wire rows after an optimiser step."""
    for module in model.modules():
        if not isinstance(module, (GateConv, GateDense, GateDenseTri, GateDense3, GateDense4)):
            continue
        for index, (_, parameter) in enumerate(wire_parameters(module)):
            pinned = getattr(module, f"pin_{index}", None)
            if pinned is None:
                continue
            rows, slots = pinned
            with torch.no_grad():
                parameter[rows] = 0.0
                parameter[rows, slots] = 12.0


def receptive_field(model):
    """How many input bits can each output bit actually depend on?

    A hardened cell reads exactly two wires from the previous layer, so its
    dependency set is the union of those two cells' sets. That can at most
    double per layer: depth 1 sees 2 inputs, depth 2 sees 4, depth 3 sees 8,
    and so on. This measures the realised sets, which are usually smaller than
    the bound because the two wires can share ancestors.
    """
    if not isinstance(model, GateDenseNet):
        return None
    current = None
    per_layer = []
    for index, layer in enumerate(model.layers):
        a = layer.wire_a.argmax(dim=1).tolist()
        b = layer.wire_b.argmax(dim=1).tolist()
        if index == 0:
            sets = [{left, right} for left, right in zip(a, b)]
        else:
            sets = [current[left] | current[right] for left, right in zip(a, b)]
        current = sets
        sizes = [len(entry) for entry in sets]
        union = set().union(*sets) if sets else set()
        per_layer.append(
            {
                "layer": index,
                "cells": len(sets),
                "max": max(sizes),
                "mean": sum(sizes) / len(sizes),
                "union": len(union),
            }
        )
    return per_layer


def dense_simplification(model):
    """Lower bound on how much a hardened dense circuit can shrink.

    Classifies every cell by its argmax gate:

    - constant (`0`, `15`) ignores both wires and folds to a literal;
    - identity (`12` -> a, `10` -> b) folds to a wire and the cell disappears;
    - inverter (`3` -> NOT a, `5` -> NOT b) keeps a cell but needs one input
      instead of two;
    - everything else is a genuine two-input gate.

    The walk is backward from the output, so folding and dead-cell elimination
    cascade: once a cell folds, whatever only fed it becomes dead.
    """
    if not isinstance(model, GateDenseNet):
        return None
    layers = []
    for layer in model.layers:
        layers.append(
            {
                "out": layer.out_features,
                "gate": layer.gates.argmax(dim=1).tolist(),
                "a": layer.wire_a.argmax(dim=1).tolist(),
                "b": layer.wire_b.argmax(dim=1).tolist(),
            }
        )

    identity = {12: "a", 10: "b"}
    inverter = {3: "a", 5: "b"}
    needed = [set() for _ in layers]
    needed[-1] = set(range(layers[-1]["out"]))

    dead = 0
    folded = 0
    inverted = 0
    total = sum(layer["out"] for layer in layers)
    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        upstream = needed[index - 1] if index > 0 else None
        for cell in needed[index]:
            gate = layer["gate"][cell]
            if gate in (0, 15):
                folded += 1
                continue
            if gate in identity:
                folded += 1
                if upstream is not None:
                    upstream.add(layer[identity[gate]][cell])
                continue
            if gate in inverter:
                inverted += 1
                if upstream is not None:
                    upstream.add(layer[inverter[gate]][cell])
                continue
            if upstream is not None:
                upstream.add(layer["a"][cell])
                upstream.add(layer["b"][cell])
        if index > 0:
            dead += layer["out"] - len(needed[index])

    remaining = total - dead - folded
    return {
        "cells": total,
        "dead": dead,
        "folded": folded,
        "inverted": inverted,
        "remaining": remaining,
        "wires": total * 2 - folded * 2 - inverted,
    }


def circuit_quality(model, x, y, circuits_per_label):
    """Balanced accuracy of each single circuit bit, per label.

    Positive rate is ~10%, so plain accuracy would rank circuits that always
    answer "no" highly. Balanced accuracy (TPR + TNR) / 2 does not.
    """
    bits = hard_outputs(model, x).reshape(-1, 10, circuits_per_label) > 0.5
    one_hot = F.one_hot(y, 10).bool()
    quality = torch.zeros(10, circuits_per_label)
    for label in range(10):
        positive = one_hot[:, label]
        for circuit in range(circuits_per_label):
            bit = bits[:, label, circuit]
            tpr = bit[positive].float().mean().item() if positive.any() else 0.0
            tnr = (~bit[~positive]).float().mean().item() if (~positive).any() else 0.0
            quality[label, circuit] = (tpr + tnr) / 2
    return quality


def ensemble_curve(model, train_x, train_y, test_x, test_y, circuits_per_label):
    """Accuracy of the forest after keeping only the best m circuits per label.

    This is the "simplify the ensemble" question: a forest is redundant if a
    few of its circuits carry the decision. Circuits are ranked on the training
    set by balanced accuracy, then the truncated forest is re-voted on test.
    """
    if circuits_per_label <= 1:
        return []
    quality = circuit_quality(model, train_x, train_y, circuits_per_label)
    bits = hard_outputs(model, test_x).reshape(-1, 10, circuits_per_label)
    curve = []
    for keep in range(1, circuits_per_label + 1):
        mask = torch.zeros(10, circuits_per_label, dtype=torch.bool)
        mask.scatter_(1, quality.topk(keep, dim=1).indices, True)
        votes = (bits * mask.unsqueeze(0)).sum(dim=2)
        accuracy = (votes.argmax(dim=1) == test_y).float().mean().item()
        curve.append(
            {
                "keep": keep,
                "circuits": int(mask.sum().item()),
                "accuracy": accuracy,
            }
        )
    return curve


def greedy_ensemble_curve(
    model, train_x, train_y, test_x, test_y, circuits_per_label, split=0.8
):
    """Greedy forward selection of circuits, scoring marginal contribution.

    Ranking circuits by their own accuracy is not the same as picking the set
    that votes best: the top few circuits tend to be correlated, so the value
    of a circuit is what it adds to the current ensemble. This starts from no
    circuits and repeatedly adds the (label, circuit) pair that most improves
    vote accuracy on a held-out slice of the training set.
    """
    if circuits_per_label <= 1:
        return []
    cut = int(len(train_x) * split)
    validation_bits = (
        hard_outputs(model, train_x[cut:]).reshape(-1, 10, circuits_per_label) > 0.5
    )
    test_bits = (
        hard_outputs(model, test_x).reshape(-1, 10, circuits_per_label) > 0.5
    )
    validation_y = train_y[cut:]

    votes_validation = torch.zeros(len(validation_y), 10, device=validation_y.device)
    votes_test = torch.zeros(len(test_y), 10, device=test_y.device)
    remaining = [
        (label, circuit)
        for label in range(10)
        for circuit in range(circuits_per_label)
    ]
    curve = []
    for step in range(len(remaining)):
        best = None
        for label, circuit in remaining:
            trial = votes_validation.clone()
            trial[:, label] += validation_bits[:, label, circuit].float()
            accuracy = (
                (trial.argmax(dim=1) == validation_y).float().mean().item()
            )
            if best is None or accuracy > best[0]:
                best = (accuracy, label, circuit)
        _, label, circuit = best
        remaining.remove((label, circuit))
        votes_validation[:, label] += validation_bits[:, label, circuit].float()
        votes_test[:, label] += test_bits[:, label, circuit].float()
        curve.append(
            {
                "circuits": step + 1,
                "accuracy": (
                    (votes_test.argmax(dim=1) == test_y).float().mean().item()
                ),
            }
        )
    return curve


def gate_stats(model):
    """How many learned cells are locally simplifiable?

    Constant gates (0 and 15) ignore both wires. The six unary/constant gates
    (0, 3, 5, 10, 12, 15) ignore at least one wire, and a cell whose two wires
    pick the same input collapses to a unary function as well.
    """
    unary = {0, 3, 5, 10, 12, 15}
    cells = 0
    constant = 0
    unary_gates = 0
    same_wires = 0
    conv_cells = 0
    conv_trivial = 0
    for module in model.modules():
        if isinstance(module, (GateConv, GateDense)):
            chosen = module.gates.argmax(dim=1)
            count = chosen.numel()
            cells += count
            trivial = sum((chosen == gate).sum().item() for gate in unary)
            constant += ((chosen == 0) | (chosen == 15)).sum().item()
            unary_gates += trivial
            same_wires += (
                module.wire_a.argmax(dim=1) == module.wire_b.argmax(dim=1)
            ).sum().item()
            if isinstance(module, GateConv):
                conv_cells += count
                conv_trivial += trivial
    return {
        "cells": cells,
        "constant": constant,
        "unaryGates": unary_gates,
        "sameWires": same_wires,
        "convCells": conv_cells,
        "convSingleTap": conv_trivial,
    }


def augment_batch(images, generator, rotation, scale, translate):
    """Random rotation/scale/translation on raw 28x28 float images.

    Applied before pooling and thresholding so the augmentation is not eaten by
    the quantisation step.
    """
    count = images.shape[0]
    device = images.device
    angles = (
        (torch.rand(count, generator=generator, device=device) * 2 - 1)
        * rotation
        * math.pi
        / 180.0
    )
    scales = 1.0 + (torch.rand(count, generator=generator, device=device) * 2 - 1) * scale
    shift_x = (torch.rand(count, generator=generator, device=device) * 2 - 1) * translate
    shift_y = (torch.rand(count, generator=generator, device=device) * 2 - 1) * translate
    cos = torch.cos(angles) / scales
    sin = torch.sin(angles) / scales
    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = shift_x
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = shift_y
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


def train(model, train_x, train_y, test_x, test_y, args, hard_capable, train_raw=None):
    # The float reference dies with relu at the gate models' high LR; keep it
    # at a conventional CNN learning rate.
    lr = args.lr if hard_capable else min(args.lr, 0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total = len(train_x)
    ste_epochs = args.ste_epochs if hard_capable else 0
    soft_epochs = max(0, args.epochs - ste_epochs)
    circuits_per_label = args.circuits_per_label if hard_capable else 1
    hard_threshold = (
        args.tri_threshold if isinstance(model, GateDenseNetTri) else None
    )
    history = []

    augment_generator = torch.Generator(device=train_x.device)
    augment_generator.manual_seed(args.seed + 1)

    def run_epoch(epoch, phase, temperature):
        model.train()
        permutation = torch.randperm(total, device=train_x.device)
        loss_sum = 0.0
        batches = 0
        for start in range(0, total, args.batch):
            indices = permutation[start : start + args.batch]
            if train_raw is not None and args.augment:
                raw_batch = augment_batch(
                    train_raw[indices],
                    augment_generator,
                    args.augment_rotation,
                    args.augment_scale,
                    args.augment_translate,
                )
                pooled = F.avg_pool2d(raw_batch, 28 // args.size)
                batch_x = (pooled >= args.threshold).float()
            else:
                batch_x = train_x[indices]
            output = model(batch_x)
            loss = forest_loss(
                output,
                train_y[indices],
                circuits_per_label,
                args.score_scale,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            reimpose_pinned(model)
            loss_sum += loss.item()
            batches += 1
        soft = soft_accuracy(model, test_x, test_y, circuits_per_label)
        line = {
            "epoch": epoch + 1,
            "phase": phase,
            "temperature": temperature,
            "loss": loss_sum / batches,
            "soft": soft,
        }
        if hard_capable:
            hard, zero = hard_accuracy(
                model,
                test_x,
                test_y,
                circuits_per_label,
                threshold=hard_threshold,
            )
            line["hard"] = hard
            line["hardZero"] = zero
        print(
            f"  epoch {epoch + 1:3d}/{args.epochs} {phase:4s} T {temperature:4.2f} "
            f"loss {line['loss']:.4f} soft {soft * 100:5.2f}%"
            + (
                f" hard {line['hard'] * 100:5.2f}% zero {line['hardZero'] * 100:5.1f}%"
                if hard_capable
                else ""
            )
        )
        history.append(line)

    # Anneal the softmax temperature down towards the operating point that
    # argmax hardening actually evaluates. Jumping from T=1 straight to argmax
    # is what collapsed the earlier runs.
    for epoch in range(soft_epochs):
        if soft_epochs <= 1:
            temperature = args.t_min
        else:
            ratio = epoch / (soft_epochs - 1)
            temperature = args.t_start * (args.t_min / args.t_start) ** ratio
        set_temperature(model, temperature)
        run_epoch(epoch, "soft", temperature)

    before = None
    if hard_capable and soft_epochs > 0:
        before_soft = soft_accuracy(model, test_x, test_y, circuits_per_label)
        before_hard, before_zero = hard_accuracy(
            model,
            test_x,
            test_y,
            circuits_per_label,
            threshold=hard_threshold,
        )
        before = {
            "soft": before_soft,
            "hard": before_hard,
            "hardZero": before_zero,
        }
        print(
            f"  before STE: T {args.t_min:4.2f} soft {before_soft * 100:5.2f}% "
            f"hard {before_hard * 100:5.2f}% zero {before_zero * 100:5.1f}%"
        )

    if ste_epochs > 0:
        # Keep the relaxation at the annealed operating point: the STE
        # gradient is taken from the soft model, so a flat T=1 relaxation
        # points away from the argmax circuit we are actually training.
        set_ste(model, True)
        set_temperature(model, args.t_min)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ste_lr)
        for offset in range(ste_epochs):
            run_epoch(soft_epochs + offset, "ste", args.t_min)
        set_ste(model, False)
        set_temperature(model, args.t_min)

    return history, before


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def _build_lut3_costs():
    """Minimal two-input-gate cost of every 3-input Boolean function.

    Truth tables are 8-bit masks over the index 4a + 2b + c, matching
    GateDense3's table layout. Breadth-first over all 256 functions using the
    16 two-input gates.
    """
    constants = 0b11110000, 0b11001100, 0b10101010, 0, 0xFF

    def apply_gate(gate, x, y):
        out = 0
        for index in range(8):
            a = (x >> index) & 1
            b = (y >> index) & 1
            table = (
                0, int(not (a or b)), int(a and not b), int(not a),
                int((not a) and b), int(not b), int(a != b), int(not (a and b)),
                int(a and b), int(a == b), int(b), int((not a) or b),
                int(a), int(a or (not b)), int(a or b), 1,
            )
            out |= table[gate] << index
        return out

    cost = {value: 0 for value in constants}
    while True:
        added = {}
        known = list(cost)
        for gate_index in range(16):
            for x in known:
                for y in known:
                    table = apply_gate(gate_index, x, y)
                    if table not in cost and table not in added:
                        added[table] = cost[x] + cost[y] + 1
        if not added:
            break
        cost.update(added)
    return [cost[value] for value in range(256)]


LUT3_COSTS = _build_lut3_costs()


def lut3_gate_cost(model):
    """Compiled two-input gate count of a trained LUT3 circuit."""
    if not isinstance(model, GateDenseNet3):
        return None
    cells = 0
    total = 0
    distribution = {}
    for layer in model.layers:
        bits = (layer.table > 0).long()
        for row in bits:
            value = 0
            for index in range(8):
                value |= int(row[index].item()) << index
            cost = LUT3_COSTS[value]
            total += cost
            cells += 1
            distribution[cost] = distribution.get(cost, 0) + 1
    return {
        "cells": cells,
        "compiledGates": total,
        "meanPerCell": total / cells if cells else 0.0,
        "distribution": distribution,
    }


def gate_cell_count(model, input_size):
    if isinstance(
        model, (GateDenseNet, GateDenseNetTri, GateDenseNet3, GateDenseNet4)
    ):
        return sum(layer.out_features for layer in model.layers)

    count = 0
    height = width = input_size
    channels = 1
    for block in model.blocks:
        for layer in block.layers:
            height = math.ceil(height / layer.stride)
            width = math.ceil(width / layer.stride)
            channels = layer.filters
            count += height * width * channels
        if block.pool == "max2":
            height = math.ceil(height / 2)
            width = math.ceil(width / 2)
        elif block.pool == "global":
            height = width = 1
    for layer in model.head:
        count += layer.out_features
    return count


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def build_model(
    name,
    input_size,
    generator,
    device,
    circuits_per_label=1,
    dense_hidden=(128, 64),
):
    outputs = 10 * circuits_per_label
    if name == "dense":
        model = GateDenseNet(
            input_size * input_size, list(dense_hidden), outputs, generator
        )
    elif name == "dense-wide":
        model = GateDenseNet(input_size * input_size, [256, 128], outputs, generator)
    elif name == "dense-lut4":
        model = GateDenseNet4(
            input_size * input_size, list(dense_hidden), outputs, generator
        )
    elif name == "dense-lut3":
        model = GateDenseNet3(
            input_size * input_size, list(dense_hidden), outputs, generator
        )
    elif name == "dense-tri":
        model = GateDenseNetTri(
            input_size * input_size, list(dense_hidden), outputs, generator
        )
    elif name == "dense-small":
        # ~30k params, matched to conv-grid-head for a fair per-parameter test.
        model = GateDenseNet(input_size * input_size, [48, 48], outputs, generator)
    elif name == "conv-shallow":
        model = GateConvNet(
            input_size,
            [
                {"layers": [{"filters": 8, "kernel": 3, "stride": 2}], "pool": "max2"},
                {"layers": [{"filters": 16, "kernel": 3, "stride": 2}], "pool": "global"},
            ],
            [64, 64],
            outputs,
            generator,
        )
    elif name == "conv-deep":
        model = GateConvNet(
            input_size,
            [
                {
                    "layers": [
                        {"filters": 8, "kernel": 3, "stride": 1},
                        {"filters": 8, "kernel": 3, "stride": 1},
                    ],
                    "pool": "max2",
                },
                {
                    "layers": [
                        {"filters": 16, "kernel": 3, "stride": 1},
                        {"filters": 16, "kernel": 3, "stride": 1},
                    ],
                    "pool": "global",
                },
            ],
            [64, 64],
            outputs,
            generator,
        )
    elif name == "conv-grid":
        # Same sharing as conv-shallow, but keeps a coarse spatial grid instead
        # of collapsing everything into one global OR, so the head sees
        # 4x4x32 = 512 features rather than 16.
        model = GateConvNet(
            input_size,
            [
                {"layers": [{"filters": 16, "kernel": 3, "stride": 1}], "pool": "max2"},
                {"layers": [{"filters": 32, "kernel": 3, "stride": 1}], "pool": "max2"},
            ],
            [64, 64],
            outputs,
            generator,
        )
    elif name == "conv-grid-head":
        # Fully shared: the 1x1 gate layer replaces the big dense head, so the
        # whole spatial stack is weight-shared (21k params instead of 92k).
        model = GateConvNet(
            input_size,
            [
                {"layers": [{"filters": 16, "kernel": 3, "stride": 1}], "pool": "max2"},
                {"layers": [{"filters": 32, "kernel": 3, "stride": 1}], "pool": "max2"},
                {"layers": [{"filters": 64, "kernel": 1, "stride": 1}], "pool": "global"},
            ],
            [64],
            outputs,
            generator,
        )
    elif name == "conv-deep-grid":
        model = GateConvNet(
            input_size,
            [
                {
                    "layers": [
                        {"filters": 12, "kernel": 3, "stride": 1},
                        {"filters": 12, "kernel": 3, "stride": 1},
                    ],
                    "pool": "max2",
                },
                {
                    "layers": [
                        {"filters": 24, "kernel": 3, "stride": 1},
                        {"filters": 24, "kernel": 3, "stride": 1},
                    ],
                    "pool": "max2",
                },
            ],
            [64, 64],
            outputs,
            generator,
        )
    elif name == "float-cnn":
        model = FloatCNN(10)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ste-epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ste-lr", type=float, default=0.01)
    parser.add_argument("--t-start", type=float, default=1.0)
    parser.add_argument("--t-min", type=float, default=0.25)
    parser.add_argument("--mask-constants", action="store_true")
    parser.add_argument(
        "--stacked-readout",
        choices=["", "linear", "gate"],
        default="",
        help="Retrain a combiner on the hardened forest bits.",
    )
    parser.add_argument("--stacked-hidden", default="32")
    parser.add_argument(
        "--tri-threshold",
        type=float,
        default=0.75,
        help="Hard readout threshold for ternary outputs (counts >= this as 1).",
    )
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment-rotation", type=float, default=10.0,
                        help="Max rotation in degrees.")
    parser.add_argument("--augment-scale", type=float, default=0.1)
    parser.add_argument("--augment-translate", type=float, default=0.1)
    parser.add_argument("--pin-sweep", default="")
    parser.add_argument(
        "--pin-fraction",
        type=float,
        default=0.0,
        help="Fraction of each layer's wires pinned to a round-robin cover.",
    )
    parser.add_argument(
        "--freeze-wires",
        action="store_true",
        help="Pin wires to a shuffled round-robin cover and train gates only.",
    )
    parser.add_argument(
        "--freeze-first-wires",
        action="store_true",
        help="Pin only layer 0 wires (guarantees input coverage); learn the rest.",
    )
    parser.add_argument(
        "--dense-hidden",
        default="128,64",
        help="Hidden sizes for the dense model, e.g. 256,128.",
    )
    parser.add_argument(
        "--width-sweep",
        default="",
        help="Comma-separated dense widths to sweep, e.g. 128x64,256x128.",
    )
    parser.add_argument(
        "--circuits-per-label",
        type=str,
        default="1",
        help="Forest size, or a comma-separated sweep such as 1,4,8,16.",
    )
    parser.add_argument("--score-scale", type=float, default=8.0)
    parser.add_argument("--prune-circuits", action="store_true")
    parser.add_argument("--prune-curve", action="store_true")
    parser.add_argument("--greedy-curve", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--models", default="dense,conv-shallow,conv-deep,float-cnn"
    )
    parser.add_argument("--json-out", default="")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.train = 1000
        args.test = 500
        args.epochs = 3
        args.ste_epochs = 1
        args.models = "dense,conv-shallow,float-cnn"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} device {device}")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    loaded = load_mnist(
        args.size,
        args.threshold,
        args.train,
        args.test,
        args.seed,
        device,
        augment=args.augment,
    )
    if args.augment:
        train_x, train_y, test_x, test_y, train_raw = loaded
    else:
        train_x, train_y, test_x, test_y = loaded
        train_raw = None
    print(
        f"train {len(train_x)} test {len(test_x)} "
        f"{args.size}x{args.size} binary"
    )

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    circuit_counts = [
        int(item) for item in str(args.circuits_per_label).split(",") if item.strip()
    ]
    raw_sweeps = args.width_sweep or ""
    separator = ";" if ";" in raw_sweeps else ","
    sweeps = [item.strip() for item in raw_sweeps.split(separator) if item.strip()]
    default_hidden = tuple(
        int(item) for item in str(args.dense_hidden).split(",") if item.strip()
    )
    pin_values = [
        float(item) for item in str(args.pin_sweep).split(",") if item.strip()
    ] or [args.pin_fraction]
    runs = [
        (name, circuits, widths, pin)
        for circuits in circuit_counts
        for name in models
        for widths in (sweeps or [None])
        for pin in pin_values
    ]

    results = []
    for name, circuits, widths, pin in runs:
        args.circuits_per_label = circuits
        args.pin_fraction = pin
        if widths:
            dense_hidden = tuple(int(item) for item in widths.split("x"))
        else:
            dense_hidden = default_hidden
        label = (
            f"{name} {'x'.join(str(size) for size in dense_hidden)}"
            if sweeps
            else name
        )
        if pin_values != [0.0]:
            label += f" pin={pin:g}"
        print(f"\n=== {label} (K={circuits}) ===")
        generator = torch.Generator().manual_seed(args.seed)
        model = build_model(
            name,
            args.size,
            generator,
            device,
            args.circuits_per_label,
            dense_hidden,
        )
        hard_capable = not isinstance(model, FloatCNN)
        if hard_capable:
            set_mask_constants(model, args.mask_constants)
            if args.freeze_wires:
                freeze_wires(model, generator)
            elif args.freeze_first_wires:
                freeze_wires(model, generator, first_only=True)
            elif args.pin_fraction > 0:
                pin_wires(model, generator, args.pin_fraction)
        print(
            f"params {parameter_count(model):,} gate cells "
            f"{gate_cell_count(model, args.size) if hard_capable else 0:,}"
        )
        started = time.time()
        history, before = train(
            model,
            train_x,
            train_y,
            test_x,
            test_y,
            args,
            hard_capable,
            train_raw,
        )
        soft = soft_accuracy(
            model,
            test_x,
            test_y,
            1 if not hard_capable else args.circuits_per_label,
        )
        hard = None
        zero = None
        hard_threshold = (
            args.tri_threshold if isinstance(model, GateDenseNetTri) else None
        )
        pruned = None
        active_fraction = None
        curve = None
        greedy = None
        stacked = None
        stats = None
        if hard_capable:
            hard, zero = hard_accuracy(
                model,
                test_x,
                test_y,
                args.circuits_per_label,
                threshold=hard_threshold,
            )
            stats = gate_stats(model)
            stats["denseSimplify"] = dense_simplification(model)
            stats["receptiveField"] = receptive_field(model)
            stats["lut3"] = lut3_gate_cost(model)
            if args.stacked_readout:
                stacked = stacked_readout(
                    model,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    args.stacked_readout,
                    tuple(
                        int(item)
                        for item in str(args.stacked_hidden).split(",")
                        if item.strip()
                    ),
                    generator=torch.Generator().manual_seed(args.seed + 7),
                )
            if args.prune_circuits:
                pruned, active_fraction = pruned_hard_accuracy(
                    model,
                    train_x,
                    test_x,
                    test_y,
                    args.circuits_per_label,
                )
            if args.prune_curve:
                curve = ensemble_curve(
                    model,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    args.circuits_per_label,
                )
            if args.greedy_curve:
                greedy = greedy_ensemble_curve(
                    model,
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    args.circuits_per_label,
                )
        elapsed = time.time() - started
        result = {
            "model": name,
            "denseHidden": list(dense_hidden),
            "circuitsPerLabel": circuits,
            "params": parameter_count(model),
            "gateCells": gate_cell_count(model, args.size) if hard_capable else 0,
            "softAccuracy": before["soft"] if before else soft,
            "hardBeforeSte": before["hard"] if before else None,
            "hardZeroBeforeSte": before["hardZero"] if before else None,
            "steSoftAccuracy": soft if before else None,
            "hardAccuracy": hard,
            "hardAllZeroRate": zero,
            "hardPrunedAccuracy": pruned,
            "activeCircuitFraction": active_fraction,
            "ensembleCurve": curve,
            "greedyCurve": greedy,
            "stackedReadout": stacked,
            "gateStats": stats,
            "seconds": elapsed,
            "history": history,
        }
        results.append(result)
        message = f"  final soft {result['softAccuracy'] * 100:.2f}%"
        if before:
            message += (
                f" hard-pre {before['hard'] * 100:.2f}%"
                f" (zero {before['hardZero'] * 100:.1f}%)"
            )
        if hard is not None:
            message += f" hard-post {hard * 100:.2f}% (zero {zero * 100:.1f}%)"
        if pruned is not None:
            message += (
                f"\n  prune: active circuits {active_fraction * 100:.1f}%"
                f" -> pruned hard {pruned * 100:.2f}%"
            )
        if stacked:
            target = stacked.get("hard", stacked.get("accuracy"))
            message += (
                f"\n  stacked {stacked['kind']}: {target * 100:.2f}%"
                f" (+{stacked['cells']} cells)"
            )
        if curve:
            message += "\n  ensemble: " + "  ".join(
                f"m={point['keep']} {point['accuracy'] * 100:.2f}%"
                for point in curve
            )
        if greedy:
            message += "\n  greedy: " + "  ".join(
                f"{point['circuits']} {point['accuracy'] * 100:.2f}%"
                for point in greedy
            )
        if stats is not None and stats["cells"]:
            message += (
                f"\n  gates: {stats['constant'] / stats['cells'] * 100:.1f}% constant"
                f", {stats['unaryGates'] / stats['cells'] * 100:.1f}% unary-or-constant"
                f", {stats['sameWires'] / stats['cells'] * 100:.1f}% same-wire"
            )
            if stats["convCells"]:
                message += (
                    f", conv filters reading <=1 tap:"
                    f" {stats['convSingleTap'] / stats['convCells'] * 100:.1f}%"
                )
            lut3 = stats.get("lut3")
            if lut3:
                message += (
                    f"\n  compiled: {lut3['cells']} LUT3 -> "
                    f"{lut3['compiledGates']} two-input gates"
                    f" ({lut3['meanPerCell']:.2f}/cell)"
                )
            field = stats.get("receptiveField")
            if field:
                output = field[-1]
                message += (
                    f"\n  depth {len(field)}: output bit sees mean "
                    f"{output['mean']:.1f} / max {output['max']} input bits; "
                    f"all outputs together see {output['union']}"
                )
            simplify = stats.get("denseSimplify")
            if simplify:
                message += (
                    f"\n  simplify: -{simplify['dead']} dead -{simplify['folded']} folded"
                    f" ({simplify['inverted']} inverters kept, half fan-in)"
                    f" -> {simplify['remaining']} of {simplify['cells']} cells"
                    f" ({simplify['remaining'] / simplify['cells'] * 100:.0f}%)"
                )
        print(message + f" {elapsed:.1f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nmodel           hidden     K   params   gates    soft  h-pre   h-post   prune  zero")
    for result in results:
        def pct(value):
            return f"{value * 100:6.2f}" if value is not None else "     -"

        hidden = "x".join(str(size) for size in result.get("denseHidden", []))
        print(
            f"{result['model']:<14} {hidden:<9} {result['circuitsPerLabel']:2d} "
            f"{result['params']:9,} {result['gateCells']:8,} "
            f"{pct(result['softAccuracy'])} {pct(result['hardBeforeSte'])} "
            f"{pct(result['hardAccuracy'])} {pct(result.get('hardPrunedAccuracy'))} "
            f"{pct(result['hardAllZeroRate'])}"
        )

    payload = {
        "config": vars(args),
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
