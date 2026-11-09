#!/usr/bin/env python3
"""CPU resume tests for train_tiny_gpt_solid2_colab.py.

Run with:
  python scripts/test_trainer_resume.py

The trainer restores model, optimizer, scheduler and RNG state. On CPU with a
fixed seed a split run should therefore match an uninterrupted run closely.
These tests are intentionally small; they check the resume mechanics, not model
quality.
"""

from __future__ import annotations

import importlib.util
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).with_name("train_tiny_gpt_solid2_colab.py")
spec = importlib.util.spec_from_file_location("tiny_gpt_trainer", SCRIPT)
trainer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trainer)

VOCAB, CTX, DIM, LAYERS, HEADS, MLP = 64, 16, 16, 1, 2, 32
BATCH = 4
LR = 1e-3
WARMUP = 2


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model() -> torch.nn.Module:
    return trainer.TinyGPT(VOCAB, CTX, DIM, LAYERS, HEADS, MLP, 0.0)


def make_optimizer(model, total_steps: int):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: trainer.lr_lambda(s, total_steps, WARMUP))
    return opt, sch


def assert_close(a: torch.nn.Module, b: torch.nn.Module, label: str) -> None:
    left = a.state_dict()
    right = b.state_dict()
    for key in left:
        if not torch.allclose(left[key], right[key], atol=1e-6, rtol=0.0):
            delta = (left[key] - right[key]).abs().max().item()
            raise AssertionError(f"{label}: {key} differs (max {delta:.3e})")


def lm_tokens(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(VOCAB) for _ in range(count)]


def checkpoint_payload(model, opt, sch, step: int, loss_sum: float, loss_count: int) -> dict:
    return {
        "format": trainer.CHECKPOINT_FORMAT,
        "version": trainer.CHECKPOINT_VERSION,
        "stage": "lm",
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "rng": trainer.capture_rng(),
        "lm_loss_sum": loss_sum,
        "lm_loss_count": loss_count,
    }


def test_lm_resume(tmp: Path) -> None:
    tokens = lm_tokens(240, 11)
    steps = 10

    seed_all(7)
    reference = make_model()
    opt, sch = make_optimizer(reference, steps)
    trainer.train_lm(reference, tokens, steps, CTX, BATCH, LR, WARMUP, 0.01, 1.0,
                     torch.device("cpu"), "ref", opt, sch)

    seed_all(7)
    split = make_model()
    opt, sch = make_optimizer(split, steps)
    calls = {"n": 0}
    saved = {}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= 5

    def save(done: int, loss_sum: float, loss_count: int) -> None:
        saved["step"] = done
        saved["loss_sum"] = loss_sum
        saved["loss_count"] = loss_count
        trainer.atomic_torch_save(tmp / "latest.pt", checkpoint_payload(split, opt, sch, done, loss_sum, loss_count))

    loss, finished, _, _, _ = trainer.train_lm(
        split, tokens, steps, CTX, BATCH, LR, WARMUP, 0.01, 1.0,
        torch.device("cpu"), "split", opt, sch,
        save_every=0, on_save=save, should_stop=stop)
    assert not finished, "split LM run should stop early"
    assert saved["step"] == 5, f"expected save at step 5, got {saved['step']}"

    state = torch.load(tmp / "latest.pt", map_location="cpu", weights_only=False)
    resumed = make_model()
    resumed.load_state_dict(state["model"])
    opt2, sch2 = make_optimizer(resumed, steps)
    opt2.load_state_dict(state["optimizer"])
    sch2.load_state_dict(state["scheduler"])
    trainer.restore_rng(state["rng"])
    trainer.train_lm(
        resumed, tokens, steps, CTX, BATCH, LR, WARMUP, 0.01, 1.0,
        torch.device("cpu"), "resumed", opt2, sch2,
        start_step=state["step"], loss_sum=state["lm_loss_sum"],
        loss_count=state["lm_loss_count"])

    assert_close(reference, resumed, "LM resume")
    print("ok: LM split run matches uninterrupted run")


def qa_data(count: int, seed: int):
    rng = random.Random(seed)
    pairs = []
    for _ in range(count):
        seq = [rng.randrange(VOCAB) for _ in range(CTX)]
        labels = list(seq)
        labels[: CTX // 2] = [-100] * (CTX // 2)
        pairs.append((seq, labels))
    return pairs


def test_qa_resume(tmp: Path) -> None:
    train_data = qa_data(8, 21)
    val_data = qa_data(2, 22)
    epochs = 2

    seed_all(9)
    reference = make_model()
    total = epochs * ((len(train_data) + BATCH - 1) // BATCH)
    opt, sch = make_optimizer(reference, total)
    (tmp / "ref").mkdir(exist_ok=True)
    trainer.train_qa(reference, train_data, val_data, epochs, BATCH, LR, WARMUP, 0.01, 1.0,
                     torch.device("cpu"), tmp / "ref", opt, sch)

    seed_all(9)
    split = make_model()
    opt, sch = make_optimizer(split, total)
    partial_dir = tmp / "split"
    partial_dir.mkdir(exist_ok=True)
    calls = {"n": 0}
    saved = {}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    def save(epoch, cursor, perm, best, best_ep, step):
        saved.update(epoch=epoch, cursor=cursor, perm=perm, best=best, best_ep=best_ep, step=step)
        payload = {
            "format": trainer.CHECKPOINT_FORMAT,
            "version": trainer.CHECKPOINT_VERSION,
            "stage": "qa",
            "model": split.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "rng": trainer.capture_rng(),
            "qa_epoch": epoch,
            "qa_cursor": cursor,
            "qa_perm": perm,
            "qa_best": best if np.isfinite(best) else None,
            "qa_best_epoch": best_ep,
            "qa_step": step,
        }
        trainer.atomic_torch_save(tmp / "qa-latest.pt", payload)

    best, best_ep, finished, step = trainer.train_qa(
        split, train_data, val_data, epochs, BATCH, LR, WARMUP, 0.01, 1.0,
        torch.device("cpu"), partial_dir, opt, sch,
        save_every=1, on_save=save, should_stop=stop)
    assert not finished, "split QA run should stop early"
    assert saved["cursor"] and saved["perm"], "mid-epoch save should record the permutation and cursor"

    state = torch.load(tmp / "qa-latest.pt", map_location="cpu", weights_only=False)
    resumed = make_model()
    resumed.load_state_dict(state["model"])
    opt2, sch2 = make_optimizer(resumed, total)
    opt2.load_state_dict(state["optimizer"])
    sch2.load_state_dict(state["scheduler"])
    trainer.restore_rng(state["rng"])
    (tmp / "resumed").mkdir(exist_ok=True)
    trainer.train_qa(
        resumed, train_data, val_data, epochs, BATCH, LR, WARMUP, 0.01, 1.0,
        torch.device("cpu"), tmp / "resumed", opt2, sch2,
        start_epoch=state["qa_epoch"], start_cursor=state["qa_cursor"],
        start_perm=state["qa_perm"], best=state.get("qa_best") or float("inf"),
        best_ep=state["qa_best_epoch"], step=state["qa_step"])

    assert_close(reference, resumed, "QA resume")
    print("ok: QA mid-epoch split run matches uninterrupted run")


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        test_lm_resume(tmp)
        test_qa_resume(tmp)
    print("all resume tests passed")


if __name__ == "__main__":
    main()
