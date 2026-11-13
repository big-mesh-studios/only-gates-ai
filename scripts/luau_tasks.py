#!/usr/bin/env python3
"""Generate deterministic synthetic Luau instruction/code tasks.

The output is JSONL consumed by:

- ``scripts/train_tiny_gpt_luau_colab.py`` for supervised fine-tuning
- ``scripts/eval_luau_tasks.py`` for execution-based evaluation

Each row contains an instruction, canonical Luau code, and a small execution
harness. The harness is never shown to the model.

Design rules
------------
- Standard library only, so this runs on a laptop, Termux, or Colab.
- Deterministic for a given seed.
- Every generated answer is short enough for the 256-token model context.
- Splits are separated by seed and identifier pool, not by random row.
- Prompt styles are split: training sees ``canonical``, ``paraphrase_a``,
  ``paraphrase_c`` (imperative) and ``paraphrase_d`` (question form); the
  held-out evaluation split uses ``paraphrase_b``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAMILIES = (
    "for_print_range",
    "for_sum_all",
    "for_sum_first_n",
    "for_count_above",
    "for_find_max",
    "for_find_min",
    "for_product",
    "while_accumulate",
    "repeat_until",
    "iterate_print",
    "iterate_filter",
    "if_elseif_grade",
    "if_else_health",
    "function_add",
    "function_typed",
    "table_literal",
    "table_access",
    "string_concat",
    "string_interp",
    "nested_loop",
)

TRAIN_STYLES = ("canonical", "paraphrase_a", "paraphrase_c", "paraphrase_d")
HELD_OUT_STYLE = "paraphrase_b"


@dataclass(frozen=True)
class NamePool:
    """A set of identifiers for one split.

    The ``unseen`` pool uses names that never appear in training so
    ``test-unseen`` measures identifier copying rather than memorisation.
    """

    arrays: Tuple[str, ...]
    accumulators: Tuple[str, ...]
    loop_vars: Tuple[str, ...]
    tables: Tuple[str, ...]
    keys: Tuple[str, ...]
    values: Tuple[str, ...]
    functions: Tuple[str, ...]
    params: Tuple[str, ...]
    strings: Tuple[str, ...]

    def pick(self, rng: random.Random, field: str) -> str:
        return rng.choice(getattr(self, field))


COMMON_NAMES = NamePool(
    arrays=(
        "values", "scores", "numbers", "items", "points", "data", "nums",
        "readings", "prices", "temperatures", "distances", "weights", "counts",
        "votes", "salaries", "populations", "amounts", "entries", "records",
        "samples", "tallies", "list", "collection",
    ),
    accumulators=(
        "sum", "total", "result", "running_total", "accumulator", "aggregate",
        "tally", "grand_total", "amount", "score_total", "counter",
        "running_sum", "output", "combined", "overall",
    ),
    loop_vars=(
        "i", "idx", "index", "n", "step", "k", "j", "cursor", "position",
        "offset", "slot", "place",
    ),
    tables=(
        "items", "players", "inventory", "names", "config", "player", "scores",
        "roster", "catalog", "registry", "archive", "records", "collection",
        "group", "team", "party", "bag", "list", "store", "market",
    ),
    keys=("index", "key", "slot", "position", "id", "place", "rank", "number", "label", "tag"),
    values=("item", "value", "entry", "element", "member", "record", "object", "thing", "row", "cell"),
    functions=(
        "add", "sum", "combine", "plus", "total", "add_values", "add_numbers",
        "sum_two", "merge", "accumulate",
    ),
    params=(
        "a", "b", "x", "y", "left", "right", "first", "second", "lhs", "rhs",
        "one", "two",
    ),
    strings=(
        "Ada", "Grace", "Alan", "Edsger", "Barbara", "Linus", "Margaret",
        "Hedy", "Katherine", "Dorothy", "Mary", "Frances", "Jean", "Annie",
        "Radia", "Shafi", "Tim", "Ken", "Dennis", "Bjarne", "Guido",
    ),
)

UNSEEN_NAMES = NamePool(
    arrays=(
        "figures", "ledger", "samples", "amounts", "tallies", "entries",
        "records", "measurements", "quotas", "balances", "metrics", "series",
        "batches", "bundles", "packs", "groups", "sets", "collections",
    ),
    accumulators=(
        "tally", "aggregate", "running_sum", "grand_total", "counter",
        "combined", "overall", "final_total", "collected", "result_value",
        "sum_value", "amount_total",
    ),
    loop_vars=(
        "cursor", "position", "slot", "offset", "step_index", "walk",
        "pointer", "location", "place", "turn",
    ),
    tables=(
        "records", "roster", "catalog", "manifest", "bank", "archive",
        "registry", "directory", "index_table", "lookup", "vault", "shelf",
        "cabinet", "folder", "binder",
    ),
    keys=("slot", "position", "tag", "label", "marker", "handle", "token"),
    values=("entry", "record", "element", "member", "row", "cell", "unit"),
    functions=(
        "combine", "merge", "accumulate", "plus", "add_values", "join_values",
        "sum_pair",
    ),
    params=("first", "second", "left_value", "right_value", "lhs", "rhs", "one_value", "two_value"),
    strings=(
        "Hedy", "Katherine", "Dorothy", "Mary", "Frances", "Jean", "Annie",
        "Radia", "Shafi", "Barbara", "Margaret", "Ada", "Grace", "Alan",
    ),
)


def lua_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def lua_array(values: Sequence[object], strings: bool = False) -> str:
    items = [lua_string(str(value)) if strings else str(value) for value in values]
    return "{ " + ", ".join(items) + " }"


def stdout_lines(lines: Sequence[object]) -> str:
    return "\n".join(str(line) for line in lines)


@dataclass
class FamilyResult:
    family: str
    instructions: Dict[str, str]
    code: str
    prelude: str = ""
    check: str = ""
    expect_stdout: Optional[str] = None
    tags: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def family_for_print_range(rng: random.Random, names: NamePool) -> FamilyResult:
    var = names.pick(rng, "loop_vars")
    direction = rng.choice((1, 1, 1, -1))
    step = rng.choice((1, 1, 2, 3, -1, -2))
    if direction > 0 and step < 0:
        step = -step
    if direction < 0 and step > 0:
        step = -step
    start = rng.randint(1, 20)
    length = rng.randint(2, 7)
    finish = start + step * (length - 1)
    if step == 1:
        code = f"for {var} = {start}, {finish} do\n    print({var})\nend"
        step_phrase = ""
    else:
        code = f"for {var} = {start}, {finish}, {step} do\n    print({var})\nend"
        step_phrase = f" with a step of {step}"
    expected = stdout_lines([start + step * i for i in range(length)])
    instructions = {
        "canonical": (
            f"Write a for loop in Luau that runs from {start} to {finish}"
            f"{step_phrase} and prints every number."
        ),
        "paraphrase_a": (
            f"Show Luau code that counts from {start} to {finish}"
            f"{step_phrase} and prints each value."
        ),
        "paraphrase_c": (
            f"Write Luau code that loops from {start} to {finish}"
            f"{step_phrase} and prints every number."
        ),
        "paraphrase_d": (
            f"How do I loop from {start} to {finish}{step_phrase} in Luau "
            f"and print every number?"
        ),
        "paraphrase_b": (
            f"What Luau loop prints the values from {start} to {finish}"
            f"{step_phrase}?"
        ),
    }
    return FamilyResult(
        family="for_print_range",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"start": start, "finish": finish, "step": step},
    )


def family_for_sum_all(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    total = names.pick(rng, "accumulators")
    var = names.pick(rng, "loop_vars")
    values = [rng.randint(0, 20) for _ in range(rng.randint(3, 8))]
    code = (
        f"local {total} = 0\n"
        f"for {var} = 1, #{array} do\n"
        f"    {total} += {array}[{var}]\n"
        f"end\n"
        f"print({total})"
    )
    expected = sum(values)
    instructions = {
        "canonical": (
            f"Write a for loop that sums every element of the array `{array}` "
            f"and prints the total in Luau."
        ),
        "paraphrase_a": (
            f"Show Luau code that adds up all the numbers in `{array}` and "
            f"prints the result."
        ),
        "paraphrase_c": (
            f"Write Luau code that adds up every element of `{array}` and "
            f"prints the total."
        ),
        "paraphrase_d": (
            f"How do I add up every element of `{array}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau code totals all the values in the `{array}` table?"
        ),
    }
    return FamilyResult(
        family="for_sum_all",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"length": len(values), "expected": expected},
    )


def family_for_sum_first_n(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    total = names.pick(rng, "accumulators")
    var = names.pick(rng, "loop_vars")
    n = rng.choice((5, 10, 20, 25, 50, 50, 50))
    extra = rng.randint(0, 5)
    values = [rng.randint(0, 20) for _ in range(n + extra)]
    code = (
        f"local {total} = 0\n"
        f"for {var} = 1, {n} do\n"
        f"    {total} += {array}[{var}]\n"
        f"end\n"
        f"print({total})"
    )
    expected = sum(values[:n])
    instructions = {
        "canonical": (
            f"Write a for loop that sums the first {n} elements of the array "
            f"`{array}` and prints the result in Luau. The array has at least "
            f"{n} elements."
        ),
        "paraphrase_a": (
            f"Show Luau code that adds up the first {n} entries in `{array}` "
            f"and prints the total."
        ),
        "paraphrase_c": (
            f"Write Luau code that sums the first {n} elements of `{array}` "
            f"and prints the total."
        ),
        "paraphrase_d": (
            f"How do I sum the first {n} elements of `{array}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau loop totals the first {n} values of the `{array}` "
            f"table?"
        ),
    }
    return FamilyResult(
        family="for_sum_first_n",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"n": n, "expected": expected},
    )


def family_for_count_above(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    count = rng.choice(("count", "matches", "hits", "passing"))
    var = names.pick(rng, "loop_vars")
    threshold = rng.choice((50, 60, 70, 80, 90))
    values = [rng.randint(0, 100) for _ in range(rng.randint(5, 12))]
    code = (
        f"local {count} = 0\n"
        f"for {var} = 1, #{array} do\n"
        f"    if {array}[{var}] > {threshold} then\n"
        f"        {count} += 1\n"
        f"    end\n"
        f"end\n"
        f"print({count})"
    )
    expected = sum(1 for value in values if value > threshold)
    instructions = {
        "canonical": (
            f"Write a for loop that counts how many elements of `{array}` are "
            f"greater than {threshold}, then prints the count in Luau."
        ),
        "paraphrase_a": (
            f"Show Luau code that counts the values in `{array}` above "
            f"{threshold} and prints how many there are."
        ),
        "paraphrase_c": (
            f"Write Luau code that counts the elements of `{array}` greater "
            f"than {threshold} and prints the count."
        ),
        "paraphrase_d": (
            f"How do I count the elements in `{array}` above {threshold} in "
            f"Luau?"
        ),
        "paraphrase_b": (
            f"What Luau code counts the values in `{array}` that are over "
            f"{threshold}?"
        ),
    }
    return FamilyResult(
        family="for_count_above",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"threshold": threshold, "expected": expected},
    )


def family_for_find_max(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    best = rng.choice(("best", "largest", "maximum", "highest"))
    var = names.pick(rng, "loop_vars")
    values = [rng.randint(0, 100) for _ in range(rng.randint(3, 10))]
    code = (
        f"local {best} = {array}[1]\n"
        f"for {var} = 2, #{array} do\n"
        f"    if {array}[{var}] > {best} then\n"
        f"        {best} = {array}[{var}]\n"
        f"    end\n"
        f"end\n"
        f"print({best})"
    )
    expected = max(values)
    instructions = {
        "canonical": (
            f"Write a for loop that finds the largest value in the array "
            f"`{array}` and prints it in Luau."
        ),
        "paraphrase_a": (
            f"Show Luau code that returns the maximum element of `{array}`."
        ),
        "paraphrase_c": (
            f"Write Luau code that finds the largest value in `{array}` and "
            f"prints it."
        ),
        "paraphrase_d": (
            f"How do I find the largest value in `{array}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau loop returns the maximum value in `{array}`?"
        ),
    }
    return FamilyResult(
        family="for_find_max",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"expected": expected},
    )


def family_for_find_min(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    best = rng.choice(("best", "smallest", "minimum", "lowest"))
    var = names.pick(rng, "loop_vars")
    values = [rng.randint(0, 100) for _ in range(rng.randint(3, 10))]
    code = (
        f"local {best} = {array}[1]\n"
        f"for {var} = 2, #{array} do\n"
        f"    if {array}[{var}] < {best} then\n"
        f"        {best} = {array}[{var}]\n"
        f"    end\n"
        f"end\n"
        f"print({best})"
    )
    expected = min(values)
    instructions = {
        "canonical": (
            f"Write a for loop that finds the smallest value in the array "
            f"`{array}` and prints it in Luau."
        ),
        "paraphrase_a": (
            f"Show Luau code that returns the minimum element of `{array}`."
        ),
        "paraphrase_c": (
            f"Write Luau code that finds the smallest value in `{array}` and "
            f"prints it."
        ),
        "paraphrase_d": (
            f"How do I find the smallest value in `{array}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau loop returns the minimum value in `{array}`?"
        ),
    }
    return FamilyResult(
        family="for_find_min",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"expected": expected},
    )


def family_for_product(rng: random.Random, names: NamePool) -> FamilyResult:
    array = names.pick(rng, "arrays")
    product = rng.choice(("product", "result", "total", "answer"))
    var = names.pick(rng, "loop_vars")
    values = [rng.randint(1, 6) for _ in range(rng.randint(3, 6))]
    code = (
        f"local {product} = 1\n"
        f"for {var} = 1, #{array} do\n"
        f"    {product} *= {array}[{var}]\n"
        f"end\n"
        f"print({product})"
    )
    expected = 1
    for value in values:
        expected *= value
    instructions = {
        "canonical": (
            f"Write a for loop that multiplies every element of the array "
            f"`{array}` together and prints the product in Luau."
        ),
        "paraphrase_a": (
            f"Show Luau code that computes the product of all values in "
            f"`{array}`."
        ),
        "paraphrase_c": (
            f"Write Luau code that multiplies every element of `{array}` and "
            f"prints the product."
        ),
        "paraphrase_d": (
            f"How do I multiply every element of `{array}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau loop computes the product of the values in `{array}`?"
        ),
    }
    return FamilyResult(
        family="for_product",
        instructions=instructions,
        code=code,
        prelude=f"local {array} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"expected": expected},
    )


def family_while_accumulate(rng: random.Random, names: NamePool) -> FamilyResult:
    count = rng.choice(
        ("count", "current", "step", "counter", "tally", "cursor", "position")
    )
    total = names.pick(rng, "accumulators")
    while total == count:
        total = names.pick(rng, "accumulators")
    limit = rng.randint(3, 25)
    code = (
        f"local {count} = 0\n"
        f"local {total} = 0\n"
        f"while {count} < {limit} do\n"
        f"    {count} += 1\n"
        f"    {total} += {count}\n"
        f"end\n"
        f"print({total})"
    )
    expected = sum(range(1, limit + 1))
    instructions = {
        "canonical": (
            f"Use a while loop in Luau to add the numbers 1 through {limit} "
            f"and print the total."
        ),
        "paraphrase_a": (
            f"Show Luau code that uses a while loop to sum 1 to {limit} and "
            f"prints the result."
        ),
        "paraphrase_c": (
            f"Write Luau code that uses a while loop to add the numbers 1 "
            f"through {limit} and prints the total."
        ),
        "paraphrase_d": (
            f"How do I add the numbers 1 through {limit} with a while loop "
            f"in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau while loop sums the numbers from 1 to {limit}?"
        ),
    }
    return FamilyResult(
        family="while_accumulate",
        instructions=instructions,
        code=code,
        expect_stdout=str(expected),
        tags={"limit": limit, "expected": expected},
    )


def family_repeat_until(rng: random.Random, names: NamePool) -> FamilyResult:
    count = rng.choice(
        ("count", "current", "step", "counter", "tally", "cursor", "position")
    )
    limit = rng.randint(3, 60)
    code = (
        f"local {count} = 0\n"
        f"repeat\n"
        f"    {count} += 1\n"
        f"until {count} >= {limit}\n"
        f"print({count})"
    )
    instructions = {
        "canonical": (
            f"Use a repeat loop in Luau to count from 0 until the count "
            f"reaches {limit}, then print the count."
        ),
        "paraphrase_a": (
            f"Show Luau code that repeats until a counter reaches {limit} "
            f"and prints it."
        ),
        "paraphrase_c": (
            f"Write Luau code that uses a repeat loop to count from 0 to "
            f"{limit} and prints the count."
        ),
        "paraphrase_d": (
            f"How do I write a repeat-until loop in Luau that stops at "
            f"{limit}?"
        ),
        "paraphrase_b": (
            f"What Luau repeat loop increments a counter until it reaches "
            f"{limit}?"
        ),
    }
    return FamilyResult(
        family="repeat_until",
        instructions=instructions,
        code=code,
        expect_stdout=str(limit),
        tags={"limit": limit},
    )


def family_iterate_print(rng: random.Random, names: NamePool) -> FamilyResult:
    table = names.pick(rng, "tables")
    key = names.pick(rng, "keys")
    value = names.pick(rng, "values")
    entries = [names.pick(rng, "strings") for _ in range(rng.randint(3, 5))]
    # Keep entries unique so the expected output is unambiguous.
    entries = list(dict.fromkeys(entries))
    while len(entries) < 3:
        candidate = names.pick(rng, "strings")
        if candidate not in entries:
            entries.append(candidate)
    code = (
        f"for {key}, {value} in {table} do\n"
        f"    print({key}, {value})\n"
        f"end"
    )
    expected = stdout_lines([f"{i + 1}\t{entry}" for i, entry in enumerate(entries)])
    instructions = {
        "canonical": (
            f"Given the array `{table} = {lua_array(entries, strings=True)}`, "
            f"write a loop in Luau that prints each index and value."
        ),
        "paraphrase_a": (
            f"Show Luau code that iterates over `{table}` and prints every "
            f"index with its value."
        ),
        "paraphrase_c": (
            f"Write Luau code that loops over `{table}` and prints each "
            f"index and value."
        ),
        "paraphrase_d": (
            f"How do I print each index and value of `{table}` in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau loop prints every key and value in the `{table}` "
            f"table?"
        ),
    }
    return FamilyResult(
        family="iterate_print",
        instructions=instructions,
        code=code,
        prelude=f"local {table} = {lua_array(entries, strings=True)}",
        expect_stdout=expected,
        tags={"length": len(entries)},
    )


def family_iterate_filter(rng: random.Random, names: NamePool) -> FamilyResult:
    table = names.pick(rng, "arrays")
    filtered = rng.choice(("filtered", "kept", "selected"))
    key = names.pick(rng, "keys")
    value = names.pick(rng, "values")
    threshold = rng.choice((5, 10, 15, 20))
    values = [rng.randint(0, 30) for _ in range(rng.randint(5, 10))]
    code = (
        f"local {filtered} = {{}}\n"
        f"for {key}, {value} in {table} do\n"
        f"    if {value} > {threshold} then\n"
        f"        {filtered}[#{filtered} + 1] = {value}\n"
        f"    end\n"
        f"end\n"
        f"print(#{filtered})"
    )
    expected = sum(1 for value in values if value > threshold)
    instructions = {
        "canonical": (
            f"Given the array `{table} = {lua_array(values)}`, write a loop "
            f"that copies every value greater than {threshold} into a new "
            f"table `{filtered}`, then prints the size of `{filtered}`."
        ),
        "paraphrase_a": (
            f"Show Luau code that filters `{table}` to values above "
            f"{threshold} and prints the size of `{filtered}`."
        ),
        "paraphrase_c": (
            f"Write Luau code that copies the values in `{table}` greater "
            f"than {threshold} into `{filtered}` and prints the size of "
            f"`{filtered}`."
        ),
        "paraphrase_d": (
            f"How do I keep the values in `{table}` above {threshold} in a "
            f"new table and print how many there are?"
        ),
        "paraphrase_b": (
            f"What Luau loop filters `{table}` to values over {threshold} and "
            f"prints the size of `{filtered}`?"
        ),
    }
    return FamilyResult(
        family="iterate_filter",
        instructions=instructions,
        code=code,
        prelude=f"local {table} = {lua_array(values)}",
        expect_stdout=str(expected),
        tags={"threshold": threshold, "expected": expected},
    )


def family_if_elseif_grade(rng: random.Random, names: NamePool) -> FamilyResult:
    score_name = rng.choice(("score", "grade", "points", "mark"))
    score = rng.randint(0, 100)
    code = (
        f"if {score_name} >= 90 then\n"
        f'    print("A")\n'
        f"elseif {score_name} >= 80 then\n"
        f'    print("B")\n'
        f"else\n"
        f'    print("C")\n'
        f"end"
    )
    expected = "A" if score >= 90 else "B" if score >= 80 else "C"
    instructions = {
        "canonical": (
            f"Given `{score_name} = {score}`, write an if/elseif/else in "
            f"Luau that prints A for 90 or above, B for 80 or above, and C "
            f"otherwise."
        ),
        "paraphrase_a": (
            f"Show Luau code that turns the value {score} into a letter grade "
            f"with A at 90, B at 80, and C otherwise."
        ),
        "paraphrase_c": (
            f"Write Luau code that prints A when `{score_name}` is 90 or "
            f"above, B when it is 80 or above, and C otherwise."
        ),
        "paraphrase_d": (
            f"How do I classify `{score_name} = {score}` as A, B, or C in "
            f"Luau?"
        ),
        "paraphrase_b": (
            f"What Luau if/elseif prints A for 90 or above, B for 80 or "
            f"above, and C otherwise for `{score_name} = {score}`?"
        ),
    }
    return FamilyResult(
        family="if_elseif_grade",
        instructions=instructions,
        code=code,
        prelude=f"local {score_name} = {score}",
        expect_stdout=expected,
        tags={"score": score, "expected": expected},
    )


def family_if_else_health(rng: random.Random, names: NamePool) -> FamilyResult:
    health_name = rng.choice(("health", "hp", "life", "health_points"))
    health = rng.randint(0, 100)
    limit = rng.choice((25, 30, 50))
    code = (
        f"if {health_name} < {limit} then\n"
        f'    print("low")\n'
        f"else\n"
        f'    print("ok")\n'
        f"end"
    )
    expected = "low" if health < limit else "ok"
    instructions = {
        "canonical": (
            f"Given `{health_name} = {health}`, write an if/else in Luau that "
            f'prints "low" when it is below {limit} and "ok" otherwise.'
        ),
        "paraphrase_a": (
            f"Show Luau code that warns with low when {health_name} is under "
            f"{limit} and prints ok otherwise."
        ),
        "paraphrase_c": (
            f"Write Luau code that prints low when `{health_name}` is below "
            f"{limit} and ok otherwise."
        ),
        "paraphrase_d": (
            f"How do I check whether `{health_name} = {health}` is below "
            f"{limit} in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau if/else prints low below {limit} and ok otherwise for "
            f"`{health_name} = {health}`?"
        ),
    }
    return FamilyResult(
        family="if_else_health",
        instructions=instructions,
        code=code,
        prelude=f"local {health_name} = {health}",
        expect_stdout=expected,
        tags={"health": health, "limit": limit, "expected": expected},
    )


def family_function_add(rng: random.Random, names: NamePool) -> FamilyResult:
    # `sum`, `total`, and `accumulate` are also accumulator variable names in
    # other families. Reusing them as a function name teaches the model to
    # emit `local function sum += ...`, so keep the callable names distinct.
    function = rng.choice(
        tuple(
            name
            for name in names.functions
            if name not in ("sum", "total", "accumulate")
        )
    )
    a, b = rng.sample(names.params, 2)
    left, right = rng.randint(1, 20), rng.randint(1, 20)
    code = (
        f"local function {function}({a}, {b})\n"
        f"    return {a} + {b}\n"
        f"end\n"
        f"\n"
        f"print({function}({left}, {right}))"
    )
    instructions = {
        "canonical": (
            f"Write a local function named `{function}` in Luau that takes "
            f"`{a}` and `{b}` and returns their sum. Then print "
            f"`{function}({left}, {right})`."
        ),
        "paraphrase_a": (
            f"Show Luau code that defines a function called `{function}` to "
            f"add two numbers, then calls it with {left} and {right}."
        ),
        "paraphrase_c": (
            f"Write Luau code that defines a local function `{function}` "
            f"returning `{a} + {b}`, then prints `{function}({left}, "
            f"{right})`."
        ),
        "paraphrase_d": (
            f"How do I write a Luau function `{function}` that adds `{a}` "
            f"and `{b}`?"
        ),
        "paraphrase_b": (
            f"What Luau code defines `{function}({a}, {b})` and prints "
            f"`{function}({left}, {right})`?"
        ),
    }
    return FamilyResult(
        family="function_add",
        instructions=instructions,
        code=code,
        expect_stdout=str(left + right),
        tags={"expected": left + right},
    )


def family_function_typed(rng: random.Random, names: NamePool) -> FamilyResult:
    function = rng.choice(
        ("greet", "welcome", "hello", "introduce", "salute", "address")
    )
    param = rng.choice(("name", "who", "person", "visitor", "guest", "friend"))
    person = names.pick(rng, "strings")
    code = (
        f"local function {function}({param}: string): string\n"
        f'    return "Hello, " .. {param}\n'
        f"end\n"
        f"\n"
        f"print({function}({lua_string(person)}))"
    )
    expected = f"Hello, {person}"
    instructions = {
        "canonical": (
            f"Write a local function named `{function}` in Luau with a "
            f"`string` parameter `{param}` and a `string` return type. It "
            f'should return "Hello, " .. {param}, then print '
            f"`{function}({lua_string(person)})`."
        ),
        "paraphrase_a": (
            f"Show Luau code that defines a typed function `{function}` "
            f"taking a string and returning a greeting for {person}."
        ),
        "paraphrase_c": (
            f"Write Luau code that defines a typed function `{function}` "
            f"returning a greeting, then prints "
            f"`{function}({lua_string(person)})`."
        ),
        "paraphrase_d": (
            f"How do I add type annotations to a Luau function `{function}` "
            f"that greets `{param}`?"
        ),
        "paraphrase_b": (
            f"What Luau code declares a typed function `{function}` and "
            f"greets {person}?"
        ),
    }
    return FamilyResult(
        family="function_typed",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"expected": expected},
    )


def family_table_literal(rng: random.Random, names: NamePool) -> FamilyResult:
    table = rng.choice(("config", "profile", "settings", "record"))
    name_field = rng.choice(("name", "title", "label"))
    name_value = names.pick(rng, "strings")
    age_field = rng.choice(("age", "level", "score", "count"))
    age_value = rng.randint(1, 60)
    code = (
        f"local {table} = {{\n"
        f"    {name_field} = {lua_string(name_value)},\n"
        f"    {age_field} = {age_value},\n"
        f"}}\n"
        f"\n"
        f"print({table}.{name_field})\n"
        f"print({table}.{age_field})"
    )
    expected = stdout_lines([name_value, age_value])
    instructions = {
        "canonical": (
            f"Create a table named `{table}` in Luau with the fields "
            f"`{name_field} = {lua_string(name_value)}` and "
            f"`{age_field} = {age_value}`, then print both fields."
        ),
        "paraphrase_a": (
            f"Show Luau code that builds a `{table}` table with "
            f"{name_field} and {age_field}, then prints each field."
        ),
        "paraphrase_c": (
            f"Write Luau code that creates `{table}` with "
            f"`{name_field} = {lua_string(name_value)}` and "
            f"`{age_field} = {age_value}`, then prints both fields."
        ),
        "paraphrase_d": (
            f"How do I create a Luau table with `{name_field}` and "
            f"`{age_field}` and print both values?"
        ),
        "paraphrase_b": (
            f"What Luau code builds `{table}` with two fields and prints both "
            f"values?"
        ),
    }
    return FamilyResult(
        family="table_literal",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"expected": expected},
    )


def family_table_access(rng: random.Random, names: NamePool) -> FamilyResult:
    table = rng.choice(("player", "account", "user", "profile"))
    field = rng.choice(("score", "level", "coins", "health"))
    field_value = rng.randint(1, 50)
    new_value = rng.randint(field_value + 1, 100)
    code = (
        f"local {table} = {{\n"
        f"    {field} = {field_value},\n"
        f"}}\n"
        f"\n"
        f"{table}.{field} = {new_value}\n"
        f"print({table}.{field})"
    )
    instructions = {
        "canonical": (
            f"Create a table named `{table}` in Luau with a `{field}` field "
            f"set to {field_value}. Change `{field}` to {new_value}, then "
            f"print it."
        ),
        "paraphrase_a": (
            f"Show Luau code that updates the `{field}` field of `{table}` "
            f"from {field_value} to {new_value} and prints the new value."
        ),
        "paraphrase_c": (
            f"Write Luau code that creates `{table}` with "
            f"`{field} = {field_value}`, sets `{field}` to {new_value}, then "
            f"prints it."
        ),
        "paraphrase_d": (
            f"How do I change `{table}.{field}` to {new_value} in Luau and "
            f"print it?"
        ),
        "paraphrase_b": (
            f"What Luau code updates `{table}.{field}` and prints the new "
            f"value?"
        ),
    }
    return FamilyResult(
        family="table_access",
        instructions=instructions,
        code=code,
        expect_stdout=str(new_value),
        tags={"expected": new_value},
    )


def family_string_concat(rng: random.Random, names: NamePool) -> FamilyResult:
    first_name = names.pick(rng, "strings")
    last_name = rng.choice(("Lovelace", "Hopper", "Turing", "Hamilton", "Johnson", "Clarke"))
    first_var = rng.choice(("first", "first_name", "given", "forename"))
    last_var = rng.choice(("last", "last_name", "family", "surname"))
    full_var = rng.choice(("full", "full_name", "display", "label"))
    code = (
        f"local {first_var} = {lua_string(first_name)}\n"
        f"local {last_var} = {lua_string(last_name)}\n"
        f"local {full_var} = {first_var} .. \" \" .. {last_var}\n"
        f"print({full_var})"
    )
    expected = f"{first_name} {last_name}"
    instructions = {
        "canonical": (
            f"Using the variables `{first_var} = {lua_string(first_name)}` "
            f"and `{last_var} = {lua_string(last_name)}`, write Luau code "
            f"that joins them with a space and prints the full name."
        ),
        "paraphrase_a": (
            f"Show Luau code that concatenates {first_name} and {last_name} "
            f"with a space and prints the result."
        ),
        "paraphrase_c": (
            f"Write Luau code that joins `{first_var}` and `{last_var}` with "
            f"a space and prints the full name."
        ),
        "paraphrase_d": (
            f"How do I join `{first_var}` and `{last_var}` into a full name "
            f"in Luau?"
        ),
        "paraphrase_b": (
            f"What Luau code concatenates `{first_var}` and `{last_var}` with "
            f"a space?"
        ),
    }
    return FamilyResult(
        family="string_concat",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"expected": expected},
    )


def family_string_interp(rng: random.Random, names: NamePool) -> FamilyResult:
    person = names.pick(rng, "strings")
    var = rng.choice(("name", "who", "person", "visitor", "guest"))
    greeting = rng.choice(("Hello", "Hi", "Hey", "Greetings", "Welcome"))
    punct = rng.choice(("!", ".", "!!"))
    text = f"{greeting}, {person}{punct}"
    code = (
        f"local {var} = {lua_string(person)}\n"
        f"local message = `{greeting}, {{{var}}}{punct}`\n"
        f"print(message)"
    )
    expected = text
    instructions = {
        "canonical": (
            f"Using `{var} = {lua_string(person)}`, write Luau code that uses "
            f"string interpolation to build and print `{text}`."
        ),
        "paraphrase_a": (
            f"Show Luau code that uses an interpolated string to greet "
            f"{person} with \"{greeting}\"."
        ),
        "paraphrase_c": (
            f"Write Luau code that uses string interpolation to build and "
            f"print `{text}`."
        ),
        "paraphrase_d": (
            f"How do I use Luau string interpolation to print a greeting "
            f"for `{var}`?"
        ),
        "paraphrase_b": (
            f"What Luau code interpolates `{var}` into a greeting?"
        ),
    }
    return FamilyResult(
        family="string_interp",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"expected": expected},
    )


def family_nested_loop(rng: random.Random, names: NamePool) -> FamilyResult:
    outer = names.pick(rng, "loop_vars")
    inner = rng.choice(("j", "k", "inner", "column", "col"))
    while inner == outer:
        inner = rng.choice(("j", "k", "inner", "column", "col"))
    outer_n = rng.randint(2, 5)
    inner_n = rng.randint(2, 4)
    code = (
        f"for {outer} = 1, {outer_n} do\n"
        f"    for {inner} = 1, {inner_n} do\n"
        f"        print({outer}, {inner})\n"
        f"    end\n"
        f"end"
    )
    expected = stdout_lines(
        [f"{i}\t{j}" for i in range(1, outer_n + 1) for j in range(1, inner_n + 1)]
    )
    instructions = {
        "canonical": (
            f"Write a nested for loop in Luau where `{outer}` runs from 1 to "
            f"{outer_n} and `{inner}` runs from 1 to {inner_n}. Print both "
            f"variables in the inner loop."
        ),
        "paraphrase_a": (
            f"Show Luau code with an outer loop of {outer_n} and an inner "
            f"loop of {inner_n} that prints every pair."
        ),
        "paraphrase_c": (
            f"Write Luau code with an outer loop from 1 to {outer_n} and an "
            f"inner loop from 1 to {inner_n} that prints both indexes."
        ),
        "paraphrase_d": (
            f"How do I write two nested Luau loops with bounds {outer_n} and "
            f"{inner_n} and print both indexes?"
        ),
        "paraphrase_b": (
            f"What Luau code nests a {outer_n}-iteration loop inside a "
            f"{inner_n}-iteration loop and prints both indexes?"
        ),
    }
    return FamilyResult(
        family="nested_loop",
        instructions=instructions,
        code=code,
        expect_stdout=expected,
        tags={"outer": outer_n, "inner": inner_n},
    )


FAMILY_BUILDERS: Dict[str, Callable[[random.Random, NamePool], FamilyResult]] = {
    "for_print_range": family_for_print_range,
    "for_sum_all": family_for_sum_all,
    "for_sum_first_n": family_for_sum_first_n,
    "for_count_above": family_for_count_above,
    "for_find_max": family_for_find_max,
    "for_find_min": family_for_find_min,
    "for_product": family_for_product,
    "while_accumulate": family_while_accumulate,
    "repeat_until": family_repeat_until,
    "iterate_print": family_iterate_print,
    "iterate_filter": family_iterate_filter,
    "if_elseif_grade": family_if_elseif_grade,
    "if_else_health": family_if_else_health,
    "function_add": family_function_add,
    "function_typed": family_function_typed,
    "table_literal": family_table_literal,
    "table_access": family_table_access,
    "string_concat": family_string_concat,
    "string_interp": family_string_interp,
    "nested_loop": family_nested_loop,
}


# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------


def make_row(
    family: str,
    result: FamilyResult,
    style: str,
    split: str,
    index: int,
) -> Dict[str, object]:
    instruction = result.instructions[style]
    digest = hashlib.sha256(
        f"{family}\0{style}\0{instruction}\0{result.code}".encode("utf-8")
    ).hexdigest()[:12]
    harness: Dict[str, str] = {}
    if result.prelude:
        harness["prelude"] = result.prelude
    if result.check:
        harness["check"] = result.check
    if result.expect_stdout is not None:
        harness["expect_stdout"] = result.expect_stdout
    return {
        "id": f"{family}/{split}/{index:06d}-{digest}",
        "family": family,
        "mode": "write",
        "prompt_style": style,
        "instruction": instruction,
        "code": result.code,
        "harness": harness,
        "split": split,
        "tags": result.tags,
    }


def generate_split(
    families: Sequence[str],
    count: int,
    seed: int,
    split: str,
    styles: Sequence[str],
    names: NamePool,
) -> List[Dict[str, object]]:
    if count <= 0 or not families:
        return []
    seen: set[Tuple[str, str]] = set()
    rows: List[Dict[str, object]] = []
    per_family = max(1, (count + len(families) - 1) // len(families))
    index = 0
    for family_index, family in enumerate(families):
        builder = FAMILY_BUILDERS[family]
        produced = 0
        attempt = 0
        while produced < per_family and attempt < per_family * 50:
            attempt += 1
            # Derive an independent stream per family, split, and attempt.
            family_seed = (
                (seed * 1_000_003)
                + (family_index * 97_409)
                + (sum(ord(c) for c in split) * 7_919)
                + attempt
            )
            rng = random.Random(family_seed)
            result = builder(rng, names)
            # Try every prompt style before discarding a sample: some small
            # families saturate one style (only a handful of distinct
            # instructions) well before the others.
            chosen: Optional[Tuple[str, str]] = None
            # Start from the style this family is due for, so the split gets an
            # even mix. Using ``attempt`` here would advance by two residues per
            # successful row and collapse onto one or two styles.
            for offset in range(len(styles)):
                style = styles[(produced + offset) % len(styles)]
                instruction = result.instructions[style]
                key = (instruction, result.code)
                if key not in seen:
                    chosen = (style, instruction)
                    break
            if chosen is None:
                continue
            style, _ = chosen
            seen.add((result.instructions[style], result.code))
            rows.append(make_row(family, result, style, split, index))
            index += 1
            produced += 1
        if produced < per_family:
            raise RuntimeError(
                f"could only generate {produced}/{per_family} unique rows for {family}"
            )
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_readme(path: Path, counts: Dict[str, int], families: Sequence[str]) -> None:
    lines = [
        "# Luau task dataset",
        "",
        "Generated by `scripts/luau_tasks.py`. Do not edit these files by hand.",
        "",
        "Each line is a JSON object with an instruction, canonical Luau code,",
        "and an execution harness. The harness is for evaluation only.",
        "",
        "## Counts",
        "",
    ]
    for split, count in counts.items():
        lines.append(f"- `{split}.jsonl`: {count:,} rows")
    lines += [
        "",
        "## Families",
        "",
        *[f"- `{family}`" for family in families],
        "",
        "## Splits",
        "",
        "- `train.jsonl`: canonical, `paraphrase_a`, `paraphrase_c`, and "
        "`paraphrase_d` instructions.",
        "- `val.jsonl`: same distribution as training.",
        "- `test-seen.jsonl`: canonical prompts with unseen values.",
        "- `test-paraphrase.jsonl`: held-out `paraphrase_b` wording.",
        "- `test-question.jsonl`: `paraphrase_d` question prompts with unseen values.",
        "- `test-unseen.jsonl`: canonical prompts with a held-out identifier pool.",
        "",
        "Regenerate with:",
        "",
        "```sh",
        "python3 scripts/luau_tasks.py --out data/luau-tasks \\",
        "  --train 12000 --val 1200 --test-seen 300 \\",
        "  --test-paraphrase 200 --test-question 200 --test-unseen 100 --seed 1337",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=Path("data/luau-tasks"))
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--val", type=int, default=1200)
    parser.add_argument("--test-seen", type=int, default=300)
    parser.add_argument("--test-paraphrase", type=int, default=200)
    parser.add_argument("--test-question", type=int, default=200)
    parser.add_argument("--test-unseen", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--family",
        action="append",
        choices=FAMILIES,
        help="Generate only this family (repeatable). Default: all.",
    )
    parser.add_argument("--list-families", action="store_true")
    args = parser.parse_args()

    if args.list_families:
        for family in FAMILIES:
            print(family)
        return

    families = tuple(args.family) if args.family else FAMILIES
    if args.train < 1:
        raise SystemExit("--train must be at least 1")

    counts: Dict[str, int] = {}
    counts["train"] = write_jsonl(
        args.out / "train.jsonl",
        generate_split(
            families, args.train, args.seed, "train", TRAIN_STYLES, COMMON_NAMES
        ),
    )
    counts["val"] = write_jsonl(
        args.out / "val.jsonl",
        generate_split(
            families, args.val, args.seed + 1, "val", TRAIN_STYLES, COMMON_NAMES
        ),
    )
    counts["test-seen"] = write_jsonl(
        args.out / "test-seen.jsonl",
        generate_split(
            families,
            args.test_seen,
            args.seed + 2,
            "test-seen",
            ("canonical",),
            COMMON_NAMES,
        ),
    )
    counts["test-paraphrase"] = write_jsonl(
        args.out / "test-paraphrase.jsonl",
        generate_split(
            families,
            args.test_paraphrase,
            args.seed + 3,
            "test-paraphrase",
            (HELD_OUT_STYLE,),
            COMMON_NAMES,
        ),
    )
    counts["test-question"] = write_jsonl(
        args.out / "test-question.jsonl",
        generate_split(
            families,
            args.test_question,
            args.seed + 5,
            "test-question",
            ("paraphrase_d",),
            COMMON_NAMES,
        ),
    )
    counts["test-unseen"] = write_jsonl(
        args.out / "test-unseen.jsonl",
        generate_split(
            families,
            args.test_unseen,
            args.seed + 4,
            "test-unseen",
            ("canonical",),
            UNSEEN_NAMES,
        ),
    )
    write_readme(args.out / "README.md", counts, families)

    print(f"wrote {args.out}/")
    for split, count in counts.items():
        print(f"  {split + '.jsonl':24s} {count:>7,} rows")
    print(f"  families                 {len(families):>7,}")


if __name__ == "__main__":
    main()
