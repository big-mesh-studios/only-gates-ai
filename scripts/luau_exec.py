#!/usr/bin/env python3
"""Small, dependency-free helpers for running generated Luau snippets.

Used by ``scripts/check_luau_tasks.py`` (dataset validation) and
``scripts/eval_luau_tasks.py`` (model evaluation).

Safety: callers must pass a real Luau binary and a short timeout. The helpers
only create temporary files; they do not grant network or filesystem access
beyond the temporary working directory.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    syntax_error: bool
    expected_stdout: Optional[str] = None
    reason: str = ""


def strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence if the model added one."""
    value = text.strip()
    match = re.match(r"^```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```$", value, re.DOTALL)
    if match:
        return match.group(1).strip()
    return value


def normalize_stdout(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def build_script(code: str, harness: Mapping[str, str]) -> str:
    parts = []
    prelude = str(harness.get("prelude", "")).strip()
    if prelude:
        parts.append(prelude)
    parts.append(code.strip())
    check = str(harness.get("check", "")).strip()
    if check:
        parts.append(check)
    return "\n".join(parts) + "\n"


def execute_luau(
    code: str,
    harness: Mapping[str, str],
    luau_bin: str,
    *,
    timeout: float = 2.0,
) -> ExecResult:
    """Run one snippet and compare its output with the harness."""
    expected = harness.get("expect_stdout")
    expected_text = str(expected) if expected is not None else None
    script = build_script(code, harness)
    with tempfile.TemporaryDirectory(prefix="luau-task-") as directory:
        path = Path(directory) / "task.luau"
        path.write_text(script, encoding="utf-8")
        try:
            completed = subprocess.run(
                [luau_bin, str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return ExecResult(
                ok=False,
                stdout="",
                stderr=f"luau binary not found: {luau_bin}",
                returncode=-1,
                timed_out=False,
                syntax_error=False,
                expected_stdout=expected_text,
                reason="missing-binary",
            )
        except subprocess.TimeoutExpired as error:
            return ExecResult(
                ok=False,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                returncode=-1,
                timed_out=True,
                syntax_error=False,
                expected_stdout=expected_text,
                reason="timeout",
            )

    stdout = normalize_stdout(completed.stdout or "")
    stderr = normalize_stdout(completed.stderr or "")
    syntax_error = "SyntaxError" in stderr or "CompileError" in stderr
    ok = completed.returncode == 0 and not syntax_error
    reason = ""
    if not ok:
        reason = "syntax" if syntax_error else "runtime"
    elif expected_text is not None and stdout != normalize_stdout(expected_text):
        ok = False
        reason = "stdout"
    return ExecResult(
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
        timed_out=False,
        syntax_error=syntax_error,
        expected_stdout=expected_text,
        reason=reason,
    )


def analyze_luau(code: str, analyze_bin: str, *, timeout: float = 5.0) -> Dict[str, object]:
    """Optional syntax/type check with ``luau-analyze``."""
    with tempfile.TemporaryDirectory(prefix="luau-analyze-") as directory:
        path = Path(directory) / "task.luau"
        path.write_text(code.strip() + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [analyze_bin, str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return {"ok": False, "error": f"luau-analyze not found: {analyze_bin}"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": normalize_stdout(completed.stdout or ""),
        "stderr": normalize_stdout(completed.stderr or ""),
    }
