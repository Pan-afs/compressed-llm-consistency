"""Running candidate solutions against HumanEval's unit tests.

Every program runs in a fresh Python subprocess with a timeout, inside a
temporary working directory and with stdin closed. This is process isolation,
not a security sandbox: only run model-generated code in a disposable
environment such as Google Colab.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

PASSED, FAILED, ERROR, TIMEOUT = "passed", "failed", "error", "timeout"


@dataclass(frozen=True)
class ExecResult:
    status: str

    @property
    def passed(self) -> bool:
        return self.status == PASSED


def build_program(solution: str, test: str, entry_point: str) -> str:
    """Append the HumanEval ``check`` function and call it on the entry point."""
    return f"{solution}\n\n{test}\n\ncheck({entry_point})\n"


def run_program(program: str, timeout: float = 10.0) -> ExecResult:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(TIMEOUT)
    if proc.returncode == 0:
        return ExecResult(PASSED)
    if b"AssertionError" in proc.stderr:
        return ExecResult(FAILED)
    return ExecResult(ERROR)


def run_many(programs: Sequence[str], timeout: float = 10.0, workers: int = 8) -> List[ExecResult]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: run_program(p, timeout), programs))
