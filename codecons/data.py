"""Loading the HumanEval benchmark."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

HF_DATASET = "openai/openai_humaneval"
_FIELDS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval(path: Optional[str] = None, limit: Optional[int] = None) -> List[Problem]:
    """Load HumanEval from a local ``.jsonl``/``.jsonl.gz`` file or from the Hugging Face Hub."""
    rows: Iterable[dict]
    if path:
        rows = _read_jsonl(Path(path))
    else:
        from datasets import load_dataset

        rows = load_dataset(HF_DATASET, split="test")
    problems = [Problem(**{k: row[k] for k in _FIELDS}) for row in rows]
    return problems[:limit] if limit else problems


def _read_jsonl(path: Path) -> List[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
