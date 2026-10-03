"""pass@1, consistency and disagreement metrics.

All metrics work on an ``Outcomes`` mapping:
``{precision: {(task_id, variant): passed}}``. With greedy decoding there is one
sample per prompt, so pass@1 is simply the fraction of problems that pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from codecons.perturbations import ORIGINAL

Outcomes = Dict[str, Dict[Tuple[str, str], bool]]


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def ci(self) -> Tuple[float, float]:
        return wilson_interval(self.k, self.n)


@dataclass(frozen=True)
class Consistency:
    rate: Rate
    pass_to_fail: List[str]  # passes on the original prompt, fails after the change
    fail_to_pass: List[str]


@dataclass(frozen=True)
class PairedComparison:
    """Consistency of a quantized model vs. the reference on the same (problem, change) pairs."""

    n: int
    ref_only_consistent: int
    other_only_consistent: int
    p_value: float


def load_outcomes(raw_dir: str) -> Outcomes:
    outcomes: Outcomes = {}
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                outcomes.setdefault(r["precision"], {})[(r["task_id"], r["variant"])] = r["passed"]
    return outcomes


def pass_at_1(outcomes: Outcomes, precision: str, variant: str) -> Rate:
    results = [ok for (_, v), ok in outcomes[precision].items() if v == variant]
    return Rate(sum(results), len(results))


def consistency(outcomes: Outcomes, precision: str, perturbation: str) -> Consistency:
    """Share of problems whose pass/fail result is unchanged by the perturbation."""
    res = outcomes[precision]
    same, p2f, f2p = 0, [], []
    for task_id in _tasks_with(res, perturbation):
        before, after = res[(task_id, ORIGINAL)], res[(task_id, perturbation)]
        if before == after:
            same += 1
        elif before:
            p2f.append(task_id)
        else:
            f2p.append(task_id)
    return Consistency(Rate(same, same + len(p2f) + len(f2p)), p2f, f2p)


def pooled_consistency(outcomes: Outcomes, precision: str, perturbations) -> Rate:
    rates = [consistency(outcomes, precision, p).rate for p in perturbations]
    return Rate(sum(r.k for r in rates), sum(r.n for r in rates))


def disagreement(outcomes: Outcomes, ref: str, other: str, variant: str) -> List[str]:
    """Problems where the reference model passes and the other model fails on the same prompt."""
    a, b = outcomes[ref], outcomes[other]
    return sorted(t for (t, v), ok in a.items() if v == variant and ok and b.get((t, v)) is False)


def change_induced_failures(outcomes: Outcomes, ref: str, other: str, perturbation: str) -> List[str]:
    """Problems where the reference passes before and after the change, the other model
    passes the original prompt, but fails once the prompt is changed."""
    a, b = outcomes[ref], outcomes[other]
    found = []
    for t in _tasks_with(a, perturbation):
        if (t, perturbation) not in b:
            continue
        if a[(t, ORIGINAL)] and a[(t, perturbation)] and b[(t, ORIGINAL)] and not b[(t, perturbation)]:
            found.append(t)
    return sorted(found)


def compare_consistency(outcomes: Outcomes, ref: str, other: str, perturbations) -> PairedComparison:
    """Exact McNemar test on (problem, perturbation) pairs: is the other model less consistent?

    Pairs from the same problem are not independent, so treat the p-value as indicative.
    """
    a, b = outcomes[ref], outcomes[other]
    n = ref_only = other_only = 0
    for p in perturbations:
        for t in _tasks_with(a, p):
            if (t, p) not in b or (t, ORIGINAL) not in b:
                continue
            n += 1
            ca = a[(t, ORIGINAL)] == a[(t, p)]
            cb = b[(t, ORIGINAL)] == b[(t, p)]
            ref_only += ca and not cb
            other_only += cb and not ca
    return PairedComparison(n, ref_only, other_only, mcnemar_exact(ref_only, other_only))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _tasks_with(res: Dict[Tuple[str, str], bool], variant: str) -> List[str]:
    return sorted(t for (t, v) in res if v == variant and (t, ORIGINAL) in res)

