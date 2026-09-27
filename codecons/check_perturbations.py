"""Sanity check: every perturbation must preserve the reference solution's correctness.

Each perturbation is applied to ``prompt + canonical_solution`` and the result
is run against the HumanEval tests. If a perturbation were not
semantics-preserving, some reference solutions would start failing.

    python -m codecons.check_perturbations [--data HumanEval.jsonl.gz]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from codecons.data import load_humaneval
from codecons.execution import build_program, run_many
from codecons.perturbations import VARIANTS, ORIGINAL, Variant, apply


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="local HumanEval jsonl(.gz); defaults to the HF Hub")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    problems = load_humaneval(args.data)
    jobs = []  # (variant name, task id, applicable on prompt alone, program)
    for p in problems:
        source = p.prompt + p.canonical_solution
        for name in VARIANTS:
            if name == ORIGINAL:
                variant = Variant(ORIGINAL, source, True)
                prompt_applicable = True
            else:
                variant = apply(name, source, p.entry_point)
                prompt_applicable = apply(name, p.prompt, p.entry_point).applicable
            if variant.applicable:
                jobs.append((name, p.task_id, prompt_applicable, build_program(variant.text, p.test, p.entry_point)))

    results = run_many([j[3] for j in jobs], workers=args.workers)

    applicable = defaultdict(int)
    passed = defaultdict(int)
    failures = defaultdict(list)
    for (name, task_id, prompt_applicable, _), result in zip(jobs, results):
        if not prompt_applicable:
            continue
        applicable[name] += 1
        if result.passed:
            passed[name] += 1
        else:
            failures[name].append(f"{task_id} ({result.status})")

    print(f"{'variant':<10} {'applicable':>10} {'reference passes':>17}")
    for name in VARIANTS:
        print(f"{name:<10} {applicable[name]:>10} {passed[name]:>17}")
    for name, items in failures.items():
        print(f"\n{name} breaks: {', '.join(items)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
