"""Generate and evaluate HumanEval solutions for every prompt variant and precision.

    python -m codecons.run --precision fp16 int8 int4

Results are appended to ``<out>/<precision>.jsonl`` after every batch, so an
interrupted run (e.g. a Colab disconnect) resumes where it stopped.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import List, Optional, Set, Tuple

from codecons.data import Problem, load_humaneval
from codecons.execution import build_program, run_many
from codecons.generation import assemble_solution, generate
from codecons.models import DEFAULT_MODEL, PRECISIONS, load_model
from codecons.perturbations import Variant, make_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--precision", nargs="+", choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", help="local HumanEval jsonl(.gz); defaults to the HF Hub")
    parser.add_argument("--limit", type=int, help="only use the first N problems (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per test run")
    parser.add_argument("--out", default="results/raw")
    args = parser.parse_args()

    problems = load_humaneval(args.data, args.limit)
    jobs = [(p, v) for p in problems for v in make_variants(p.prompt, p.entry_point) if v.applicable]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for precision in args.precision:
        run_precision(precision, jobs, out_dir, args)


def run_precision(precision: str, jobs: List[Tuple[Problem, Variant]], out_dir: Path, args) -> None:
    import torch

    path = out_dir / f"{precision}.jsonl"
    done = _completed(path, args.model)
    todo = [(p, v) for p, v in jobs if (p.task_id, v.name) not in done]
    print(f"[{precision}] {len(jobs) - len(todo)} done, {len(todo)} to go")
    if not todo:
        return

    torch.manual_seed(0)
    model, tokenizer = load_model(args.model, precision)
    footprint = model.get_memory_footprint()
    print(f"[{precision}] model footprint: {footprint / 2**20:.0f} MiB")

    start = time.time()
    with path.open("a", encoding="utf-8") as f:
        prompts = [v.text for _, v in todo]
        finished = 0
        for batch in generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens):
            programs = []
            for i, response in batch:
                problem, variant = todo[i]
                solution = assemble_solution(variant.text, response, problem.entry_point)
                programs.append(build_program(solution, problem.test, problem.entry_point))
            results = run_many(programs, timeout=args.timeout)
            for (i, response), result in zip(batch, results):
                problem, variant = todo[i]
                record = {
                    "task_id": problem.task_id,
                    "variant": variant.name,
                    "precision": precision,
                    "model": args.model,
                    "prompt": variant.text,
                    "response": response,
                    "status": result.status,
                    "passed": result.passed,
                }
                f.write(json.dumps(record) + "\n")
            f.flush()
            finished += len(batch)
            print(f"[{precision}] {finished}/{len(todo)} ({time.time() - start:.0f}s)", flush=True)

    _write_meta(out_dir / f"{precision}.meta.json", args, precision, footprint, time.time() - start)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _completed(path: Path, model_id: str) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["model"] != model_id:
                raise SystemExit(f"{path} holds results for {record['model']}; use a different --out")
            done.add((record["task_id"], record["variant"]))
    return done


def _write_meta(path: Path, args, precision: str, footprint: int, seconds: float) -> None:
    import torch

    meta = {
        "model": args.model,
        "precision": precision,
        "memory_footprint_mib": round(footprint / 2**20, 1),
        "generation_seconds": round(seconds),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            "python": platform.python_version(),
            **{pkg: _version(pkg) for pkg in ("torch", "transformers", "accelerate", "bitsandbytes")},
        },
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
