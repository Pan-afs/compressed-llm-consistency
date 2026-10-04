# Do compressed code LLMs stay consistent?

When you quantize a code LLM to make it cheaper, does it start giving inconsistent answers when the prompt
changes in ways that shouldn't matter?

Compression work usually reports accuracy (pass@1) and stops there. Consistency testing asks a different
question: does the model give the same answer to two prompts that mean the same thing? This project puts
the two together. It quantizes a small code model to 8 and 4 bits, then measures both accuracy and how
often a meaning-preserving prompt change flips a problem from pass to fail, or from fail to pass.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pan-afs/compressed-llm-consistency/blob/main/notebooks/run_on_colab.ipynb)

## Setup

| | |
| --- | --- |
| Model | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) (the 0.5B model works with `--model`) |
| Precisions | FP16 (reference), 8-bit LLM.int8(), 4-bit NF4 with FP16 compute; both quantized versions use [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) |
| Data | [HumanEval](https://github.com/openai/human-eval): 164 Python problems with unit tests |
| Decoding | greedy, up to 512 new tokens, chat template with a short instruction |
| Hardware | free Google Colab T4 GPU |

Each prompt is sent as *"Complete the following Python function…"* followed by the HumanEval prompt in a
code block. The function is extracted from the response and run against the HumanEval tests in a separate
Python process with a 10-second timeout.

## Prompt changes

Each problem is run with its original prompt and with four changes that should not affect a correct
solution:

| Change | What it does | Applies to |
| --- | --- | --- |
| `rename` | Renames the function's arguments to `arg0`, `arg1`, … in the signature and where the docstring mentions them | 164 |
| `format` | Halves the indentation (4 → 2 spaces) and adds a blank line after the signature | 164 |
| `comment` | Adds `# Last reviewed during the 2023 code-style cleanup.` above the function | 164 |
| `reorder` | Reverses the order of the examples in the docstring (doctests, `f(x) == y`, `f(x) => y`, …) | 115 |

```diff
-def has_close_elements(numbers: List[float], threshold: float) -> bool:
+def has_close_elements(arg0: List[float], arg1: float) -> bool:
     """ Check if in given list of numbers, are any two numbers closer to each other than
-    given threshold.
+    given arg1.
```

`reorder` is skipped when a docstring has fewer than two examples it can recognise, and those problems
are left out of the `reorder` numbers.

**Checking the changes.** `python -m codecons.check_perturbations` applies every change to
*prompt + reference solution* and runs the tests. All 164 reference solutions still pass under every change
that applies to them, so the changes do not alter what a correct answer is.

## Metrics

- **pass@1**: share of problems solved. With greedy decoding there is one sample per prompt.
- **Consistency rate**: for each change, the share of problems where the pass/fail result is the same on the
  original and the changed prompt. Reported with a 95% Wilson interval.
- **Outcome flips**: the inconsistent problems split into *pass → fail* and *fail → pass*.
- **Disagreement with FP16**: problems where the FP16 model passes and the quantized model fails on the
  same prompt.
- **Change-induced failures**: FP16 passes both before and after the change, and the quantized model passes
  the original prompt but fails the changed one. These are failures that come from the prompt change and
  not from a general drop in accuracy.
- **Consistency vs FP16**: an exact McNemar test on the paired (problem, change) consistency outcomes.

## Results

<!-- RESULTS:START -->
_The results have not been generated yet. Run the [Colab notebook](notebooks/run_on_colab.ipynb), then
`python -m codecons.analyze --update-readme` writes the tables and chart into this section._
<!-- RESULTS:END -->

The task ids behind every count (which problems flipped, which ones FP16 and 4-bit disagree on) are in
[`results/summary.json`](results/summary.json).

## Discussion

<!-- Summarise the main finding here in three or four sentences once the results are in. -->

**How to read the numbers.** pass@1 alone can hide the effect. A quantized model can keep the same pass@1
while failing on *different* problems, or while flipping more often when the prompt changes. The
change-induced failure counts and the McNemar comparison are the most direct test of the question.

**Limitations.**

- **Small sample.** With 164 problems, a 95% interval on a consistency rate is roughly ±3–6 percentage
  points. A difference of a few problems between precisions is within noise. The McNemar p-value pools
  changes from the same problem, which are not independent, so it is only a rough guide.
- **One model, one quantization method.** The results cover a single small model and bitsandbytes'
  weight-only quantization. GPTQ, AWQ, larger models or other model families may behave differently.
- **Numerical noise.** Greedy decoding is deterministic for a fixed input, but batched generation with
  padding, and quantized kernels in particular, can change a token when the input length changes. Part of
  any measured inconsistency may be this kind of noise and not real prompt sensitivity.
- **The changes are not equally strong.** `rename` removes descriptive argument names, which carry real
  information for the model, so it is the strongest change. `comment` and `format` are much milder.
  `reorder` only applies to 115 problems.
- **Pass/fail is coarse.** Consistency here means the same *test outcome*, not the same code. Two different
  correct programs count as consistent.

## Repository layout

```
codecons/
  data.py                  HumanEval loading
  perturbations.py         the four prompt changes
  check_perturbations.py   checks that reference solutions survive every change
  models.py                FP16 / 8-bit / 4-bit model loading
  generation.py            prompting, batched greedy decoding, code extraction
  execution.py             running candidate solutions against the tests
  run.py                   CLI: generate + evaluate, resumable
  metrics.py               pass@1, consistency, disagreement, statistical tests
  analyze.py               CLI: tables, chart, README update
notebooks/run_on_colab.ipynb
results/                   results.md, summary.json, consistency.png
tests/                     unit tests (no GPU needed)
```

## Reproducing

On Colab, open the notebook with the badge above and run all cells. To run locally on a CUDA GPU:

```bash
pip install -r requirements.txt
python -m codecons.check_perturbations
python -m codecons.run --precision fp16 int8 int4      # writes results/raw/<precision>.jsonl
python -m codecons.analyze --update-readme             # writes results/ and updates this README
pytest                                                 # unit tests, CPU only
```

`python -m codecons.run --limit 3` gives a quick smoke test. Runs resume from the last finished batch.

Running the tests means executing code written by a model. Each program runs in its own process with a
timeout, but this is not a security sandbox. Use a disposable environment such as Colab.

## Acknowledgement

This project was inspired by MuCoCo (EMNLP 2026) and its consistency testing of code LLMs under
semantics-preserving changes.
