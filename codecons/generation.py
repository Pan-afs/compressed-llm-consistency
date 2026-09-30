"""Prompting the model and turning its response into a runnable solution."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Iterator, List, Sequence, Tuple

INSTRUCTION = (
    "Complete the following Python function. "
    "Return the complete function in a single Python code block.\n\n"
    "```python\n{prompt}\n```"
)

_KEEP_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def build_messages(prompt: str) -> List[dict]:
    return [{"role": "user", "content": INSTRUCTION.format(prompt=prompt.strip("\n"))}]


def generate(
    model,
    tokenizer,
    prompts: Sequence[str],
    batch_size: int = 16,
    max_new_tokens: int = 512,
) -> Iterator[List[Tuple[int, str]]]:
    """Greedy-decode every prompt, yielding ``[(index, response), ...]`` one batch at a time.

    Prompts are processed longest first so that padding is minimal and an
    out-of-memory error shows up on the first batch rather than the last.
    """
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(build_messages(prompts[i]), tokenize=False, add_generation_prompt=True)
            for i in idx
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # Override the sampling defaults shipped in the model's generation config.
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        yield list(zip(idx, responses))


def extract_code(response: str, entry_point: str) -> str:
    """Pull the most relevant Python code block out of a chat response."""
    blocks = re.findall(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*?)```", response, re.DOTALL)
    if not blocks:
        # A response cut off by max_new_tokens has no closing fence.
        unterminated = re.search(r"```[ \t]*(?:python|py|Python)?[ \t]*\n(.*)", response, re.DOTALL)
        blocks = [unterminated.group(1)] if unterminated else [response]
    for block in blocks:
        if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", block):
            return block
    return blocks[0]


def assemble_solution(prompt: str, response: str, entry_point: str) -> str:
    """Combine the (perturbed) prompt and the model response into one module.

    If the response defines the entry-point function, it is appended after the
    prompt so it replaces the stub while keeping the prompt's imports and helper
    functions. Otherwise the response is treated as the function body.
    """
    code = extract_code(response, entry_point)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE):
        return prompt.rstrip() + "\n\n\n" + sanitize(code)
    body = textwrap.indent(textwrap.dedent(code).strip("\n"), _body_indent(prompt, entry_point))
    return prompt.rstrip() + "\n" + body + "\n"


def sanitize(code: str) -> str:
    """Keep imports, definitions and call-free assignments; drop top-level statements
    such as ``print(...)`` or ``if __name__ == "__main__":`` blocks that could
    block on input or slow down evaluation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    kept = []
    for node in tree.body:
        keep = isinstance(node, _KEEP_NODES) or (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and not any(isinstance(n, ast.Call) for n in ast.walk(node))
        )
        if keep:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            kept.append("\n".join(lines[first - 1 : node.end_lineno]))
    return "\n\n".join(kept) + "\n"


def _body_indent(prompt: str, entry_point: str) -> str:
    lines = prompt.split("\n")
    seen_def = False
    for line in lines:
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
            seen_def = True
            continue
        if seen_def and line.strip():
            return line[: len(line) - len(line.lstrip())]
    return "    "
