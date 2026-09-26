"""Semantics-preserving prompt perturbations.

Each perturbation rewrites a HumanEval prompt in a way that should not change
what a correct solution looks like (in the spirit of MuCoCo-style consistency
testing):

* ``rename``  - rename the entry-point function's arguments to neutral names
* ``format``  - change indentation width and add blank lines
* ``comment`` - add an irrelevant comment above the function
* ``reorder`` - reverse the order of the examples in the docstring

The functions work on any source that contains the entry-point function, so the
same code can be applied to ``prompt + canonical_solution`` to check that the
reference solution still passes (see ``codecons.check_perturbations``).

A perturbation that cannot be applied to a prompt (for example, a docstring with
fewer than two examples cannot be reordered) returns the source unchanged with
``applicable=False``; such problems are excluded from that perturbation's
consistency numbers.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

ORIGINAL = "original"
PERTURBATIONS = ("rename", "format", "comment", "reorder")
VARIANTS = (ORIGINAL,) + PERTURBATIONS

IRRELEVANT_COMMENT = "# Last reviewed during the 2023 code-style cleanup."

# Argument names that also read as ordinary English in docstrings. Mentions of
# these in prose are left alone so that, e.g., "a supplied string" stays readable.
_PROSE_WORDS = frozenset(
    {
        "a", "i", "array", "base", "brackets", "date", "grid", "list", "message",
        "number", "numbers", "operand", "operator", "sentence", "string", "strings",
        "text", "value", "values", "word", "words",
    }
)


@dataclass(frozen=True)
class Variant:
    name: str
    text: str
    applicable: bool


def make_variants(source: str, entry_point: str) -> List[Variant]:
    """Return the original prompt followed by one variant per perturbation."""
    variants = [Variant(ORIGINAL, source, True)]
    variants.extend(apply(name, source, entry_point) for name in PERTURBATIONS)
    return variants


def apply(name: str, source: str, entry_point: str) -> Variant:
    text = _REGISTRY[name](source, entry_point)
    if text is None or text == source or not _parses(text):
        return Variant(name, source, False)
    return Variant(name, text, True)


# --------------------------------------------------------------------------- #
# rename
# --------------------------------------------------------------------------- #


def rename_arguments(source: str, entry_point: str) -> Optional[str]:
    fn = _find_function(source, entry_point)
    if fn is None:
        return None
    a = fn.args
    params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    params += [p.arg for p in (a.vararg, a.kwarg) if p is not None]
    if not params:
        return None

    taken = _identifiers(source)
    mapping: Dict[str, str] = {}
    for i, old in enumerate(params):
        new = f"arg{i}"
        while new in taken:
            new += "_"
        mapping[old] = new

    docstring = _docstring_node(fn)
    body_start = fn.body[0].lineno
    edits: List[Tuple[int, int, int, str]] = []  # (row, col_start, col_end, replacement)

    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    depth = 0
    for i, tok in enumerate(toks):
        if tok.type == tokenize.OP and tok.string in "([{":
            depth += 1
        elif tok.type == tokenize.OP and tok.string in ")]}":
            depth -= 1
        row, col = tok.start
        if row < fn.lineno:
            continue

        if tok.type == tokenize.NAME and tok.string in mapping:
            prev_tok = toks[i - 1].string if i else ""
            next_tok = toks[i + 1].string if i + 1 < len(toks) else ""
            is_attribute = prev_tok == "."
            is_keyword_arg = row >= body_start and depth > 0 and next_tok == "="
            if not (is_attribute or is_keyword_arg):
                edits.append((row, col, tok.end[1], mapping[tok.string]))
        elif (
            tok.type == tokenize.STRING
            and docstring is not None
            and (row, col) == (docstring.lineno, docstring.col_offset)
        ):
            new_text = _rename_in_prose(tok.string, mapping)
            if new_text != tok.string:
                edits.append((row, col, -1, new_text))

    return _apply_edits(source, edits, toks)


def _rename_in_prose(text: str, mapping: Dict[str, str]) -> str:
    for old, new in mapping.items():
        if old.lower() in _PROSE_WORDS:
            continue
        # Skip mentions inside quotes (string literals in examples) and attributes.
        text = re.sub(rf"(?<![\w.'\"]){re.escape(old)}(?![\w'\"])", new, text)
    return text


def _apply_edits(source: str, edits, toks) -> str:
    """Apply token-position edits. ``col_end == -1`` replaces a whole (possibly multi-line) token."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    token_end = {tok.start: tok.end for tok in toks}

    spans = []
    for row, col, col_end, repl in edits:
        start = offsets[row - 1] + col
        if col_end == -1:
            end_row, end_col = token_end[(row, col)]
            end = offsets[end_row - 1] + end_col
        else:
            end = offsets[row - 1] + col_end
        spans.append((start, end, repl))

    out = source
    for start, end, repl in sorted(spans, reverse=True):
        out = out[:start] + repl + out[end:]
    return out


# --------------------------------------------------------------------------- #
# format
# --------------------------------------------------------------------------- #


def reformat(source: str, entry_point: str) -> Optional[str]:
    """Halve every indentation level (4 -> 2 spaces) and add a blank line after the signature."""
    fn = _find_function(source, entry_point)
    if fn is None:
        return None
    lines = source.split("\n")
    out = []
    for line in lines:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        out.append(" " * (indent // 2) + stripped)
    # body[0].lineno is 1-based; inserting at that 0-based index puts the blank
    # line directly above the first body line, i.e. right after the signature.
    out.insert(fn.body[0].lineno - 1, "")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# comment
# --------------------------------------------------------------------------- #


def add_comment(source: str, entry_point: str) -> Optional[str]:
    fn = _find_function(source, entry_point)
    if fn is None:
        return None
    lines = source.split("\n")
    def_line = lines[fn.lineno - 1]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    lines.insert(fn.lineno - 1, indent + IRRELEVANT_COMMENT)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# reorder
# --------------------------------------------------------------------------- #

_EXAMPLE_MARKERS = re.compile(r"(==|=>|➞|->|\breturns?\b|\bshould (?:return|be)\b)")


def reorder_examples(source: str, entry_point: str) -> Optional[str]:
    """Reverse each run of two or more consecutive examples in the docstring.

    Recognised example styles (all present in HumanEval):
    doctest blocks (``>>> f(x)`` followed by output lines) and single-line
    examples such as ``f(x) == y``, ``f(x) => y``, ``f(x) ➞ y``.
    """
    fn = _find_function(source, entry_point)
    doc = _docstring_node(fn) if fn is not None else None
    if doc is None:
        return None

    lines = source.split("\n")
    first, last = doc.lineno - 1, doc.end_lineno - 1  # 0-based, inclusive
    units = _example_units(lines, first, last, entry_point)

    # Group units that directly follow each other into runs.
    runs: List[List[Tuple[int, int]]] = []
    for unit in units:
        if runs and runs[-1][-1][1] == unit[0]:
            runs[-1].append(unit)
        else:
            runs.append([unit])

    changed = False
    for run in runs:
        if len(run) < 2:
            continue
        start, end = run[0][0], run[-1][1]
        reordered = [line for s, e in reversed(run) for line in lines[s:e]]
        if reordered != lines[start:end]:
            lines[start:end] = reordered
            changed = True
    return "\n".join(lines) if changed else None


def _example_units(lines: List[str], first: int, last: int, entry_point: str) -> List[Tuple[int, int]]:
    """Return ``[start, end)`` line ranges of individual examples inside the docstring."""
    call = re.compile(rf"\b{re.escape(entry_point)}\s*\(")
    units = []
    i = first
    while i <= last:
        stripped = lines[i].strip()
        if stripped.startswith(">>>"):
            j = i + 1
            while j <= last:
                nxt = lines[j].strip()
                if not nxt or nxt.startswith(">>>") or nxt.startswith(('"""', "'''")):
                    break
                j += 1
            units.append((i, j))
            i = j
            continue
        if (call.search(stripped) and _EXAMPLE_MARKERS.search(stripped)) or re.search(
            r"output should be", stripped, re.IGNORECASE
        ):
            if not stripped.startswith(('"""', "'''")) and not stripped.endswith(('"""', "'''")):
                units.append((i, i + 1))
        i += 1
    return units


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def _find_function(source: str, name: str) -> Optional[ast.FunctionDef]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            found = node  # last definition wins, as at runtime
    return found


def _docstring_node(fn: ast.FunctionDef) -> Optional[ast.Constant]:
    if fn.body and isinstance(fn.body[0], ast.Expr):
        value = fn.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value
    return None


def _identifiers(source: str) -> set:
    try:
        return {
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.NAME
        }
    except (tokenize.TokenError, IndentationError):
        return set()


_REGISTRY: Dict[str, Callable[[str, str], Optional[str]]] = {
    "rename": rename_arguments,
    "format": reformat,
    "comment": add_comment,
    "reorder": reorder_examples,
}
