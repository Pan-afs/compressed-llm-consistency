import ast

from codecons.perturbations import (
    IRRELEVANT_COMMENT,
    PERTURBATIONS,
    apply,
    make_variants,
)

DOCTEST_PROMPT = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

ARROW_PROMPT = '''
def is_happy(s):
    """You are given a string s.
    For example:
    is_happy(a) => False
    is_happy(abcd) => True
    is_happy('s') => False
    """
'''

HELPER_PROMPT = '''

def is_palindrome(string: str) -> bool:
    """ Test if given string is a palindrome """
    return string == string[::-1]


def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.
    >>> make_palindrome('cat')
    'catac'
    """
'''


def _args(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise AssertionError(name)


def test_make_variants_returns_original_first_and_every_perturbation():
    variants = make_variants(DOCTEST_PROMPT, "has_close_elements")
    assert [v.name for v in variants] == ["original", *PERTURBATIONS]
    assert variants[0].text == DOCTEST_PROMPT
    for v in variants:
        ast.parse(v.text)


def test_rename_changes_signature_and_identifier_mentions():
    v = apply("rename", DOCTEST_PROMPT, "has_close_elements")
    assert v.applicable
    assert _args(v.text, "has_close_elements") == ["arg0", "arg1"]
    assert "given arg1." in v.text
    # "numbers" is also an English word; prose mentions are kept.
    assert "list of numbers" in v.text


def test_rename_leaves_quoted_literals_and_helpers_alone():
    v = apply("rename", ARROW_PROMPT, "is_happy")
    assert "a string arg0." in v.text
    assert "is_happy('s')" in v.text

    v = apply("rename", HELPER_PROMPT, "make_palindrome")
    assert _args(v.text, "is_palindrome") == ["string"]
    assert _args(v.text, "make_palindrome") == ["arg0"]


def test_rename_skips_keyword_arguments_in_body():
    source = "def f(key):\n    return sorted([3, 1], key=None) if key else key\n"
    v = apply("rename", source, "f")
    assert "sorted([3, 1], key=None) if arg0 else arg0" in v.text


def test_format_halves_indentation_and_adds_blank_line():
    v = apply("format", DOCTEST_PROMPT, "has_close_elements")
    lines = v.text.split("\n")
    def_idx = next(i for i, line in enumerate(lines) if line.startswith("def "))
    assert lines[def_idx + 1] == ""
    assert lines[def_idx + 2].startswith('  """')


def test_comment_is_inserted_above_entry_point():
    v = apply("comment", HELPER_PROMPT, "make_palindrome")
    lines = v.text.split("\n")
    idx = lines.index(IRRELEVANT_COMMENT)
    assert lines[idx + 1].startswith("def make_palindrome")


def test_reorder_reverses_doctest_blocks_with_their_outputs():
    v = apply("reorder", DOCTEST_PROMPT, "has_close_elements")
    assert v.applicable
    first = v.text.index("[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True")
    second = v.text.index("[1.0, 2.0, 3.0], 0.5)\n    False")
    assert first < second


def test_reorder_handles_single_line_examples():
    v = apply("reorder", ARROW_PROMPT, "is_happy")
    assert v.text.index("is_happy('s')") < v.text.index("is_happy(abcd)") < v.text.index("is_happy(a)")


def test_reorder_not_applicable_with_a_single_example():
    v = apply("reorder", HELPER_PROMPT, "make_palindrome")
    assert not v.applicable
    assert v.text == HELPER_PROMPT
