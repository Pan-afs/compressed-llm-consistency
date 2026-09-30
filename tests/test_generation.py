from codecons.execution import build_program, run_program
from codecons.generation import assemble_solution, build_messages, extract_code, sanitize

PROMPT = '''from typing import List


def total(numbers: List[int]) -> int:
    """ Return the sum of the numbers.
    >>> total([1, 2])
    3
    """
'''
TEST = """
def check(candidate):
    assert candidate([1, 2]) == 3
    assert candidate([]) == 0
"""


def _passes(prompt, response):
    solution = assemble_solution(prompt, response, "total")
    return run_program(build_program(solution, TEST, "total")).passed


def test_build_messages_wraps_prompt_in_code_fence():
    (message,) = build_messages(PROMPT)
    assert message["role"] == "user"
    assert "```python\nfrom typing import List" in message["content"]


def test_extract_code_prefers_block_defining_entry_point():
    response = "Usage:\n```python\nprint(total([1]))\n```\nCode:\n```python\ndef total(numbers):\n    return sum(numbers)\n```"
    assert extract_code(response, "total").startswith("def total")


def test_extract_code_handles_truncated_block():
    response = "```python\ndef total(numbers):\n    return sum(numbers)\n"
    assert "return sum(numbers)" in extract_code(response, "total")


def test_full_function_response_passes():
    response = "Here you go:\n```python\ndef total(numbers: List[int]) -> int:\n    return sum(numbers)\n```"
    assert _passes(PROMPT, response)


def test_body_only_response_is_indented_to_match_prompt():
    two_space_prompt = 'def total(numbers):\n  """Sum."""\n'
    assert _passes(two_space_prompt, "```python\nreturn sum(numbers)\n```")


def test_wrong_function_fails():
    assert not _passes(PROMPT, "```python\ndef total(numbers):\n    return 0\n```")


def test_sanitize_drops_main_block_and_prints():
    code = (
        "import math\n\n@staticmethod\ndef f():\n    return 1\n\nLIMIT = 3\n"
        "print(f())\nx = input()\nif __name__ == '__main__':\n    f()\n"
    )
    cleaned = sanitize(code)
    assert "import math" in cleaned
    assert "@staticmethod\ndef f():" in cleaned
    assert "LIMIT = 3" in cleaned
    assert "print" not in cleaned and "input" not in cleaned and "__main__" not in cleaned
