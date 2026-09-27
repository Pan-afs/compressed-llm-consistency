from codecons.execution import ERROR, FAILED, PASSED, TIMEOUT, build_program, run_program

TEST = """
def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(-1, 1) == 0
"""


def _run(solution, timeout=10.0):
    return run_program(build_program(solution, TEST, "add"), timeout=timeout)


def test_correct_solution_passes():
    result = _run("def add(a, b):\n    return a + b\n")
    assert result.status == PASSED
    assert result.passed


def test_wrong_answer_fails():
    assert _run("def add(a, b):\n    return a - b\n").status == FAILED


def test_crash_is_an_error():
    assert _run("def add(a, b):\n    return undefined_name\n").status == ERROR


def test_syntax_error_is_an_error():
    assert _run("def add(a, b)\n    return a + b\n").status == ERROR


def test_infinite_loop_times_out():
    assert _run("def add(a, b):\n    while True:\n        pass\n", timeout=1.0).status == TIMEOUT


def test_reading_stdin_does_not_hang():
    assert _run("def add(a, b):\n    input()\n    return a + b\n", timeout=5.0).status == ERROR
