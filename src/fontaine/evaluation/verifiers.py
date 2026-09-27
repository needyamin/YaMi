"""Verifiers that check a completion against a known result.

Code execution runs in a subprocess with a timeout and a working directory.
It does not grant network access by itself; the child still has the parent's
network unless the operating system sandbox says otherwise. Callers that need
a filesystem jail should pass a directory from the agent sandbox.
"""

import ast
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def exact_match(prediction: str, expected: str) -> bool:
    return prediction.strip() == expected.strip()


def mathematical(prediction: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Compare the last number in ``prediction`` to ``expected``."""
    found = _NUMBER.findall(prediction)
    wanted = _NUMBER.findall(expected)
    if not found or not wanted:
        return False
    return abs(float(found[-1]) - float(wanted[-1])) <= tolerance


def run_python(code: str, workdir: Path, timeout_s: float = 5.0) -> subprocess.CompletedProcess[str]:
    """Run ``code`` with ``python -I`` in ``workdir``. The caller bounds the directory."""
    workdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def unit_test_verifier(code: str, assertion: str, workdir: Path, timeout_s: float = 5.0) -> bool:
    """True when ``assertion`` is a Python expression that evaluates true after ``code``."""
    try:
        ast.parse(assertion)
    except SyntaxError as exc:
        raise ValueError(f"unit test assertion is not valid Python: {exc}") from exc
    program = code + "\n" + f"assert ({assertion})"
    try:
        result = run_python(program, workdir, timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


_CUSTOM: dict[str, Callable[[str, str], bool]] = {}


def register_verifier(name: str, fn: Callable[[str, str], bool]) -> None:
    if name in _CUSTOM:
        raise ValueError(f"verifier {name!r} is already registered")
    _CUSTOM[name] = fn


def custom_verifier(name: str, prediction: str, expected: str) -> bool:
    if name not in _CUSTOM:
        raise ValueError(
            f"unknown custom verifier {name!r}. Registered: {sorted(_CUSTOM) or '(none)'}."
        )
    return _CUSTOM[name](prediction, expected)


def verify(kind: str, prediction: str, expected: str, **kwargs: object) -> bool:
    if kind == "exact":
        return exact_match(prediction, expected)
    if kind == "mathematical":
        return mathematical(prediction, expected, float(kwargs.get("tolerance", 1e-6)))
    if kind == "custom":
        return custom_verifier(str(kwargs["name"]), prediction, expected)
    raise ValueError(
        f"unknown verifier {kind!r}. Choose exact, mathematical, unit_test, or custom. "
        "unit_test needs unit_test_verifier because it runs code."
    )
