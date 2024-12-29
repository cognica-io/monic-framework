#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=unused-variable

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
)
from monic.expressions.registry import registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_comprehension_with_named_expressions():
    """Test comprehensions with named expressions."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test list comprehension with named expression
    code = """
numbers = [1, 2, 3, 4, 5]
squares = [(n, square := n**2) for n in numbers if (double := n*2) > 5]
squares
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [(3, 9), (4, 16), (5, 25)]

    # Test nested comprehension with named expression
    code = """
matrix = [[1, 2], [3, 4]]
sums = [(i, j, s := row[j]) for i, row in enumerate(matrix) for j in range(len(row))]
sums
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4)]


def test_error_handling_additional_cases():
    """Test additional error handling cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test re-raising an exception
    code = """
def outer():
    try:
        inner()
    except ValueError as e:
        raise RuntimeError("Wrapped error") from e

def inner():
    raise ValueError("Original error")

try:
    outer()
except RuntimeError as e:
    result = str(e)

result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "Wrapped error"

    # Test finally block with return
    code = """
def test_func():
    try:
        return 1
    finally:
        return 2

result = test_func()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 2

    # Test nested exception handling
    code = """
results = []
try:
    try:
        raise ValueError("test")
    except ValueError:
        results.append("caught ValueError")
        raise TypeError("converted")
except TypeError:
    results.append("caught TypeError")

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["caught ValueError", "caught TypeError"]


def test_class_inheritance_edge_cases():
    """Test edge cases in class inheritance."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test multiple inheritance method resolution
    code = """
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return super().method() + "B"

class C(A):
    def method(self):
        return super().method() + "C"

class D(C, B):
    def method(self):
        return super().method() + "D"

result = D().method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "ABCD"

    # Test property inheritance and override
    code = """
class Base:
    @property
    def value(self):
        return 1

class Child(Base):
    @property
    def value(self):
        return super().value + 1

class GrandChild(Child):
    @property
    def value(self):
        return super().value + 1

result = GrandChild().value
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 3


def test_generator_expression():
    """Test generator expressions and iteration."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test list comprehension with filtering
    code = """
def sum_even_squares():
    numbers = range(10)
    squares = [n*n for n in numbers if n % 2 == 0]
    return sum(squares)

result = sum_even_squares()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 120  # 0^2 + 2^2 + 4^2 + 6^2 + 8^2

    # Test nested list comprehension
    code = """
matrix = [[1, 2], [3, 4]]
flat = [x for row in matrix for x in row if x > 1]
flat
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [2, 3, 4]


def test_match_or_patterns():
    """Test match statement with OR patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test OR patterns in match statement
    code = """
def describe_number(n):
    match n:
        case 0 | 1:
            return "binary digit"
        case 2 | 3 | 4:
            return "small number"
        case n if n < 0:
            return "negative"
        case _:
            return "large number"

results = []
results.append(describe_number(0))
results.append(describe_number(3))
results.append(describe_number(-1))
results.append(describe_number(10))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "binary digit",
        "small number",
        "negative",
        "large number",
    ]

    # Test OR patterns with capture
    code = """
def describe_command(cmd):
    match cmd:
        case ["quit"] | ["exit"] | ["bye"] as exit_cmd:
            return f"Exit command: {exit_cmd[0]}"
        case ["help"] | ["?"] as help_cmd:
            return f"Help command: {help_cmd[0]}"
        case _:
            return "Unknown command"

results = []
results.append(describe_command(["quit"]))
results.append(describe_command(["?"]))
results.append(describe_command(["unknown"]))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "Exit command: quit",
        "Help command: ?",
        "Unknown command",
    ]
