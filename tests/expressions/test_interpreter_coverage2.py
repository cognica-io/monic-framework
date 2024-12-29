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


def test_augmented_assignment_edge_cases():
    """Test edge cases in augmented assignment operations."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test augmented assignment with attribute
    code = """
class Counter:
    def __init__(self):
        self.value = 0

counter = Counter()
counter.value += 5
counter.value
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 5

    # Test augmented assignment with subscript
    code = """
data = {'count': 0}
data['count'] += 10
data['count']
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 10

    # Test all augmented assignment operators
    code = """
results = []
x = 10

x += 5
results.append(x)  # 15

x -= 3
results.append(x)  # 12

x *= 2
results.append(x)  # 24

x //= 5
results.append(x)  # 4

x %= 3
results.append(x)  # 1

x = 2
x **= 3
results.append(x)  # 8

x = 12
x &= 10
results.append(x)  # 8

x |= 5
results.append(x)  # 13

x ^= 3
results.append(x)  # 14

x = 8
x <<= 1
results.append(x)  # 16

x >>= 2
results.append(x)  # 4

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [15, 12, 24, 4, 1, 8, 8, 13, 14, 16, 4]


def test_function_definition_edge_cases():
    """Test edge cases in function definition and calling."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test function with keyword-only arguments
    code = """
def greet(*, name="World", prefix="Hello"):
    return f"{prefix}, {name}!"

results = []
results.append(greet(name="Alice"))
results.append(greet(prefix="Hi", name="Bob"))
try:
    results.append(greet("Charlie"))  # Should fail
except TypeError:
    results.append("TypeError")
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["Hello, Alice!", "Hi, Bob!", "TypeError"]

    # Test function with *args and **kwargs
    code = """
def func(*args, **kwargs):
    return len(args), sorted(kwargs.keys())

result = func(1, 2, 3, a=4, b=5)
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == (3, ["a", "b"])


def test_with_statement_edge_cases():
    """Test edge cases in with statement handling."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test with statement with multiple context managers
    code = """
class Context:
    def __init__(self, name):
        self.name = name
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

ctx1 = Context("first")
ctx2 = Context("second")

with ctx1 as c1, ctx2 as c2:
    pass

result = [ctx1.entered, ctx1.exited, ctx2.entered, ctx2.exited]
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [True, True, True, True]

    # Test with statement exception handling
    code = """
class ExceptionSuppress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # Suppress all exceptions

results = []
with ExceptionSuppress():
    raise ValueError("This should be suppressed")
results.append("After suppressed exception")

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["After suppressed exception"]


def test_match_statement_additional_patterns():
    """Test additional match statement patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test match with mapping patterns
    code = """
def process_config(config):
    match config:
        case {"type": "user", "name": str() as name}:
            return f"User: {name}"
        case {"type": "group", "members": list() as members}:
            return f"Group with {len(members)} members"
        case {"type": str() as type, **rest}:
            return f"Other type: {type} with {len(rest)} extra fields"
        case _:
            return "Unknown"

results = []
results.append(process_config({"type": "user", "name": "Alice"}))
results.append(process_config({"type": "group", "members": [1, 2, 3]}))
results.append(process_config({"type": "other", "field1": 1, "field2": 2}))
results.append(process_config({"invalid": "config"}))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "User: Alice",
        "Group with 3 members",
        "Other type: other with 2 extra fields",
        "Unknown",
    ]

    # Test match with class patterns
    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    __match_args__ = ("x", "y")

def classify_point(point):
    match point:
        case Point(0, 0):
            return "Origin"
        case Point(x, 0):
            return f"On X-axis at {x}"
        case Point(0, y):
            return f"On Y-axis at {y}"
        case Point(x, y):
            return f"Point at ({x}, {y})"
        case _:
            return "Not a point"

results = []
results.append(classify_point(Point(0, 0)))
results.append(classify_point(Point(5, 0)))
results.append(classify_point(Point(0, 3)))
results.append(classify_point(Point(2, 4)))
results.append(classify_point("not a point"))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "Origin",
        "On X-axis at 5",
        "On Y-axis at 3",
        "Point at (2, 4)",
        "Not a point",
    ]
