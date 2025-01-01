#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import sys
import typing as t

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
)


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Pattern matching requires Python 3.10 or newer",
)
class TestPatternMatching:
    """Test pattern matching functionality in Python 3.10+ match-case
    statements.

    Tests various pattern matching features including literal patterns,
    sequence patterns, mapping patterns, OR patterns, AS patterns,
    guard patterns, and class patterns.
    """

    parser: ExpressionsParser
    interpreter: ExpressionsInterpreter

    def setup_method(self) -> None:
        """Set up test environment."""
        self.parser = ExpressionsParser()
        self.interpreter = ExpressionsInterpreter()

        # Define and register Point class
        class Point:
            __match_args__: t.Tuple[
                t.Literal["x"],
                t.Literal["y"],
            ] = (
                "x",
                "y",
            )

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.interpreter.global_env["Point"] = Point

    def test_literal_patterns(self):
        """Test literal pattern matching in match-case statements.

        Tests:
        1. Integer literal matching
        2. String literal matching
        3. Boolean literal matching
        4. Wildcard pattern matching (_)
        """
        code = """
        def match_value(x):
            match x:
                case 1:
                    result = "one"
                case "hello":
                    result = "greeting"
                case True:
                    result = "boolean"
                case _:
                    result = "other"
            return result

        r1 = match_value(1)
        r2 = match_value("hello")
        r3 = match_value(True)
        r4 = match_value(42)
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "one"
        assert self.interpreter.get_name_value("r2") == "greeting"
        assert self.interpreter.get_name_value("r3") == "boolean"
        assert self.interpreter.get_name_value("r4") == "other"

    def test_sequence_patterns(self):
        """Test sequence pattern matching in match-case statements.

        Tests:
        1. Empty sequence matching
        2. Single element sequence matching
        3. Two element sequence matching
        4. Sequence with rest (*) pattern matching
        """
        code = """
        def match_sequence(lst):
            match lst:
                case []:
                    result = "empty"
                case [x]:
                    result = f"singleton: {x}"
                case [x, y]:
                    result = f"pair: {x}, {y}"
                case [x, *rest]:
                    result = f"first: {x}, rest: {rest}"
            return result

        r1 = match_sequence([])
        r2 = match_sequence([1])
        r3 = match_sequence([1, 2])
        r4 = match_sequence([1, 2, 3])
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "empty"
        assert self.interpreter.get_name_value("r2") == "singleton: 1"
        assert self.interpreter.get_name_value("r3") == "pair: 1, 2"
        assert self.interpreter.get_name_value("r4") == "first: 1, rest: [2, 3]"

    def test_mapping_patterns(self):
        """Test mapping (dictionary) pattern matching in match-case statements.

        Tests:
        1. Exact key matching with value capture
        2. Partial key matching with rest (**) capture
        3. Non-matching pattern fallback
        """
        code = """
        def match_mapping(data):
            match data:
                case {"name": name, "age": age}:
                    result = f"{name} is {age} years old"
                case {"items": items, **rest}:
                    result = f"items: {items}, other fields: {rest}"
                case _:
                    result = "no match"
            return result

        r1 = match_mapping({"name": "Alice", "age": 30})
        r2 = match_mapping({"items": [1, 2, 3], "extra": "data"})
        r3 = match_mapping({"other": "value"})
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "Alice is 30 years old"
        assert (
            self.interpreter.get_name_value("r2")
            == "items: [1, 2, 3], other fields: {'extra': 'data'}"
        )
        assert self.interpreter.get_name_value("r3") == "no match"

    def test_or_patterns(self):
        """Test OR pattern matching (pattern1 | pattern2) in match-case
        statements.

        Tests:
        1. Multiple numeric alternatives
        2. Multiple string alternatives
        3. Default case handling
        """
        code = """
        def match_or(value):
            match value:
                case 1 | 2 | 3:
                    result = "small number"
                case "a" | "b" | "c":
                    result = "first letters"
                case _:
                    result = "something else"
            return result

        r1 = match_or(1)
        r2 = match_or("b")
        r3 = match_or(42)
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "small number"
        assert self.interpreter.get_name_value("r2") == "first letters"
        assert self.interpreter.get_name_value("r3") == "something else"

    def test_as_patterns(self):
        """Test AS pattern binding in match-case statements.

        Tests:
        1. List pattern with AS binding
        2. Dictionary pattern with AS binding
        3. Simple value capture with AS pattern
        """
        code = """
        def match_as(value):
            match value:
                case [1, 2] as lst:
                    result = f"matched list: {lst}"
                case {"x": x} as d:
                    result = f"matched dict: {d}"
                case x:
                    result = f"captured value: {x}"
            return result

        r1 = match_as([1, 2])
        r2 = match_as({"x": 42})
        r3 = match_as("hello")
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "matched list: [1, 2]"
        assert (
            self.interpreter.get_name_value("r2") == "matched dict: {'x': 42}"
        )
        assert self.interpreter.get_name_value("r3") == "captured value: hello"

    def test_guard_patterns(self):
        """Test guard conditions in match-case patterns.

        Tests:
        1. Pattern matching with equality guard
        2. Pattern matching with comparison guard
        3. Pattern matching without guard (default case)
        """
        code = """
        def match_guard(point):
            match point:
                case (x, y) if x == y:
                    result = "on diagonal"
                case (x, y) if x > y:
                    result = "above diagonal"
                case (x, y):
                    result = "below diagonal"
            return result

        r1 = match_guard((5, 5))
        r2 = match_guard((10, 5))
        r3 = match_guard((3, 7))
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "on diagonal"
        assert self.interpreter.get_name_value("r2") == "above diagonal"
        assert self.interpreter.get_name_value("r3") == "below diagonal"

    def test_class_patterns(self):
        """Test class pattern matching with positional and keyword arguments.

        Tests:
        1. Origin point matching (0, 0)
        2. Y-axis point matching (x=0)
        3. X-axis point matching (y=0)
        4. General point matching with coordinate capture
        """
        code = """
        def match_class(point):
            match point:
                case Point(0, 0):
                    result = "origin"
                case Point(x=0):
                    result = "on y-axis"
                case Point(y=0):
                    result = "on x-axis"
                case Point(x, y):
                    result = f"point({x}, {y})"
            return result

        r1 = match_class(Point(0, 0))
        r2 = match_class(Point(0, 5))
        r3 = match_class(Point(5, 0))
        r4 = match_class(Point(3, 4))
        """
        self.interpreter.execute(self.parser.parse(code))

        assert self.interpreter.get_name_value("r1") == "origin"
        assert self.interpreter.get_name_value("r2") == "on y-axis"
        assert self.interpreter.get_name_value("r3") == "on x-axis"
        assert self.interpreter.get_name_value("r4") == "point(3, 4)"

    def test_match_pattern_edge_cases(self):
        """Test edge cases in pattern matching.

        Tests:
        1. Multiple starred expressions
        2. Invalid mapping patterns
        3. Invalid class patterns
        4. Invalid sequence patterns
        5. Invalid value patterns
        """
        parser = ExpressionsParser()
        interpreter = ExpressionsInterpreter()

        # Test invalid sequence pattern with multiple starred expressions
        code = """
def test_multiple_stars():
    match [1, 2, 3]:
        case [*x, 2, *y]:
            result = (x, y)
        case _:
            result = "no match"
    return result
"""
        with pytest.raises(SyntaxError) as exc_info:
            interpreter.execute(parser.parse(code))
        assert "multiple starred expressions in sequence pattern" in str(
            exc_info.value
        )

        # Test invalid mapping pattern with non-dict
        code = """
result = None
def test_invalid_mapping():
    match None:
        case {"key": value}:
            result = value
        case _:
            result = "no match"
    return result

result = test_invalid_mapping()
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == "no match"

        # Test invalid class pattern with non-class instance
        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def test_invalid_class():
    match 42:
        case Point(x=0, y=0):
            result = (x, y)
        case _:
            result = "no match"
    return result

result = test_invalid_class()
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == "no match"

        # Test invalid sequence pattern with wrong length
        code = """
def test_invalid_sequence():
    match [1]:
        case [x, y]:  # Expects 2 elements but gets 1
            result = (x, y)
        case _:
            result = "no match"
    return result

result = test_invalid_sequence()
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == "no match"

        # Test invalid value pattern
        code = """
def test_invalid_value():
    match 42:
        case "42":  # String vs int
            result = "string"
        case _:
            result = "no match"
    return result

result = test_invalid_value()
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == "no match"

    def test_match_pattern_binding(self):
        """Test pattern binding in match statements.

        Tests:
        1. Binding in sequence patterns
        2. Binding in mapping patterns
        3. Binding in class patterns
        4. Binding with guards
        """
        parser = ExpressionsParser()
        interpreter = ExpressionsInterpreter()

        # Test sequence pattern binding
        code = """
match [1, 2, 3, 4]:
    case [first, *rest, last]:
        result = (first, rest, last)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, [2, 3], 4)

        # Test mapping pattern binding
        code = """
match {"a": 1, "b": 2, "c": 3}:
    case {"a": x, **rest}:
        result = (x, rest)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, {"b": 2, "c": 3})

        # Test class pattern binding
        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
match p:
    case Point(x=x, y=y):
        result = (x, y)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, 2)

        # Test binding with guards
        code = """
match [1, 2, 3]:
    case [x, y, z] if x < y < z:
        result = "ascending"
    case [x, y, z]:
        result = "not ascending"
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == "ascending"

    def test_match_pattern_or(self):
        """Test OR patterns in match statements.

        Tests:
        1. Simple OR patterns
        2. OR patterns with binding
        3. OR patterns with guards
        4. Nested OR patterns
        """
        parser = ExpressionsParser()
        interpreter = ExpressionsInterpreter()

        # Test simple OR patterns
        code = """
def classify(x):
    match x:
        case 0 | 1 | 2:
            return "small"
        case 3 | 4 | 5:
            return "medium"
        case _:
            return "large"

result = [classify(i) for i in range(7)]
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == [
            "small",
            "small",
            "small",
            "medium",
            "medium",
            "medium",
            "large",
        ]

        # Test OR patterns with binding
        code = """
def extract(x):
    match x:
        case [a] | (a,):
            return f"single: {a}"
        case [a, b] | (a, b):
            return f"pair: {a}, {b}"
        case _:
            return "other"

result = [
    extract([1]),
    extract((2,)),
    extract([1, 2]),
    extract((3, 4)),
    extract([1, 2, 3])
]
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == [
            "single: 1",
            "single: 2",
            "pair: 1, 2",
            "pair: 3, 4",
            "other",
        ]

        # Test OR patterns with guards
        code = """
def check(x):
    match x:
        case (n, m) | [n, m] if n < m:
            return "ascending"
        case (n, m) | [n, m]:
            return "not ascending"
        case _:
            return "other"

result = [
    check((1, 2)),
    check([2, 1]),
    check([1, 2]),
    check((2, 1)),
    check(42)
]
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == [
            "ascending",
            "not ascending",
            "ascending",
            "not ascending",
            "other",
        ]

        # Test nested OR patterns
        code = """
def deep_match(x):
    match x:
        case [1 | 2, 3 | 4]:
            return "pattern 1"
        case {"a": 1 | 2, "b": 3 | 4}:
            return "pattern 2"
        case _:
            return "other"

result = [
    deep_match([1, 3]),
    deep_match([2, 4]),
    deep_match({"a": 1, "b": 3}),
    deep_match({"a": 2, "b": 4}),
    deep_match([1, 5])
]
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == [
            "pattern 1",
            "pattern 1",
            "pattern 2",
            "pattern 2",
            "other",
        ]

    def test_match_pattern_as(self):
        """Test AS patterns in match statements.

        Tests:
        1. Simple AS patterns
        2. AS patterns with sequence unpacking
        3. AS patterns with mapping
        4. AS patterns with class patterns
        """
        parser = ExpressionsParser()
        interpreter = ExpressionsInterpreter()

        # Test simple AS patterns
        code = """
match 42:
    case x as value:
        result = (x, value)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (42, 42)

        # Test AS patterns with sequence unpacking
        code = """
match [1, 2, 3]:
    case [first, *rest] as full:
        result = (first, rest, full)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, [2, 3], [1, 2, 3])

        # Test AS patterns with mapping
        code = """
match {"x": 1, "y": 2}:
    case {"x": x, "y": y} as point:
        result = (x, y, point)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, 2, {"x": 1, "y": 2})

        # Test AS patterns with class patterns
        code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
match p:
    case Point(x=x, y=y) as point:
        result = (x, y, point is p)
"""
        interpreter.execute(parser.parse(code))
        assert interpreter.local_env["result"] == (1, 2, True)
