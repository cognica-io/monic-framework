#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=unused-variable

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    SecurityError,
    monic_bind,
)
from monic.expressions.registry import registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    registry.reset()
    yield


def test_complex_method_handling():
    """Test complex method handling including property descriptors,
    bound methods, and static methods.

    Tests:
    1. Property descriptor access and modification
    2. Method binding and execution
    3. Static method calls on both bound and unbound classes
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test property descriptor
    @monic_bind("test.TestClass")
    class TestClass:
        def __init__(self):
            self._value = 0

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, val):
            self._value = val

        def method(self):
            return self._value

        @staticmethod
        def static_method():
            return 42

    # Add registry namespace to interpreter
    interpreter.global_env.update(registry.get_all())

    # Test property access and modification
    code = """
obj = test.TestClass()
obj.value = 42
result = obj.value + obj.method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 84  # 42 + 42

    # Test method binding
    code = """
obj = test.TestClass()
method = obj.method
obj.value = 10
result = method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 10

    # Test bound class static method
    code = """
result = test.TestClass.static_method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 42

    # Test static method
    code = """
class StaticTest:
    @staticmethod
    def static_method():
        return 42

result = StaticTest.static_method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 42


def test_comprehension_edge_cases():
    """Test edge cases in list, set, and dict comprehensions.

    Tests:
    1. Nested comprehensions with multiple if conditions
    2. Set comprehension with complex conditions
    3. Dict comprehension with key transformation
    4. Nested dict comprehension with enumeration
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test nested comprehensions with multiple if conditions
    code = """
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix if sum(row) > 10 for x in row if x % 2 == 0]
flat
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [4, 6, 8]  # Only from rows with sum > 10

    # Test set comprehension with complex conditions
    code = """
data = [1, 2, 2, 3, 3, 3]
unique_squares = {x*x for x in data if x > 1}
unique_squares
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == {4, 9}

    # Test dict comprehension with key transformation
    code = """
items = ['a', 'bb', 'ccc']
lengths = {s: len(s) for s in items if len(s) > 1}
lengths
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == {"bb": 2, "ccc": 3}

    # Test nested dict comprehension
    code = """
matrix = [[1, 2], [3, 4]]
result = {i: {j: i*j for j in row} for i, row in enumerate(matrix, 1)}
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == {1: {1: 1, 2: 2}, 2: {3: 6, 4: 8}}


def test_class_definition_special_cases():
    """Test special cases in class definition and inheritance."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Add required built-ins
    interpreter.global_env.update(
        {
            "type": type,
            "super": super,
            "classmethod": classmethod,
            "property": property,
            "staticmethod": staticmethod,
        }
    )

    # Test class with class method
    code = """
class Test:
    @classmethod
    def class_method(cls):
        return 42

result = Test.class_method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 42

    # Test multiple inheritance and super()
    code = """
class A:
    def method(self):
        return 1

class B:
    def method(self):
        return 2

class C(A, B):
    def method(self):
        return super().method() + 10

obj = C()
result = obj.method()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 11  # super() calls A.method() which returns 1, + 10

    # Test property with inheritance
    code = """
class Base:
    def get_value(self):
        return 42

    @property
    def value(self):
        return self.get_value()

class Child(Base):
    def get_value(self):
        return super().get_value() + 1

obj = Child()
result = obj.value
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 43


def test_match_statement_patterns():
    """Test various match statement patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test basic pattern matching
    code = """
def get_command(cmd):
    match cmd:
        case ["quit"]:
            return "QUIT"
        case ["load", filename]:
            return f"LOAD {filename}"
        case ["save", filename]:
            return f"SAVE {filename}"
        case _:
            return "UNKNOWN"

result = []
result.append(get_command(["quit"]))
result.append(get_command(["load", "test.txt"]))
result.append(get_command(["save", "data.dat"]))
result.append(get_command(["invalid"]))
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["QUIT", "LOAD test.txt", "SAVE data.dat", "UNKNOWN"]

    # Test complex pattern matching with guards
    code = """
def classify_point(point):
    match point:
        case (x, y) if x == 0 and y == 0:
            return "origin"
        case (0, y):
            return "y-axis"
        case (x, 0):
            return "x-axis"
        case (x, y) if x == y:
            return "diagonal"
        case _:
            return "other"

result = []
result.append(classify_point((0, 0)))
result.append(classify_point((0, 5)))
result.append(classify_point((3, 0)))
result.append(classify_point((2, 2)))
result.append(classify_point((1, 2)))
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["origin", "y-axis", "x-axis", "diagonal", "other"]

    # Test match with class patterns
    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def classify_object(obj):
    match obj:
        case Point(x=0, y=0):
            return "origin"
        case Point(x=0):
            return "y-axis"
        case Point(y=0):
            return "x-axis"
        case Point():
            return "point"
        case _:
            return "unknown"

result = []
result.append(classify_object(Point(0, 0)))
result.append(classify_object(Point(0, 5)))
result.append(classify_object(Point(3, 0)))
result.append(classify_object(Point(2, 2)))
result.append(classify_object(42))
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["origin", "y-axis", "x-axis", "point", "unknown"]


def test_registry_error_paths():
    """Test error paths in the registry module.

    Tests:
    1. Binding invalid names (namespace conflicts)
    2. Duplicate binding attempts
    3. Accessing non-existent items
    4. Accessing non-existent namespaces
    """
    # Test binding invalid names
    with pytest.raises(ValueError, match="is already bound as a non-namespace"):

        @monic_bind("test")
        def test_func():
            return 42  # pragma: no cover

        @monic_bind("test.func")
        def test_func2():
            return 43  # pragma: no cover

    # Test duplicate binding
    @monic_bind("math.add")
    def add1():
        return 42  # pragma: no cover

    with pytest.raises(ValueError, match="already bound"):

        @monic_bind("math.add")
        def add2():
            return 43  # pragma: no cover

    # Test accessing non-existent items
    with pytest.raises(AttributeError):
        _ = getattr(getattr(registry, "test"), "non_existent")  # type: ignore

    # Test accessing non-existent namespace
    with pytest.raises(AttributeError):
        _ = getattr(registry, "non_existent")  # type: ignore


def test_error_handling_edge_cases():
    """Test error handling edge cases in the interpreter.

    Tests:
    1. Invalid unpacking scenarios
    2. Return/break/continue outside proper context
    3. Invalid attribute access
    4. Forbidden function calls
    5. Invalid import attempts
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test invalid unpacking
    code = """
a, b = 42  # Can't unpack non-iterable
"""
    tree = parser.parse(code)
    with pytest.raises(
        TypeError, match="cannot unpack non-iterable int object"
    ):
        interpreter.execute(tree)

    # Test too many values to unpack
    code = """
a, b = [1, 2, 3]  # Too many values
"""
    tree = parser.parse(code)
    with pytest.raises(ValueError, match="too many values to unpack"):
        interpreter.execute(tree)

    # Test not enough values to unpack
    code = """
a, b, c = [1, 2]  # Not enough values
"""
    tree = parser.parse(code)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        interpreter.execute(tree)

    # Test invalid starred unpacking
    code = """
a, *b, *c = [1, 2, 3]  # Multiple starred targets
"""
    tree = parser.parse(code)
    with pytest.raises(
        SyntaxError,
        match="multiple starred expressions in assignment",
    ):
        interpreter.execute(tree)

    # Test return outside function
    code = """
return 42  # Return outside function
"""
    tree = parser.parse(code)
    with pytest.raises(SyntaxError, match="'return' outside function"):
        interpreter.execute(tree)

    # Test break outside loop
    code = """
break  # Break outside loop
"""
    tree = parser.parse(code)
    with pytest.raises(SyntaxError, match="'break' outside loop"):
        interpreter.execute(tree)

    # Test continue outside loop
    code = """
continue  # Continue outside loop
"""
    tree = parser.parse(code)
    with pytest.raises(SyntaxError, match="'continue' outside loop"):
        interpreter.execute(tree)

    # Test invalid attribute access
    code = """
class Test:
    pass

obj = Test()
result = obj.__code__  # Forbidden attribute
"""
    tree = parser.parse(code)
    with pytest.raises(
        SecurityError, match="Access to '__code__' attribute is not allowed"
    ):
        interpreter.execute(tree)

    # Test invalid function call
    code = """
eval("1 + 1")  # Forbidden function
"""
    tree = parser.parse(code)
    with pytest.raises(
        SecurityError, match="Call to builtin 'eval' is not allowed"
    ):
        interpreter.execute(tree)

    # Test invalid import
    code = """
try:
    import os  # Import not allowed
except:
    pass
"""
    tree = parser.parse(code)
    with pytest.raises(
        SecurityError, match="Import statements are not allowed"
    ):
        interpreter.execute(tree)


def test_control_flow_edge_cases():
    """Test control flow edge cases in loops and conditionals.

    Tests:
    1. Break in nested loops
    2. Continue in nested loops
    3. While loop with break and continue
    4. For loop with else clause
    5. While loop with else clause
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test break in nested loops
    code = """
result = []
for i in range(3):
    for j in range(3):
        if i == j == 1:
            break
        result.append((i, j))
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)]

    # Test continue in nested loops
    code = """
result = []
for i in range(3):
    for j in range(3):
        if i == j:
            continue
        result.append((i, j))
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    # Test while loop with break and continue
    code = """
i = 0
result = []
while i < 5:
    i += 1
    if i == 2:
        continue
    if i == 4:
        break
    result.append(i)
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 3]

    # Test for loop with else
    code = """
result = []
for i in range(3):
    if i == 4:  # Never true
        break
    result.append(i)
    if i == 2:
        result.append('done')
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 1, 2, "done"]

    # Test while loop with else
    code = """
i = 0
result = []
while i < 3:
    result.append(i)
    i += 1
    if i == 3:
        result.append('done')
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 1, 2, "done"]


def test_interpreter_catch_generic_exception():
    """Test the interpreter's generic exception handling.

    Verifies that when a user exception occurs during execution,
    it is caught and re-raised with appropriate error information.
    Uses ZeroDivisionError as a test case.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    code = "1/0"  # Guarantee a ZeroDivisionError
    tree = parser.parse(code)
    with pytest.raises(ZeroDivisionError) as exc_info:
        interpreter.execute(tree)
    # Confirm the custom message is in the exception text
    assert "division by zero" in str(exc_info.value)


def test_nonlocal_binding_error():
    """Test nonlocal binding error handling.

    Verifies that attempting to use a nonlocal reference without a proper
    outer scope binding raises a SyntaxError.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    code = """
def outer():
    def inner():
        nonlocal x
        x = 123
    inner()
outer()
"""
    tree = parser.parse(code)
    with pytest.raises(SyntaxError, match="No binding for nonlocal 'x'"):
        interpreter.execute(tree)


def test_global_name_error():
    """Test global name error handling.

    Verifies that accessing a global name that is declared but not defined
    raises a NameError with appropriate error message.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    code = """
global x
x
"""
    # 'x' is declared global but never defined in global_env => NameError
    tree = parser.parse(code)
    with pytest.raises(NameError, match="Global name 'x' is not defined"):
        interpreter.execute(tree)


def test_nonlocal_name_error():
    """Test nonlocal name error handling.

    Verifies that attempting to use a nonlocal declaration for a name
    that doesn't exist in any outer scope raises a SyntaxError.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    code = """
def func():
    nonlocal y
"""
    tree = parser.parse(code)
    with pytest.raises(SyntaxError, match="No binding for nonlocal 'y'"):
        interpreter.execute(tree)
