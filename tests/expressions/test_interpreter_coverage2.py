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


def test_raise_statement_edge_cases_advanced():
    """Test edge cases in raise statement handling."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test raise from
    code = """
def test_raise():
    try:
        try:
            raise ValueError("inner error")
        except ValueError as e:
            raise RuntimeError("outer error") from e
    except RuntimeError as e:
        return (str(e), str(e.__cause__))

result = test_raise()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ("outer error", "inner error")

    # Test raise in except block
    code = """
def test_raise_in_except():
    error = None
    try:
        try:
            raise ValueError("original error")
        except ValueError:
            raise  # Re-raise the current exception
    except ValueError as e:
        error = str(e)
    return error

result = test_raise_in_except()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "original error"

    # Test raise without active exception
    code = """
def test_raise_no_exception():
    try:
        raise
    except RuntimeError as e:
        return str(e)

result = test_raise_no_exception()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "No active exception to re-raise"


def test_unpacking_error_cases():
    """Test error cases in sequence unpacking."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test unpacking non-iterable
    code = """
try:
    a, b = 42
except TypeError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "cannot unpack non-iterable int object"

    # Test unpacking with wrong number of values
    code = """
try:
    a, b, c = [1, 2]
except ValueError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "not enough values to unpack (expected 3, got 2)"

    # Test unpacking with too many values
    code = """
try:
    a, b = [1, 2, 3]
except ValueError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "too many values to unpack (expected 2)"


def test_attribute_access_errors():
    """Test error cases in attribute access."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test accessing non-existent attribute on a type
    code = """
error1 = None
try:
    int.nonexistent
except AttributeError as e:
    error1 = str(e)
error1
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "type object 'int' has no attribute 'nonexistent'"

    # Test accessing non-existent attribute on an instance
    code = """
error2 = None
try:
    x = 42
    x.nonexistent
except AttributeError as e:
    error2 = str(e)
error2
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "'int' object has no attribute 'nonexistent'"

    # Test accessing forbidden attributes
    code = """
class TestClass:
    pass

obj = TestClass()
error3 = None
error4 = None

try:
    x = obj.__code__
except Exception as e:
    error3 = str(e)

try:
    x = obj.__class__
except Exception as e:
    error4 = str(e)

[error3, error4]
"""
    with pytest.raises(SecurityError):
        tree = parser.parse(code)
        interpreter.execute(tree)


def test_error_handling_edge_cases():
    """Test various error handling edge cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test invalid attribute access
    code = """
class TestClass:
    pass

obj = TestClass()
try:
    obj.__code__
except SecurityError as e:
    error1 = str(e)

try:
    obj.__class__
except SecurityError as e:
    error2 = str(e)

[error1, error2]
"""
    tree = parser.parse(code)
    with pytest.raises(SecurityError):
        interpreter.execute(tree)

    # Test invalid function call
    code = """
def test_func(x):
    pass

test_func(x=1, x=2)  # Duplicate keyword argument
"""
    tree = parser.parse(code)
    with pytest.raises(
        TypeError, match="got multiple values for keyword argument 'x'"
    ):
        interpreter.execute(tree)

    # Test invalid sequence unpacking
    code = """
a, b = 1  # Cannot unpack non-iterable
"""
    tree = parser.parse(code)
    with pytest.raises(
        TypeError, match="cannot unpack non-iterable int object"
    ):
        interpreter.execute(tree)

    # Test invalid starred unpacking
    code = """
[*a, *b] = [1, 2, 3]  # Multiple starred targets
"""
    tree = parser.parse(code)
    with pytest.raises(
        SyntaxError, match="multiple starred expressions in assignment"
    ):
        interpreter.execute(tree)


def test_complex_generator_expressions():
    """Test complex generator expression scenarios."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test nested generator expressions
    code = """
def test_nested_gen():
    result = []
    gen1 = (i for i in range(3))
    gen2 = (j for j in gen1)
    gen3 = (k for k in gen2)
    result.extend(gen3)
    return result

test_nested_gen()
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 1, 2]

    # Test generator expression with multiple if conditions
    code = """
def test_multi_if_gen():
    result = []
    gen = (x for x in range(10) if x % 2 == 0 if x % 3 == 0)
    result.extend(gen)
    return result

test_multi_if_gen()
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 6]

    # Test generator expression with complex conditions
    code = """
def test_complex_gen():
    data = {'a': 1, 'b': 2, 'c': 3}
    result = []
    gen = (k.upper() + str(v) for k, v in data.items() if v > 1)
    result.extend(sorted(gen))
    return result

test_complex_gen()
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["B2", "C3"]


def test_match_statement_edge_cases():
    """Test edge cases in match statement patterns."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test match with sequence patterns and star patterns
    code = """
def match_sequence(value):
    match value:
        case [1, *rest, 2]:
            return ('pattern1', rest)
        case [*start, 3, 4]:
            return ('pattern2', start)
        case [1, 2, *starred, 5]:
            return ('pattern3', starred)
        case _:
            return ('no_match', None)

results = []
results.append(match_sequence([1, 3, 4, 2]))
results.append(match_sequence([1, 2, 3, 4]))
results.append(match_sequence([1, 2, 3, 4, 5]))
results.append(match_sequence([0, 0, 0]))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        ("pattern1", [3, 4]),
        ("pattern2", [1, 2]),
        ("pattern3", [3, 4]),
        ("no_match", None),
    ]

    # Test match with mapping patterns and extra keys
    code = """
def match_mapping(value):
    match value:
        case {'x': x, 'y': y, **rest} if x > y:
            return ('greater', rest)
        case {'x': x, 'y': y, **rest} if x < y:
            return ('lesser', rest)
        case {'x': x, 'y': y, **rest}:
            return ('equal', rest)
        case _:
            return ('invalid', None)

results = []
results.append(match_mapping({'x': 5, 'y': 3, 'z': 1}))
results.append(match_mapping({'x': 2, 'y': 4, 'z': 1}))
results.append(match_mapping({'x': 3, 'y': 3, 'z': 1}))
results.append(match_mapping({'a': 1, 'b': 2}))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        ("greater", {"z": 1}),
        ("lesser", {"z": 1}),
        ("equal", {"z": 1}),
        ("invalid", None),
    ]


def test_class_pattern_matching():
    """Test class pattern matching in match statements."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

def classify_shape(shape):
    match shape:
        case Point(x=0, y=0):
            return "origin"
        case Point(x=x, y=y) if x == y:
            return f"diagonal point at {x}"
        case Circle(center=Point(x=0, y=0), radius=r):
            return f"centered circle with radius {r}"
        case Circle(center=c, radius=r) if c.x == c.y:
            return f"diagonal circle at {c.x} with radius {r}"
        case _:
            return "other shape"

results = []
results.append(classify_shape(Point(0, 0)))
results.append(classify_shape(Point(5, 5)))
results.append(classify_shape(Circle(Point(0, 0), 10)))
results.append(classify_shape(Circle(Point(3, 3), 5)))
results.append(classify_shape(Circle(Point(1, 2), 7)))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "origin",
        "diagonal point at 5",
        "centered circle with radius 10",
        "diagonal circle at 3 with radius 5",
        "other shape",
    ]


def test_function_definition_advanced():
    """Test advanced function definition features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test function with complex parameter handling
    code = """
def complex_func(a, b=1, *args, c, d=2, **kwargs):
    return (a, b, args, c, d, kwargs)

results = []
results.append(complex_func(1, c=3))
results.append(complex_func(1, 2, 3, 4, c=5, d=6, x=7))
results.append(complex_func(1, 2, c=3, d=4, x=5, y=6))

try:
    complex_func(1)  # Missing required keyword-only argument
except TypeError as e:
    results.append(str(e))

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        (1, 1, (), 3, 2, {}),
        (1, 2, (3, 4), 5, 6, {"x": 7}),
        (1, 2, (), 3, 4, {"x": 5, "y": 6}),
        "complex_func() missing 1 required keyword-only argument: 'c'",
    ]

    # Test function with nonlocal and closure
    code = """
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c1 = make_counter()
c2 = make_counter()
results = []
results.append(c1())
results.append(c1())
results.append(c2())
results.append(c1())
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 2, 1, 3]


def test_generator_function_edge_cases():
    """Test edge cases in generator functions."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test generator with yield from
    code = """
def inner_gen():
    yield 1
    yield 2

def outer_gen():
    yield 'a'
    yield from inner_gen()
    yield 'b'

results = []
for x in outer_gen():
    results.append(x)
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["a", 1, 2, "b"]

    # Test generator with return
    code = """
def gen_with_return():
    yield 1
    yield 2
    return 'done'

results = []
try:
    g = gen_with_return()
    results.append(next(g))
    results.append(next(g))
    results.append(next(g))
except StopIteration as e:
    results.append(e.value)
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 2, "done"]


def test_with_statement_advanced():
    """Test advanced with statement features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test with statement with multiple context managers and exception handling
    code = """
class Context:
    def __init__(self, name, suppress=False):
        self.name = name
        self.suppress = suppress
        self.events = []

    def __enter__(self):
        self.events.append(f"enter {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.events.append(f"exit {self.name}")
        return self.suppress

ctx1 = Context("first", suppress=False)
ctx2 = Context("second", suppress=True)
results = []

try:
    with ctx1 as c1, ctx2 as c2:
        results.extend(ctx1.events)
        results.extend(ctx2.events)
        raise ValueError("test error")
except ValueError:
    results.append("caught error")

results.extend(ctx1.events)
results.extend(ctx2.events)
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "enter first",
        "enter second",
        "enter first",
        "exit first",
        "enter second",
        "exit second",
    ]


def test_async_function_basics():
    """Test basic async function features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test async function definition and await
    code = """
async def fetch(x):
    return x * 2

async def process():
    result = await fetch(21)
    return result

result = await process()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 42


def test_formatted_string_edge_cases():
    """Test edge cases in formatted string handling."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test formatted string with various format specifiers
    code = """
value = 42.12345
results = []
results.append(f"{value:.2f}")
results.append(f"{value:>10.1f}")
results.append(f"{value:010.3f}")
results.append(f"{value!r}")
results.append(f"{value!s}")
results.append(f"{value = :.2f}")  # Using '=' debugging syntax
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "42.12",
        "      42.1",
        "000042.123",
        "42.12345",
        "42.12345",
        "value = 42.12",
    ]

    # Test formatted string with expressions
    code = """
x, y = 10, 20
results = []
results.append(f"{x + y}")
results.append(f"{x * 2 + y:>5}")
results.append(f"{'x + y = '}{x + y}")
results.append(f"{[i for i in range(3)]}")
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == ["30", "   40", "x + y = 30", "[0, 1, 2]"]


def test_comprehension_edge_cases():
    """Test edge cases in list/set/dict comprehensions."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test nested comprehensions
    code = """
# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(3)]

# Nested set comprehension
nested_set = {(i, j) for i in range(2) for j in range(2) if i != j}

# Nested dict comprehension
nested_dict = {i: {j: i*j for j in range(2)} for i in range(2)}

[transposed, sorted(nested_set), nested_dict]
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
        [(0, 1), (1, 0)],
        {0: {0: 0, 1: 0}, 1: {0: 0, 1: 1}},
    ]

    # Test comprehensions with complex conditions
    code = """
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]

# List comprehension with complex condition
names = [d['name'] for d in data if d['age'] > 28]

# Dict comprehension with complex transformation
age_map = {d['name'].lower(): d['age'] + 1 for d in data if len(d['name']) > 3}

# Set comprehension with complex expression
letter_ages = {(name[0], age) for d in data
              if (age := d['age']) < 32 and (name := d['name'])}

[names, age_map, sorted(letter_ages)]
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        ["Bob", "Charlie"],
        {"alice": 26, "charlie": 36},
        [("A", 25), ("B", 30)],
    ]


def test_scope_edge_cases():
    """Test edge cases in scope handling."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test complex scope interactions
    code = """
def make_functions():
    funcs = []
    for i in range(3):
        def make_func(x):
            return x + i
        funcs.append(make_func)
    return funcs

f0, f1, f2 = make_functions()
results = []
results.append(f0(10))
results.append(f1(10))
results.append(f2(10))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [12, 12, 12]

    # Test nonlocal declaration errors
    code = """
def outer():
    try:
        def inner():
            nonlocal x  # Error: no binding for nonlocal 'x' found
            x = 1
        return inner()
    except SyntaxError as e:
        return str(e)

result = outer()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert "no binding for nonlocal 'x' found" in result


def test_binary_operation_edge_cases():
    """Test edge cases in binary operations."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test division by zero
    code = """
try:
    result = 1 / 0
except ZeroDivisionError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "division by zero"

    # Test floor division by zero
    code = """
try:
    result = 1 // 0
except ZeroDivisionError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "integer division or modulo by zero"

    # Test power operation with invalid operands
    code = """
try:
    result = "2" ** 3
except TypeError as e:
    error = str(e)
error
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert "unsupported operand type(s) for **" in result


def test_attribute_access_advanced():
    """Test advanced cases of attribute access."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test attribute access with property
    code = """
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

temp = Temperature(25)
results = []
results.append(temp.fahrenheit)
temp.fahrenheit = 77
results.append(temp._celsius)
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [77.0, 25.0]

    # Test attribute access with descriptors
    code = """
class Validator:
    def __init__(self, minvalue=None, maxvalue=None):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.name = None

    def __get__(self, obj, objtype):
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Value must be >= {self.minvalue}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(f"Value must be <= {self.maxvalue}")
        obj.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

class Score:
    score = Validator(0, 100)

results = []
s = Score()
s.score = 85
results.append(s.score)

try:
    s.score = -1
except ValueError as e:
    results.append(str(e))

try:
    s.score = 101
except ValueError as e:
    results.append(str(e))

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [85, "Value must be >= 0", "Value must be <= 100"]


def test_function_call_edge_cases():
    """Test edge cases in function calls."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test function call with invalid arguments
    code = """
def func(a, b, *, c):
    return a + b + c

results = []
try:
    func(1, 2, 3)  # Positional argument after keyword-only argument
except TypeError as e:
    results.append(str(e))

try:
    func(1, b=2)  # Missing required keyword-only argument
except TypeError as e:
    results.append(str(e))

try:
    func(1, 2, d=3)  # Unknown keyword argument
except TypeError as e:
    results.append(str(e))

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "func() takes 2 positional arguments but 3 were given",
        "func() missing 1 required keyword-only argument: 'c'",
        "func() got an unexpected keyword argument 'd'",
    ]

    # Test method call with invalid arguments
    code = """
class Test:
    def method(self, a, b=2):
        return a + b

    @classmethod
    def class_method(cls, x):
        return x * 2

    @staticmethod
    def static_method(y):
        return y + 1

obj = Test()
results = []

# Test instance method
results.append(obj.method(1))
results.append(obj.method(1, 3))

# Test classmethod
results.append(Test.class_method(5))
results.append(obj.class_method(5))

# Test staticmethod
results.append(Test.static_method(10))
results.append(obj.static_method(10))

try:
    obj.method()  # Missing required argument
except TypeError as e:
    results.append(str(e))

try:
    obj.method(1, 2, 3)  # Too many arguments
except TypeError as e:
    results.append(str(e))

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        3,
        4,
        10,
        10,
        11,
        11,
        "method() missing 1 required positional argument: 'a'",
        "method() takes from 2 to 3 positional arguments but 4 were given",
    ]


def test_generator_advanced_features():
    """Test advanced generator features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test generator with try/finally
    code = """
def gen():
    yield 1
    yield 2

results = []
g = gen()
results.append(next(g))
results.append(next(g))
try:
    next(g)
except StopIteration:
    results.append('stopped')
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 2, "stopped"]

    # Test generator with throw
    code = """
def gen_with_throw():
    yield 1
    yield 2
    yield 3
    yield 'end'

results = []
g = gen_with_throw()
results.append(next(g))
results.append(next(g))
try:
    g.throw(ValueError)
except ValueError:
    results.append('caught')
except StopIteration:
    results.append('stopped')
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 2, "caught"]


def test_comprehension_advanced_features():
    """Test advanced comprehension features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test nested comprehensions with complex conditions
    code = """
def get_data():
    return [[j for j in range(i + 1)] for i in range(3)]

results = []
# List comprehension
results.append([sum(row) for row in get_data()])

# Set comprehension with tuple unpacking
points = [(1, 2), (3, 4), (1, 2), (5, 6)]
results.append({(y, x) for x, y in points})

# Dict comprehension with complex key generation
words = ['hello', 'world', 'python']
results.append({word: {c: word.count(c) for c in set(word)}
               for word in words if len(word) > 4})

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        [0, 1, 3],  # Sum of each row
        {(2, 1), (4, 3), (6, 5)},  # Swapped coordinates
        {
            "hello": {"h": 1, "o": 1, "e": 1, "l": 2},
            "world": {"o": 1, "l": 1, "r": 1, "d": 1, "w": 1},
            "python": {"o": 1, "t": 1, "p": 1, "y": 1, "n": 1, "h": 1},
        },
    ]

    # Test comprehension with error handling
    code = """
def problematic_data():
    yield 1
    yield 'not a number'
    yield 3

results = []
try:
    results.append([x * 2 for x in problematic_data()])
except TypeError as e:
    results.append(str(e))

# Generator expression with error
def gen_expr():
    return (x * 2 for x in problematic_data())

g = gen_expr()
results.append(next(g))
try:
    next(g)
except TypeError as e:
    results.append(str(e))

results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [[2, "not a numbernot a number", 6], 2]


def test_match_pattern_advanced():
    """Test advanced pattern matching features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test match with custom class patterns
    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    __match_args__ = ('x', 'y')

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    __match_args__ = ('start', 'end')

def classify(shape):
    match shape:
        case Point(0, y):
            return f"Point on y-axis at y={y}"
        case Point(x, 0):
            return f"Point on x-axis at x={x}"
        case Point(x, y) if x == y:
            return f"Point on diagonal at {x}"
        case Line(Point(x1, y1), Point(x2, y2)) if x1 == x2:
            return f"Vertical line at x={x1}"
        case Line(Point(x1, y1), Point(x2, y2)) if y1 == y2:
            return f"Horizontal line at y={y1}"
        case _:
            return "Other shape"

results = []
results.append(classify(Point(0, 5)))
results.append(classify(Point(3, 0)))
results.append(classify(Point(4, 4)))
results.append(classify(Line(Point(2, 1), Point(2, 5))))
results.append(classify(Line(Point(1, 3), Point(5, 3))))
results.append(classify(Line(Point(1, 1), Point(2, 2))))
results
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [
        "Point on y-axis at y=5",
        "Point on x-axis at x=3",
        "Point on diagonal at 4",
        "Vertical line at x=2",
        "Horizontal line at y=3",
        "Other shape",
    ]


def test_async_function_advanced():
    """Test advanced async function features."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test async function with error handling
    code = """
async def the_answer():
    return 42

async def async_with_error():
    try:
        await the_answer()
        raise ValueError('test error')
    except ValueError as e:
        return str(e)

async def run_async():
    return await async_with_error()

result = await run_async()
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == "test error"
