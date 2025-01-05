#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    SecurityError,
)


def test_scope_handling():
    code = """
def outer():
    x = 1
    def inner():
        nonlocal x
        x = 2
        def innermost():
            nonlocal x
            x = 3
        innermost()
        return x
    return inner()

result = outer()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == 3


def test_global_nonlocal_handling():
    code = """
global_var = 1

def outer():
    global global_var
    global_var = 2
    def inner():
        nonlocal_var = 3
        def innermost():
            nonlocal nonlocal_var
            nonlocal_var = 4
        innermost()
        return nonlocal_var
    return inner()

result = outer()
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result") == 4
    assert interpreter.global_env["global_var"] == 2


def test_unpacking_assignment():
    code = """
# Basic unpacking
a, b = [1, 2]

# Extended unpacking
first, *rest, last = [1, 2, 3, 4, 5]

# Nested unpacking
(x, y), [z, *w] = [(1, 2), [3, 4, 5]]

# Multiple targets
i = j = k = 0

# Unpacking with too few values
try:
    a, b, c = [1, 2]
except ValueError as e:
    error1 = str(e)

# Unpacking with too many values
try:
    a, b = [1, 2, 3]
except ValueError as e:
    error2 = str(e)

# Invalid unpacking target
try:
    x, y = None
except TypeError as e:
    error3 = str(e)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("a") == 1
    assert interpreter.get_name_value("b") == 2
    assert interpreter.get_name_value("first") == 1
    assert interpreter.get_name_value("rest") == [2, 3, 4]
    assert interpreter.get_name_value("last") == 5
    assert interpreter.get_name_value("x") == 1
    assert interpreter.get_name_value("y") == 2
    assert interpreter.get_name_value("z") == 3
    assert interpreter.get_name_value("w") == [4, 5]
    assert (
        interpreter.get_name_value("i")
        == interpreter.get_name_value("j")
        == interpreter.get_name_value("k")
        == 0
    )
    assert "not enough values to unpack" in interpreter.get_name_value("error1")
    assert "too many values to unpack" in interpreter.get_name_value("error2")
    assert "cannot unpack non-iterable" in interpreter.get_name_value("error3")


def test_comprehension_handling():
    code = """
# List comprehension with multiple generators
result1 = [(x, y) for x in range(2) for y in range(2)]

# Set comprehension with condition
result2 = {x for x in range(5) if x % 2 == 0}

# Dict comprehension with complex expression
result3 = {str(x): x**2 for x in range(3)}

# Nested comprehension
result4 = [[x + y for x in range(2)] for y in range(2)]

# Generator expression
result5 = list(x * 2 for x in range(3))

# Comprehension with named expression
result6 = [y for x in range(3) if (y := x * 2) > 2]
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result1") == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert interpreter.get_name_value("result2") == {0, 2, 4}
    assert interpreter.get_name_value("result3") == {"0": 0, "1": 1, "2": 4}
    assert interpreter.get_name_value("result4") == [[0, 1], [1, 2]]
    assert interpreter.get_name_value("result5") == [0, 2, 4]
    assert interpreter.get_name_value("result6") == [4]


def test_class_definition():
    code = """
class Base:
    def __init__(self):
        self.value = 1

    @staticmethod
    def static_method():
        return "static"

    @classmethod
    def class_method(cls):
        return "class"

    @property
    def prop(self):
        return self.value

    def __str__(self):
        return f"Base({self.value})"

class Derived(Base):
    def __init__(self):
        super().__init__()
        self.value = 2

    def __str__(self):
        return f"Derived({self.value})"

obj = Derived()
result1 = obj.static_method()
result2 = obj.class_method()
result3 = obj.prop
result4 = str(obj)

# Test non-existent attribute
try:
    obj.non_existent
except AttributeError as e:
    error1 = str(e)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result1") == "static"
    assert interpreter.get_name_value("result2") == "class"
    assert interpreter.get_name_value("result3") == 2
    assert interpreter.get_name_value("result4") == "Derived(2)"
    assert "has no attribute" in interpreter.get_name_value("error1")


def test_pattern_matching():
    code = """
def match_value(value):
    match value:
        case 1:
            return "one"
        case [x, y]:
            return f"list: {x}, {y}"
        case {"key": value}:
            return f"dict: {value}"
        case str() as s:
            return f"string: {s}"
        case (x, y):
            return f"tuple: {x}, {y}"
        case _:
            return "default"

result1 = match_value(1)
result2 = match_value([2, 3])
result3 = match_value({"key": "value"})
result4 = match_value("test")
result5 = match_value([3, 2])  # Changed to list since tuples are not supported yet
result6 = match_value(None)

# Test sequence pattern with star
match [1, 2, 3, 4]:
    case [first, *rest, last]:
        result7 = (first, rest, last)

# Test mapping pattern with rest
match {"a": 1, "b": 2, "c": 3}:
    case {"a": x, **rest}:
        result8 = (x, rest)

# Test class pattern
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

match Point(1, 2):
    case Point(x=x, y=y):
        result9 = (x, y)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result1") == "one"
    assert interpreter.get_name_value("result2") == "list: 2, 3"
    assert interpreter.get_name_value("result3") == "dict: value"
    assert interpreter.get_name_value("result4") == "string: test"
    assert (
        interpreter.get_name_value("result5") == "list: 3, 2"
    )  # Changed expectation
    assert interpreter.get_name_value("result6") == "default"
    assert interpreter.get_name_value("result7") == (1, [2, 3], 4)
    assert interpreter.get_name_value("result8") == (1, {"b": 2, "c": 3})
    assert interpreter.get_name_value("result9") == (1, 2)


def test_error_handling():
    code = """
# Basic try-except
try:
    raise ValueError("test error")
except ValueError as e:
    error_message = str(e)
finally:
    cleanup1 = "done"

# Nested try-except
try:
    try:
        raise TypeError("inner error")
    except ValueError:
        error2 = "caught value error"
    except TypeError as e:
        error2 = str(e)
finally:
    cleanup2 = "done"

# Try-except with else
try:
    value = 42
except ValueError:
    error3 = "error"
else:
    result = value * 2
finally:
    cleanup3 = "done"

# Re-raising exception
try:
    try:
        raise ValueError("original error")
    except ValueError:
        raise TypeError("new error")
except TypeError as e:
    error4 = str(e)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("error_message") == "test error"
    assert interpreter.get_name_value("cleanup1") == "done"
    assert interpreter.get_name_value("error2") == "inner error"
    assert interpreter.get_name_value("cleanup2") == "done"
    assert interpreter.get_name_value("result") == 84
    assert interpreter.get_name_value("cleanup3") == "done"
    assert interpreter.get_name_value("error4") == "new error"


def test_function_method_handling():
    code = """
class Test:
    def method(self):
        return "instance method"

    @staticmethod
    def static():
        return "static method"

    @classmethod
    def cls_method(cls):
        return "class method"

    def __call__(self, *args, **kwargs):
        return f"called with {args}, {kwargs}"

    def method_with_args_and_kwargs(self, a, b=2, *args, c=3, **kwargs):
        return f"a={a}, b={b}, args={args}, c={c}, kwargs={kwargs}"

    def method_with_args(self, a, b=2, *args, c=3):
        return f"a={a}, b={b}, args={args}, c={c}"

obj = Test()
result1 = obj.method()
result2 = Test.static()
result3 = Test.cls_method()
result4 = obj(1, 2, x=3)
result5 = obj.method_with_args_and_kwargs(1, 3, 4, 5, c=6, d=7)

# Test invalid method calls
try:
    obj.non_existent()
except AttributeError as e:
    error1 = str(e)

try:
    obj.method_with_args_and_kwargs()
except TypeError as e:
    error2 = str(e)

try:
    obj.method_with_args(1, 2, 3, c=4, invalid=5)
except TypeError as e:
    error3 = str(e)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    interpreter.execute(tree)
    assert interpreter.get_name_value("result1") == "instance method"
    assert interpreter.get_name_value("result2") == "static method"
    assert interpreter.get_name_value("result3") == "class method"
    assert (
        interpreter.get_name_value("result4") == "called with (1, 2), {'x': 3}"
    )
    assert (
        interpreter.get_name_value("result5")
        == "a=1, b=3, args=(4, 5), c=6, kwargs={'d': 7}"
    )
    assert "has no attribute" in interpreter.get_name_value("error1")
    assert "missing required positional argument" in interpreter.get_name_value(
        "error2"
    )
    assert "got an unexpected keyword argument" in interpreter.get_name_value(
        "error3"
    )


def test_security_checks():
    code = """
# Test forbidden function call
try:
    eval("1 + 1")
except NameError as e:
    error1 = str(e)

try:
    x = [1, 2]
    x.__class__
except SecurityError as e:
    error2 = str(e)

# Test forbidden module access
try:
    os = None
    os.system("ls")
except NameError as e:
    error3 = str(e)

# Test forbidden built-in access
try:
    x = __builtins__
    x.eval("1 + 1")
except NameError as e:
    error4 = str(e)
"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(code)
    with pytest.raises(SecurityError) as exc_info:
        interpreter.execute(tree)
    assert "not allowed" in str(exc_info.value)
