#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
)


def test_error_handling_paths():
    """Test error handling paths."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test complex error scenarios
    code = """
def test_func():
    try:
        raise ValueError("test")
    except ValueError as e:
        try:
            raise TypeError("nested") from e
        except TypeError:
            raise RuntimeError("complex")
        except Exception as e:
            raise e
    except Exception as e:
        raise e from None
    finally:
        return "done"

result = test_func()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == "done"

    # Test loop error handling
    code = """
result = []
try:
    for i in range(3):
        try:
            if i == 1:
                raise ValueError("test")
            result.append(i)
        except ValueError:
            continue
        else:
            result.append(f"else{i}")
except Exception as e:
    result.append("outer")
else:
    result.append("no error")
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == [
        0,
        "else0",
        2,
        "else2",
        "no error",
    ]

    # Test multiple exception handlers with chained exceptions
    code = """
def test_exc():
    try:
        raise ValueError("original")
    except ValueError as e:
        try:
            raise TypeError("converted")
        except TypeError as te:
            raise RuntimeError("chained") from te

try:
    test_exc()
except RuntimeError as e:
    result = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == "chained"

    # Test complex parameter handling with defaults and unpacking
    code = """
def complex_func(a, b=2, *args, c=3, **kwargs):
    result = []
    result.append(a)
    result.append(b)
    result.extend(args)
    result.append(c)
    result.extend(sorted(kwargs.items()))
    return result

test_args = [4, 5]
test_kwargs = {"d": 6, "e": 7}
result1 = complex_func(1)
result2 = complex_func(1, 3, *test_args, c=6, **test_kwargs)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == [1, 2, 3]
    assert interpreter.get_name_value("result2") == [
        1,
        3,
        4,
        5,
        6,
        ("d", 6),
        ("e", 7),
    ]

    # Test control flow with break/continue in nested loops and error handling
    code = """
def test_flow():
    result = []
    x = 0
    while x < 3:
        try:
            if x == 1:
                raise ValueError("test")
            for y in range(2):
                if x + y == 2:
                    break
                result.append((x, y))
                for z in range(2):
                    if z == 1:
                        continue
                    result.append((x, y, z))
        except ValueError:
            x += 1
            continue
        x += 1
    return result

result = test_flow()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == [
        (0, 0),
        (0, 0, 0),
        (0, 1),
        (0, 1, 0),
    ]

    # Test class definition with properties, inheritance and error handling
    code = """
class Base:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if val < 0:
            raise ValueError("Value cannot be negative")
        self._value = val

class Child(Base):
    def __init__(self):
        super().__init__()
        self.extra = 1

    @property
    def combined(self):
        return self.value + self.extra

obj = Child()
try:
    obj.value = -1
except ValueError:
    obj.value = 5
result1 = obj.value
result2 = obj.combined
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 5
    assert interpreter.get_name_value("result2") == 6


def test_function_edge_cases():
    """Test function-related edge cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test complex parameter handling
    code = """
def func(a, b=2, *args, c=3, **kwargs):
    return (a, b, args, c, kwargs)

result1 = func(1)
result2 = func(1, 3, 4, 5, c=6, d=7)
result3 = func(1, c=4, d=5)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == (1, 2, (), 3, {})
    assert interpreter.get_name_value("result2") == (1, 3, (4, 5), 6, {"d": 7})
    assert interpreter.get_name_value("result3") == (1, 2, (), 4, {"d": 5})

    # Test nested function definitions
    code = """
def outer(x):
    def middle(y):
        def inner(z):
            return x + y + z
        return inner
    return middle

f = outer(1)(2)(3)
result = f
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 6


def test_control_flow_cases():
    """Test control flow related code."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test complex control flow
    code = """
def test_flow():
    result = []
    x = 0
    while x < 3:
        if x == 1:
            x += 1
            continue
        for y in range(2):
            if x + y == 2:
                break
            result.append((x, y))
        x += 1
    return result

result = test_flow()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == [(0, 0), (0, 1)]

    # Test nested loops with break/continue
    code = """
def complex_loop():
    result = []
    for i in range(3):
        found = False
        for j in range(3):
            if i == j:
                found = True
                break
            result.append((i, j))
        if found:
            continue
        result.append(f"end {i}")
    return result

result = complex_loop()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == [(1, 0), (2, 0), (2, 1)]


def test_object_class_handling():
    """Test object and class related code."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test class attribute handling
    code = """
class TestClass:
    class_var = 1

    def __init__(self):
        self.instance_var = 2

    @property
    def prop(self):
        return self.instance_var * 2

    @classmethod
    def class_method(cls):
        return cls.class_var * 3

obj = TestClass()
result1 = obj.instance_var
result2 = obj.prop
result3 = TestClass.class_method()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 2
    assert interpreter.get_name_value("result2") == 4
    assert interpreter.get_name_value("result3") == 3

    # Test complex class inheritance
    code = """
class Base:
    def method(self):
        return 1

class Child(Base):
    def method(self):
        return super().method() + 1

class GrandChild(Child):
    def method(self):
        return super().method() + 1

obj = GrandChild()
result = obj.method()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 3

    # Test pattern matching with class patterns
    code = """
class Point:
    __match_args__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

def classify_point(point):
    match point:
        case Point(0, 0):
            return "origin"
        case Point(x, 0):
            return f"x-axis at {x}"
        case Point(0, y):
            return f"y-axis at {y}"
        case Point(x, y) if x == y:
            return f"diagonal at {x}"
        case Point(x, y):
            return f"point at {x}, {y}"
        case _:
            return "not a point"

result1 = classify_point(Point(0, 0))
result2 = classify_point(Point(5, 0))
result3 = classify_point(Point(0, 3))
result4 = classify_point(Point(2, 2))
result5 = classify_point(Point(1, 4))
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == "origin"
    assert interpreter.get_name_value("result2") == "x-axis at 5"
    assert interpreter.get_name_value("result3") == "y-axis at 3"
    assert interpreter.get_name_value("result4") == "diagonal at 2"
    assert interpreter.get_name_value("result5") == "point at 1, 4"

    # Test pattern matching with sequence and mapping patterns
    code = """
def analyze_data(data):
    match data:
        case {"type": "point", "coords": [x, y, *rest]}:
            return f"2D point at {x}, {y} with {len(rest)} extra dims"
        case {"type": "line", "points": [start, end]}:
            return f"line from {start} to {end}"
        case [x, y, z]:
            return f"3D point at {x}, {y}, {z}"
        case [head, *tail]:
            return f"sequence with head {head} and {len(tail)} more items"
        case _:
            return "unknown"

result1 = analyze_data({"type": "point", "coords": [1, 2, 3, 4]})
result2 = analyze_data({"type": "line", "points": ["A", "B"]})
result3 = analyze_data([1, 2, 3])
result4 = analyze_data([10, 20, 30, 40])
"""
    interpreter.execute(parser.parse(code))
    assert (
        interpreter.get_name_value("result1")
        == "2D point at 1, 2 with 2 extra dims"
    )
    assert interpreter.get_name_value("result2") == "line from A to B"
    assert interpreter.get_name_value("result3") == "3D point at 1, 2, 3"
    assert (
        interpreter.get_name_value("result4")
        == "sequence with head 10 and 3 more items"
    )

    # Test scope declarations and error handling
    code = """
x = 1
y = 2

def outer():
    global x
    nonlocal_var = 10

    def inner():
        nonlocal nonlocal_var
        global y

        nonlocal_var += 1
        x += 1
        y += 1

        return (x, y, nonlocal_var)

    return inner()

try:
    result = outer()
except SyntaxError as e:
    result = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == (2, 3, 11)


def test_sequence_unpacking():
    """Test sequence unpacking and assignment."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test basic sequence unpacking
    code = """
# Test basic unpacking
a, b = [1, 2]
result1 = (a, b)

# Test starred unpacking
first, *middle, last = [1, 2, 3, 4, 5]
result2 = (first, middle, last)

# Test not enough values
try:
    x, y, z = [1, 2]
except ValueError as e:
    error1 = str(e)

# Test too many values
try:
    x, y = [1, 2, 3]
except ValueError as e:
    error2 = str(e)

# Test multiple starred expressions
try:
    *x, *y = [1, 2, 3]
except SyntaxError as e:
    error3 = str(e)

# Test unpacking non-iterable
try:
    a, b = 42
except TypeError as e:
    error4 = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == (1, 2)
    assert interpreter.get_name_value("result2") == (1, [2, 3, 4], 5)
    assert (
        interpreter.get_name_value("error1")
        == "not enough values to unpack (expected 3, got 2)"
    )
    assert (
        interpreter.get_name_value("error2")
        == "too many values to unpack (expected 2)"
    )
    assert (
        interpreter.get_name_value("error3")
        == "multiple starred expressions in assignment"
    )
    assert (
        interpreter.get_name_value("error4")
        == "cannot unpack non-iterable int object"
    )

    # Test augmented assignment
    code = """
# Test basic augmented assignment
x = 5
x += 3
result1 = x

# Test augmented assignment with attributes
class Test:
    def __init__(self):
        self.value = 10

obj = Test()
obj.value += 5
result2 = obj.value

# Test augmented assignment with subscripts
lst = [1, 2, 3]
lst[1] += 5
result3 = lst

# Test augmented assignment with unsupported operator
try:
    x @= 2
except NotImplementedError as e:
    error1 = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 8
    assert interpreter.get_name_value("result2") == 15
    assert interpreter.get_name_value("result3") == [1, 7, 3]
    assert (
        "Unsupported augmented assignment operator"
        in interpreter.get_name_value("error1")
    )


def test_error_handling_edge_cases():
    """Test error handling edge cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    code = """
def test_nested_errors():
    result = []
    try:
        for i in range(3):
            try:
                if i == 0:
                    raise ValueError("first")
                elif i == 1:
                    raise TypeError("second")
                result.append(i)
            except ValueError:
                result.append("caught ValueError")
            except TypeError:
                result.append("caught TypeError")
            finally:
                result.append(f"finally {i}")
    except Exception as e:
        result.append(f"outer {str(e)}")
    else:
        result.append("no outer error")
    return result

result = test_nested_errors()
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == [
        "caught ValueError",
        "finally 0",
        "caught TypeError",
        "finally 1",
        2,
        "finally 2",
        "no outer error",
    ]


def test_operator_handling():
    """Test operator handling and error cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test unary operators
    code = """
# Test all unary operators
result1 = +42
result2 = -42
result3 = not True
result4 = ~5

# Test unsupported unary operator
try:
    class BadUnaryOp:
        def __init__(self):
            pass
    bad = BadUnaryOp()
    result = +bad
except TypeError as e:
    error1 = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 42
    assert interpreter.get_name_value("result2") == -42
    assert interpreter.get_name_value("result3") is False
    assert interpreter.get_name_value("result4") == -6
    assert "bad operand type" in interpreter.get_name_value("error1").lower()

    # Test binary operators and error handling
    code = """
# Test binary operators
result1 = 5 + 3
result2 = 10 - 4
result3 = 3 * 4
result4 = 15 / 3
result5 = 17 // 5
result6 = 7 % 4
result7 = 2 ** 3

# Test division by zero
try:
    result = 1 / 0
except ZeroDivisionError as e:
    error1 = str(e)

# Test invalid operand types
try:
    result = "hello" - 42
except TypeError as e:
    error2 = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 8
    assert interpreter.get_name_value("result2") == 6
    assert interpreter.get_name_value("result3") == 12
    assert interpreter.get_name_value("result4") == 5.0
    assert interpreter.get_name_value("result5") == 3
    assert interpreter.get_name_value("result6") == 3
    assert interpreter.get_name_value("result7") == 8
    assert "division by zero" in interpreter.get_name_value("error1").lower()
    assert "unsupported operand" in interpreter.get_name_value("error2").lower()


def test_super_handling():
    """Test super() handling and error cases."""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    code = """
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return super().method() + "B"

class C(B):
    def method(self):
        return super().method() + "C"

# Test normal super() usage
obj = C()
result1 = obj.method()

# Test super() with explicit arguments
result2 = super(B, obj).method()

# Test super() without arguments in instance method
class D(A):
    def method(self):
        return super().method() + "D"

obj2 = D()
result3 = obj2.method()

# Test super() error cases
try:
    super()
except RuntimeError as e:
    error1 = str(e)

try:
    super(None, None)
except TypeError as e:
    error2 = str(e)
"""
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == "ABC"
    assert interpreter.get_name_value("result2") == "A"
    assert interpreter.get_name_value("result3") == "AD"
    assert "super(): __class__ cell not found" in interpreter.get_name_value(
        "error1"
    )
    assert "super() argument 1 must be" in interpreter.get_name_value("error2")
