#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions import ExpressionsParser, ExpressionsInterpreter


def test_complex_comprehensions():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test nested list comprehension
    result = interpreter.execute(
        parser.parse("[x + y for x in range(2) for y in range(3)]")
    )
    assert result == [0, 1, 2, 1, 2, 3]

    # Test set comprehension with condition
    result = interpreter.execute(
        parser.parse("{x for x in range(5) if x % 2 == 0}")
    )
    assert result == {0, 2, 4}

    # Test dict comprehension with complex key/value
    result = interpreter.execute(
        parser.parse("{str(x): x**2 for x in range(3)}")
    )
    assert result == {"0": 0, "1": 1, "2": 4}

    # Test comprehension with multiple if conditions
    result = interpreter.execute(
        parser.parse("[x for x in range(10) if x > 2 if x < 7 if x % 2 == 0]")
    )
    assert result == [4, 6]


def test_class_definition_complex():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test class with property and methods
    code = """
class Test:
    def __init__(self, x):
        self._x = x

    @property
    def value(self):
        return self._x

    @classmethod
    def create(cls, x):
        return cls(x)

    @staticmethod
    def helper():
        return 42

t = Test(10)
result = [t.value, Test.create(20).value, Test.helper()]
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == [10, 20, 42]

    # Test class inheritance
    code = """
class Base:
    def method(self):
        return 1

class Child(Base):
    def method(self):
        return super().method() + 1

c = Child()
result = c.method()
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == 2


def test_context_manager():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test basic context manager
    code = """
class MyContext:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

ctx = MyContext()
with ctx:
    pass
result = [ctx.entered, ctx.exited]
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == [True, True]

    # Test nested context managers
    code = """
class Counter:
    def __init__(self):
        self.count = 0

    def __enter__(self):
        self.count += 1
        return self

    def __exit__(self, *args):
        self.count -= 1
        return False

c = Counter()
with c:
    with c:
        pass
result = c.count
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == 0


def test_lambda_functions():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test lambda with closure
    code = """
def make_adder(x):
    return lambda y: x + y

add5 = make_adder(5)
result = add5(3)
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == 8

    # Test lambda with multiple arguments and defaults
    code = """
f = lambda x, y=10: x * y
result = [f(2), f(2, 3)]
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == [20, 6]


def test_match_statement():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test basic pattern matching
    code = """
def check(x):
    match x:
        case 0:
            return "zero"
        case 1:
            return "one"
        case _:
            return "other"

result = [check(0), check(1), check(2)]
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == ["zero", "one", "other"]

    # Test pattern matching with guards
    code = """
def check_type(x):
    match x:
        case str() as s if len(s) > 2:
            return "long string"
        case str():
            return "short string"
        case int():
            return "number"
        case _:
            return "other"

result = [check_type("a"), check_type("abc"), check_type(42), check_type([])]
result
"""
    result = interpreter.execute(parser.parse(code))
    assert result == ["short string", "long string", "number", "other"]
