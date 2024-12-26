#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionParser,
    ExpressionInterpreter,
)


def test_basic_function():
    """Test basic function definition and calling"""
    code = """
    def add(x, y):
        return x + y
    result = add(2, 3)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 5


def test_function_default_args():
    """Test function with default arguments"""
    code = """
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"
    result1 = greet("World")
    result2 = greet("Python", greeting="Hi")
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == "Hello, World!"
    assert interpreter.get_name_value("result2") == "Hi, Python!"


def test_function_keyword_only_args():
    """Test function with keyword-only arguments"""
    code = """
    def process(data, *, format="json", strict=True):
        return f"Processing {data} with {format} (strict={strict})"
    result = process("data", format="xml", strict=False)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert (
        interpreter.get_name_value("result")
        == "Processing data with xml (strict=False)"
    )


def test_function_varargs():
    """Test function with variable arguments"""
    code = """
    def sum_all(*args):
        return sum(args)
    result = sum_all(1, 2, 3, 4, 5)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 15


def test_function_kwargs():
    """Test function with keyword arguments"""
    code = """
    def format_user(**kwargs):
        parts = []
        for key, value in sorted(kwargs.items()):
            parts.append(f"{key}={value}")
        return ", ".join(parts)
    result = format_user(name="John", age=30, city="New York")
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert (
        interpreter.get_name_value("result")
        == "age=30, city=New York, name=John"
    )


def test_function_all_arg_types():
    """Test function with all types of arguments"""
    code = """
    def complex_func(a, b="default", *args, c, d="d_default", **kwargs):
        parts = [f"a={a}", f"b={b}", f"args={args}", f"c={c}", f"d={d}"]
        if kwargs:
            parts.append(f"kwargs={dict(sorted(kwargs.items()))}")
        return ", ".join(parts)
    result = complex_func(1, "custom", 3, 4, 5, c="c_val", x=10, y=20)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert (
        interpreter.get_name_value("result")
        == "a=1, b=custom, args=(3, 4, 5), c=c_val, d=d_default, kwargs={'x': 10, 'y': 20}"
    )


def test_function_recursion():
    """Test recursive function calls"""
    code = """
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)
    result = factorial(5)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 120


def test_lambda_basic():
    """Test basic lambda expression"""
    code = """
    multiply = lambda x, y: x * y
    result = multiply(3, 4)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 12


def test_lambda_with_defaults():
    """Test lambda with default arguments"""
    code = """
    greet = lambda name, greeting="Hello": f"{greeting}, {name}!"
    result1 = greet("World")
    result2 = greet("Python", greeting="Hi")
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == "Hello, World!"
    assert interpreter.get_name_value("result2") == "Hi, Python!"


def test_lambda_with_varargs():
    """Test lambda with variable arguments"""
    code = """
    sum_all = lambda *args: sum(args)
    result = sum_all(1, 2, 3, 4, 5)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 15


def test_lambda_with_closure():
    """Test lambda with closure"""
    code = """
    def make_multiplier(n):
        return lambda x: x * n
    double = make_multiplier(2)
    triple = make_multiplier(3)
    result1 = double(5)
    result2 = triple(5)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 10
    assert interpreter.get_name_value("result2") == 15


def test_function_error_cases():
    """Test various error cases in function calls"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Missing required argument
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
        def func(x, y): return x + y
        result = func(1)
        """
            )
        )

    # Too many positional arguments
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
        def func(x, y): return x + y
        result = func(1, 2, 3)
        """
            )
        )

    # Unexpected keyword argument
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
        def func(x): return x
        result = func(x=1, y=2)
        """
            )
        )

    # Missing keyword-only argument
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
        def func(*, x): return x
        result = func()
        """
            )
        )


def test_nested_function_scope():
    """Test nested function scope handling"""
    code = """
    def outer(x):
        def inner():
            return x * 2
        return inner()
    result = outer(5)
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result") == 10


def test_function_nonlocal():
    """Test nonlocal variable handling in nested functions"""
    code = """
    def counter():
        count = 0
        def increment():
            nonlocal count
            count += 1
            return count
        return increment

    inc = counter()
    result1 = inc()
    result2 = inc()
    result3 = inc()
    """
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("result1") == 1
    assert interpreter.get_name_value("result2") == 2
    assert interpreter.get_name_value("result3") == 3
