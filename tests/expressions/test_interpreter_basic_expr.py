#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionParser,
    ExpressionInterpreter,
    SecurityError,
)


def test_basic_constant_evaluation():
    """Test basic constant evaluation"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    tree = parser.parse("42")
    result = interpreter.execute(tree)
    assert result == 42


def test_binary_operations():
    """Test various binary operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    test_cases = [
        ("2 + 3", 5),
        ("10 - 4", 6),
        ("5 * 6", 30),
        ("15 / 3", 5.0),
        ("17 // 5", 3),
        ("2 ** 3", 8),
        ("17 % 5", 2),
    ]

    for expr, expected in test_cases:
        tree = parser.parse(expr)
        result = interpreter.execute(tree)
        assert result == expected, f"Failed for expression: {expr}"


def test_comparison_operations():
    """Test comparison operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    test_cases = [
        ("2 < 3", True),
        ("5 > 3", True),
        ("4 <= 4", True),
        ("6 >= 6", True),
        ("3 == 3", True),
        ("3 != 4", True),
        ("'a' in 'cat'", True),
        ("'z' not in 'cat'", True),
    ]

    for expr, expected in test_cases:
        tree = parser.parse(expr)
        result = interpreter.execute(tree)
        assert result == expected, f"Failed for expression: {expr}"


def test_list_operations():
    """Test list-related operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # List creation
    tree = parser.parse("[1, 2, 3]")
    result = interpreter.execute(tree)
    assert result == [1, 2, 3]

    # List comprehension
    tree = parser.parse("[x * 2 for x in [1, 2, 3]]")
    result = interpreter.execute(tree)
    assert result == [2, 4, 6]


def test_dict_operations():
    """Test dictionary-related operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Dict creation
    tree = parser.parse("{'a': 1, 'b': 2}")
    result = interpreter.execute(tree)
    assert result == {"a": 1, "b": 2}

    # Dict comprehension
    tree = parser.parse("{k: v * 2 for k, v in {'a': 1, 'b': 2}.items()}")
    result = interpreter.execute(tree)
    assert result == {"a": 2, "b": 4}


def test_lambda_function():
    """Test lambda function creation and execution"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Simple lambda
    tree = parser.parse("(lambda x: x * 2)(3)")
    result = interpreter.execute(tree)
    assert result == 6


def test_function_definition():
    """Test function definition and calling"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()
    code = """
def square(x):
    return x * x

square(4)
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 16


def test_list_unpacking():
    """Test list unpacking assignments"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Standard unpacking
    tree = parser.parse("a, b = [1, 2]")
    interpreter.execute(tree)
    assert interpreter.local_env["a"] == 1
    assert interpreter.local_env["b"] == 2

    # Starred unpacking
    tree = parser.parse("a, *rest = [1, 2, 3, 4]")
    interpreter.execute(tree)
    assert interpreter.local_env["a"] == 1
    assert interpreter.local_env["rest"] == [2, 3, 4]


def test_security_checks():
    """Test security-related checks"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Forbidden function calls
    forbidden_funcs = ["eval", "exec", "compile", "__import__"]
    for func in forbidden_funcs:
        with pytest.raises(
            SecurityError, match=f"Use of '{func}' is not allowed"
        ):
            tree = parser.parse(f"{func}('print(1)')")
            interpreter.execute(tree)

    # Forbidden attribute access
    forbidden_attrs = ["__code__", "__globals__", "__dict__"]
    for attr in forbidden_attrs:
        with pytest.raises(
            SecurityError, match=f"Access to '{attr}' attribute is not allowed"
        ):
            tree = parser.parse(f"[1,2,3].{attr}")
            interpreter.execute(tree)


def test_error_handling():
    """Test error handling mechanisms"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Division by zero
    with pytest.raises(ZeroDivisionError):
        tree = parser.parse("1 / 0")
        interpreter.execute(tree)

    # Undefined variable
    with pytest.raises(NameError):
        tree = parser.parse("undefined_var")
        interpreter.execute(tree)


def test_augmented_assignments():
    """Test augmented assignment operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Simple augmented assignment
    tree = parser.parse("x = 5; x += 3")
    interpreter.execute(tree)
    assert interpreter.local_env["x"] == 8

    # Augmented assignment with list
    tree = parser.parse("lst = [1, 2, 3]; lst[1] += 10")
    interpreter.execute(tree)
    assert interpreter.local_env["lst"] == [1, 12, 3]


def test_f_string_formatting():
    """Test f-string formatting capabilities"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Basic f-string
    tree = parser.parse("x = 42; f'The answer is {x}'")
    result = interpreter.execute(tree)
    assert result == "The answer is 42"

    # F-string with conversion
    tree = parser.parse("x = [1, 2, 3]; f'Repr: {x!r}'")
    result = interpreter.execute(tree)
    assert result == f"Repr: {repr([1, 2, 3])}"


def test_slice_operations():
    """Test slice operations"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Basic slicing
    tree = parser.parse("lst = [0, 1, 2, 3, 4]; lst[1:4]")
    result = interpreter.execute(tree)
    assert result == [1, 2, 3]

    # Slice with step
    tree = parser.parse("lst = [0, 1, 2, 3, 4]; lst[::2]")
    result = interpreter.execute(tree)
    assert result == [0, 2, 4]


def test_try_except_handling():
    """Test try-except block handling"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Basic try-except
    code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    x = 42
x
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == 42


def test_break_statement():
    """Test break statement in loops"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Break in for loop
    code = """
result = []
for i in range(10):
    if i == 5:
        break
    result.append(i)
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 1, 2, 3, 4]

    # Break in while loop
    code = """
result = []
i = 0
while True:
    if i == 5:
        break
    result.append(i)
    i += 1
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [0, 1, 2, 3, 4]


def test_continue_statement():
    """Test continue statement in loops"""
    parser = ExpressionParser()
    interpreter = ExpressionInterpreter()

    # Continue in for loop
    code = """
result = []
for i in range(10):
    if i % 2 == 0:
        continue
    result.append(i)
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 3, 5, 7, 9]

    # Continue in while loop
    code = """
result = []
i = 0
while i < 10:
    i += 1
    if i % 2 == 0:
        continue
    result.append(i)
result
"""
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    assert result == [1, 3, 5, 7, 9]
