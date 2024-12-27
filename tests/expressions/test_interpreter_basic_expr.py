#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
)


def test_basic_constant_evaluation():
    """Test basic constant evaluation"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse("42")
    result = interpreter.execute(tree)
    assert result == 42


def test_binary_operations():
    """Test various binary operations"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Simple lambda
    tree = parser.parse("(lambda x: x * 2)(3)")
    result = interpreter.execute(tree)
    assert result == 6


def test_function_definition():
    """Test function definition and calling"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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

    # Starred unpacking with multiple variables
    tree = parser.parse("first, *middle, last = [1, 2, 3, 4, 5]")
    interpreter.execute(tree)
    assert interpreter.local_env["first"] == 1
    assert interpreter.local_env["middle"] == [2, 3, 4]
    assert interpreter.local_env["last"] == 5

    # Starred unpacking with empty middle
    tree = parser.parse("first, *middle, last = [1, 2]")
    interpreter.execute(tree)
    assert interpreter.local_env["first"] == 1
    assert interpreter.local_env["middle"] == []
    assert interpreter.local_env["last"] == 2


def test_error_handling():
    """Test error handling mechanisms"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

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


def test_nested_comprehension():
    """Test nested list/dict comprehensions"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Nested list comprehension
    code = """
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flattened = [x for row in matrix for x in row]
    transposed = [[row[i] for row in matrix] for i in range(3)]
    """
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("flattened") == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    assert interpreter.get_name_value("transposed") == [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]


def test_complex_slicing():
    """Test complex slicing operations"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    code = """
    lst = list(range(10))
    slice1 = lst[::2]  # Every second element
    slice2 = lst[::-1]  # Reverse
    slice3 = lst[1:7:2]  # Start:stop:step
    slice4 = lst[-3::-2]  # Negative indices
    """
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("slice1") == [0, 2, 4, 6, 8]
    assert interpreter.get_name_value("slice2") == [
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ]
    assert interpreter.get_name_value("slice3") == [1, 3, 5]
    assert interpreter.get_name_value("slice4") == [7, 5, 3, 1]


def test_exception_handling_details():
    """Test detailed exception handling cases"""
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test exception variable scope
    code = """
    try:
        raise ValueError("test")
    except ValueError as e:
        error_msg = str(e)

    try:
        e  # Should raise NameError
    except NameError:
        error_exists = False
    else:
        error_exists = True
    """
    interpreter.execute(parser.parse(code))
    assert interpreter.get_name_value("error_msg") == "test"
    assert interpreter.get_name_value("error_exists") is False
