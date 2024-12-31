#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
)


def test_named_expr_basic():
    """Test basic assignment and return value"""
    code = "(x := 42)"
    parser = ExpressionsParser()
    tree = parser.parse(code)
    interpreter = ExpressionsInterpreter()
    result = interpreter.execute(tree)

    assert result == 42  # Check return value
    assert interpreter.get_name_value("x") == 42  # Check assignment


def test_named_expr_in_if():
    """Test walrus in if condition"""
    code = """
if (x := 10) > 5:
    y = x * 2
    """
    parser = ExpressionsParser()
    tree = parser.parse(code)
    interpreter = ExpressionsInterpreter()
    interpreter.execute(tree)

    assert interpreter.get_name_value("x") == 10
    assert interpreter.get_name_value("y") == 20


def test_named_expr_in_while():
    """Test walrus in while loop"""
    code = """
nums = [1, 2, 3]
i = 0
while (n := nums[i]) < 3:
    i += 1
    """
    parser = ExpressionsParser()
    tree = parser.parse(code)
    interpreter = ExpressionsInterpreter()
    interpreter.execute(tree)

    # Last value checked in while condition
    assert interpreter.get_name_value("n") == 3
    # Incremented until n becomes 3
    assert interpreter.get_name_value("i") == 2


def test_named_expr_in_comprehension():
    """Test walrus in list comprehension"""
    code = """
numbers = [1, 2, 3, 4]
[y for n in numbers if (y := n * 2) > 5]
"""
    parser = ExpressionsParser()
    tree = parser.parse(code)
    interpreter = ExpressionsInterpreter()
    result = interpreter.execute(tree)

    assert result == [6, 8]  # Only values > 5
    assert interpreter.get_name_value("y") == 8  # Last assigned value


def test_named_expr_scope():
    """Test scope handling"""
    code = """
def outer():
    def inner():
        nonlocal x
        x = 200
    x = 100  # Regular assignment
    inner()
    return x
result = outer()
"""
    parser = ExpressionsParser()
    tree = parser.parse(code)
    interpreter = ExpressionsInterpreter()
    interpreter.execute(tree)

    assert interpreter.get_name_value("result") == 200


def test_named_expr_error():
    """Test invalid target"""
    code = "(1 := 42)"  # Can't assign to literal
    parser = ExpressionsParser()

    with pytest.raises(SyntaxError):
        parser.parse(code)
