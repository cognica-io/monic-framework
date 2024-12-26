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


@pytest.mark.parametrize(
    "source, expected",
    [
        ("True and True", True),
        ("True and False", False),
        ("False and True", False),
        ("False and False", False),
        ("1 and 2", 2),  # In Python, '1 and 2' evaluates to 2
        ("0 and 2", 0),  # '0 and 2' -> 0
        ("[] and [1,2]", []),  # An empty list is falsy, so '[] and [1,2]' -> []
    ],
)
def test_and_operator(source, expected):
    """
    Test the 'and' boolean operator.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(source)
    result = interpreter.execute(tree)
    assert result == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("True or True", True),
        ("True or False", True),
        ("False or True", True),
        ("False or False", False),
        ("1 or 2", 1),  # '1 or 2' -> 1
        ("0 or 2", 2),  # '0 or 2' -> 2
        ("[] or [1,2]", [1, 2]),  # '[] or [1,2]' -> [1,2]
    ],
)
def test_or_operator(source, expected):
    """
    Test the 'or' boolean operator.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(source)
    result = interpreter.execute(tree)
    assert result == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ("not True", False),
        ("not False", True),
        ("not 0", True),  # 'not 0' -> True, since 0 is falsy
        ("not 1", False),  # 'not 1' -> False, since 1 is truthy
        ("not []", True),  # 'not []' -> True
        ("not [1]", False),
    ],
)
def test_not_operator(source, expected):
    """
    Test the 'not' boolean operator.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(source)
    result = interpreter.execute(tree)
    assert result == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        # (True and False) or True -> False or True -> True
        ("True and False or True", True),
        # False or (True and True) -> False or True -> True
        ("False or True and True", True),
        # True or (False and True) -> True or False -> True
        ("True or False and True", True),
        # False or (True and False) -> False or False -> False
        ("False or True and False", False),
        # (1 and 2) or 3 -> 2 or 3 -> 2
        ("1 and 2 or 3", 2),
        # (0 and 2) or 3 -> 0 or 3 -> 3
        ("0 and 2 or 3", 3),
        # True or (0 and 5) -> True or 0 -> True
        ("True or 0 and 5", True),
        # [] and 'a' or 'b' -> [] or 'b' -> 'b'
        ("[] and 'a' or 'b'", "b"),
        # '' or 'hello' and 'world' -> '' or 'world' -> 'world'
        ("'' or 'hello' and 'world'", "world"),
    ],
)
def test_and_or_combinations(source, expected):
    """
    Test various combinations of 'and' and 'or'.
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()
    tree = parser.parse(source)
    result = interpreter.execute(tree)
    assert result == expected
