#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsParser,
    ExpressionsInterpreter,
    SecurityError,
    UnsupportedUnpackingError,
)


def test_forbidden_function_calls():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test eval
    with pytest.raises(
        SecurityError, match="Call to builtin 'eval' is not allowed"
    ):
        interpreter.execute(parser.parse("eval('1 + 1')"))

    # Test exec
    with pytest.raises(
        SecurityError, match="Call to builtin 'exec' is not allowed"
    ):
        interpreter.execute(parser.parse("exec('x = 1')"))

    # Test compile
    with pytest.raises(
        SecurityError, match="Call to builtin 'compile' is not allowed"
    ):
        interpreter.execute(
            parser.parse("compile('x = 1', '<string>', 'exec')")
        )


def test_unpacking_errors():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test unpacking non-iterable
    with pytest.raises(
        UnsupportedUnpackingError, match="Cannot unpack non-iterable value"
    ):
        interpreter.execute(parser.parse("a, b = 42"))

    # Test too many values to unpack
    with pytest.raises(
        UnsupportedUnpackingError,
        match="Not enough values to unpack",
    ):
        interpreter.execute(parser.parse("a, b, c = [1, 2]"))

    # Test too few values to unpack
    with pytest.raises(
        UnsupportedUnpackingError,
        match="Too many values to unpack",
    ):
        interpreter.execute(parser.parse("a, b = [1, 2, 3]"))

    # Test multiple starred expressions
    with pytest.raises(
        UnsupportedUnpackingError,
        match="Cannot use multiple starred expressions in assignment",
    ):
        interpreter.execute(parser.parse("*a, *b = [1, 2, 3]"))


def test_function_call_errors():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test calling non-callable
    with pytest.raises(TypeError, match="'int' object is not callable"):
        interpreter.execute(parser.parse("x = 42; x()"))

    # Test missing required positional argument
    with pytest.raises(
        TypeError, match="missing required positional argument: 'x'"
    ):
        interpreter.execute(
            parser.parse(
                """
def f(x):
    return x
result = f()
"""
            )
        )

    # Test unexpected keyword argument
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument: 'x'"
    ):
        interpreter.execute(
            parser.parse(
                """
def f():
    pass
result = f(x=1)
"""
            )
        )

    # Test **kwargs with non-dict
    with pytest.raises(TypeError, match="Argument after \\*\\* must be a dict"):
        interpreter.execute(
            parser.parse(
                """
def f():
    pass
result = f(**[1,2,3])
"""
            )
        )


def test_control_flow_errors():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test break outside loop
    with pytest.raises(SyntaxError, match="'break' outside loop"):
        interpreter.execute(parser.parse("break"))

    # Test continue outside loop
    with pytest.raises(SyntaxError, match="'continue' outside loop"):
        interpreter.execute(parser.parse("continue"))

    # Test return outside function
    with pytest.raises(SyntaxError, match="'return' outside function"):
        interpreter.execute(parser.parse("return 42"))


def test_attribute_access_errors():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test forbidden attribute access
    with pytest.raises(
        SecurityError, match="Access to '__dict__' attribute is not allowed"
    ):
        interpreter.execute(parser.parse("[].__dict__"))

    with pytest.raises(
        SecurityError, match="Access to '__class__' attribute is not allowed"
    ):
        interpreter.execute(parser.parse("[].__class__"))

    with pytest.raises(
        SecurityError, match="Access to '__bases__' attribute is not allowed"
    ):
        interpreter.execute(parser.parse("object.__bases__"))


def test_import_errors():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test direct import
    with pytest.raises(
        SecurityError, match="Import statements are not allowed"
    ):
        interpreter.execute(parser.parse("import os"))

    # Test from import
    with pytest.raises(
        SecurityError, match="Import statements are not allowed"
    ):
        interpreter.execute(parser.parse("from os import path"))
