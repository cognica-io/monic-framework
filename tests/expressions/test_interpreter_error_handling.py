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
        TypeError, match="cannot unpack non-iterable int object"
    ):
        interpreter.execute(parser.parse("a, b = 42"))

    # Test too many values to unpack
    with pytest.raises(
        ValueError,
        match="not enough values to unpack",
    ):
        interpreter.execute(parser.parse("a, b, c = [1, 2]"))

    # Test too few values to unpack
    with pytest.raises(
        ValueError,
        match="too many values to unpack",
    ):
        interpreter.execute(parser.parse("a, b = [1, 2, 3]"))

    # Test multiple starred expressions
    with pytest.raises(
        SyntaxError,
        match="multiple starred expressions in assignment",
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
        TypeError, match=r"f\(\) missing 1 required positional argument: 'x'"
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
        TypeError, match=r"f\(\) got an unexpected keyword argument 'x'"
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
        AttributeError, match="'list' object has no attribute '__dict__'"
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


def test_function_call_edge_cases():
    """Test edge cases in function calls.

    Tests:
    1. Calling non-callable objects
    2. Missing required arguments
    3. Invalid keyword arguments
    4. Invalid **kwargs
    5. Invalid argument types
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test calling non-callable object
    with pytest.raises(TypeError, match="'int' object is not callable"):
        interpreter.execute(parser.parse("x = 42; x()"))

    # Test missing required argument
    with pytest.raises(
        TypeError, match=r"f\(\) missing 1 required positional argument: 'x'"
    ):
        interpreter.execute(
            parser.parse(
                """
def f(x):
    return x
f()
"""
            )
        )

    # Test unexpected keyword argument
    with pytest.raises(
        TypeError, match=r"f\(\) got an unexpected keyword argument 'x'"
    ):
        interpreter.execute(
            parser.parse(
                """
def f():
    pass
f(x=1)
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
f(**[1,2,3])
"""
            )
        )

    # Test invalid argument types
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
def f(x: int):
    return x + 1
f("not an int")
"""
            )
        )


def test_attribute_error_handling():
    """Test attribute error handling.

    Tests:
    1. Access to non-existent attributes
    2. Access to attributes of None
    3. Setting attributes on immutable objects
    4. Deleting attributes
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test non-existent attribute access
    with pytest.raises(AttributeError):
        interpreter.execute(
            parser.parse(
                """
class Test:
    pass
t = Test()
t.non_existent
"""
            )
        )

    # Test attribute access on None
    with pytest.raises(AttributeError):
        interpreter.execute(
            parser.parse(
                """
x = None
x.anything
"""
            )
        )

    # Test setting attribute on immutable object
    with pytest.raises(AttributeError):
        interpreter.execute(
            parser.parse(
                """
x = 42
x.new_attr = 1
"""
            )
        )

    # Test deleting non-existent attribute
    with pytest.raises(AttributeError):
        interpreter.execute(
            parser.parse(
                """
class Test:
    pass
t = Test()
del t.non_existent
"""
            )
        )


def test_subscript_error_handling():
    """Test subscript error handling.

    Tests:
    1. Index out of range
    2. Invalid index types
    3. Invalid slice parameters
    4. Modifying immutable sequences
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test index out of range
    with pytest.raises(IndexError):
        interpreter.execute(
            parser.parse(
                """
lst = [1, 2, 3]
lst[10]
"""
            )
        )

    # Test invalid index type
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
lst = [1, 2, 3]
lst["not an index"]
"""
            )
        )

    # Test invalid slice parameters
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
lst = [1, 2, 3]
lst[1:"2"]
"""
            )
        )

    # Test modifying immutable sequence
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
t = (1, 2, 3)
t[0] = 42
"""
            )
        )


def test_scope_error_handling():
    """Test scope-related error handling.

    Tests:
    1. Undefined variables
    2. Invalid nonlocal declarations
    3. Name conflicts
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test undefined variable
    with pytest.raises(NameError):
        interpreter.execute(
            parser.parse(
                """
x = undefined_variable
"""
            )
        )

    # Test invalid nonlocal declaration
    with pytest.raises(SyntaxError):
        interpreter.execute(
            parser.parse(
                """
def f():
    def g():
        nonlocal x  # x is not in any enclosing scope
        x = 1
    g()
f()
"""
            )
        )

    # Test name conflicts
    with pytest.raises(SyntaxError):
        interpreter.execute(
            parser.parse(
                """
def f():
    x = 1
    def g():
        global x
        nonlocal x  # can't be both global and nonlocal
        x = 2
    g()
f()
"""
            )
        )


def test_operator_error_handling():
    """Test operator error handling.

    Tests:
    1. Invalid operand types
    2. Division by zero
    3. Invalid comparison
    4. Invalid boolean operations
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test invalid operand types
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
"string" + 42
"""
            )
        )

    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        interpreter.execute(
            parser.parse(
                """
1 / 0
"""
            )
        )

    # Test invalid comparison
    with pytest.raises(TypeError):
        interpreter.execute(
            parser.parse(
                """
[1, 2] < "string"
"""
            )
        )


def test_raise_error_handling():
    """Test raise error handling.

    Tests:
    1. Raising an exception
    2. Re-raising an exception
    3. Raising an exception with a cause
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test raising an exception
    with pytest.raises(ValueError):
        interpreter.execute(
            parser.parse(
                """
raise ValueError("test error")
"""
            )
        )

    # Test re-raising an exception
    with pytest.raises(ValueError):
        interpreter.execute(
            parser.parse(
                """
try:
    raise ValueError("test error")
except ValueError:
    raise
"""
            )
        )

    # Test raising an exception with a cause
    with pytest.raises(ValueError):
        interpreter.execute(
            parser.parse(
                """
try:
    raise ValueError("test error")
except ValueError as e:
    raise ValueError("test error") from e
"""
            )
        )


def test_delete_attribute_error_handling():
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    code = """
del _
"""
    with pytest.raises(SyntaxError):
        interpreter.execute(parser.parse(code))
