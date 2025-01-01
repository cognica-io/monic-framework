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


def test_security_checks():
    """Test comprehensive security checks in the interpreter.

    Tests:
    1. Forbidden built-in function calls (eval, exec, compile, __import__)
    2. Forbidden attribute access (__code__, __globals__, __dict__)
    3. Forbidden access to __builtins__
    4. Forbidden import statements
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Forbidden function calls
    forbidden_funcs = ["eval", "exec", "compile", "__import__"]
    for func in forbidden_funcs:
        with pytest.raises(
            SecurityError, match=f"Call to builtin '{func}' is not allowed"
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

    # Forbidden builtins
    with pytest.raises(
        SecurityError, match="Access to '__builtins__' attribute is not allowed"
    ):
        tree = parser.parse("__builtins__")
        interpreter.execute(tree)

    # Forbidden import statements
    with pytest.raises(
        SecurityError, match="Import statements are not allowed"
    ):
        tree = parser.parse("import os")
        interpreter.execute(tree)


def test_forbidden_function_call():
    """Test security restrictions on specific function calls.

    Verifies that attempting to call time.sleep(), which is in the
    FORBIDDEN_NAMES list, raises a SecurityError.
    """
    parser = ExpressionsParser()
    tree = parser.parse("time.sleep(1)")
    interpreter = ExpressionsInterpreter()
    with pytest.raises(SecurityError):
        interpreter.execute(tree)


def test_forbidden_attribute_access():
    """Test forbidden attribute access in more detail.

    Tests:
    1. Access to __class__ attribute
    2. Access to __bases__ attribute
    3. Access to __subclasses__ attribute
    4. Access to __mro__ attribute
    5. Access to __qualname__ attribute
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    forbidden_attrs = [
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__qualname__",
    ]

    for attr in forbidden_attrs:
        with pytest.raises(
            SecurityError, match=f"Access to '{attr}' attribute is not allowed"
        ):
            tree = parser.parse(f"[1,2,3].{attr}")
            interpreter.execute(tree)

        with pytest.raises(
            SecurityError, match=f"Access to '{attr}' attribute is not allowed"
        ):
            tree = parser.parse(f"str.{attr}")
            interpreter.execute(tree)


def test_forbidden_module_access():
    """Test forbidden module access in more detail.

    Tests:
    1. Access to time.sleep
    2. Access to os.system
    3. Access to sys.modules
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Test time.sleep access
    with pytest.raises(
        SecurityError, match="Call to 'time.sleep' is not allowed"
    ):
        tree = parser.parse("time.sleep(1)")
        interpreter.execute(tree)

    # Test os.system access
    with pytest.raises(NameError):
        tree = parser.parse("os.system('ls')")
        interpreter.execute(tree)

    # Test sys.modules access
    with pytest.raises(NameError):
        tree = parser.parse("sys.modules")
        interpreter.execute(tree)


def test_forbidden_builtins_access():
    """Test forbidden builtins access in more detail.

    Tests:
    1. Direct access to __builtins__
    2. Indirect access through globals()
    3. Access to eval through __builtins__
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Direct access to __builtins__
    with pytest.raises(
        SecurityError, match="Access to '__builtins__' attribute is not allowed"
    ):
        tree = parser.parse("__builtins__")
        interpreter.execute(tree)

    # Indirect access through globals
    with pytest.raises(
        SecurityError, match="Call to builtin 'globals' is not allowed"
    ):
        tree = parser.parse("globals()")
        interpreter.execute(tree)

    # Access to eval through __builtins__
    with pytest.raises(
        SecurityError, match="Access to '__builtins__' attribute is not allowed"
    ):
        tree = parser.parse("__builtins__.eval('1 + 1')")
        interpreter.execute(tree)


def test_security_check_combinations():
    """Test combinations of security checks.

    Tests:
    1. Nested attribute access
    2. Function calls with forbidden attributes
    3. Complex expressions with forbidden operations
    """
    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Nested attribute access
    with pytest.raises(SecurityError):
        tree = parser.parse("(lambda x: x).__code__.__dict__")
        interpreter.execute(tree)

    # Function calls with forbidden attributes
    with pytest.raises(SecurityError):
        tree = parser.parse("getattr([], '__class__')")
        interpreter.execute(tree)

    # Complex expressions with forbidden operations
    with pytest.raises(SecurityError):
        tree = parser.parse(
            """
def get_class():
    return [].__class__

result = get_class()
"""
        )
        interpreter.execute(tree)
