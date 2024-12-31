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
