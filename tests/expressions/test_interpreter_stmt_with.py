#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
)


class TestWithStatement:
    """Test suite for with statement implementation in the interpreter.

    Tests various aspects of with statement behavior including scope handling,
    context manager protocol, variable lifetime, and exception handling.
    """

    @pytest.fixture
    def parser(self):
        return ExpressionsParser()

    @pytest.fixture
    def interpreter(self):
        return ExpressionsInterpreter(ExpressionsContext())

    class SimpleContextManager:
        def __init__(self):
            self.entered = False
            self.exited = False
            self.exit_args = None

        def __enter__(self):
            self.entered = True
            return "test_value"

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True
            self.exit_args = (exc_type, exc_val, exc_tb)
            return False

    def test_with_statement_scope_isolation(self, parser, interpreter):
        """Test variable scope isolation in with statements.

        Tests:
        1. Variables defined inside with block are not accessible outside
        2. Outer scope variables are accessible inside with block
        3. Variable cleanup after with block execution
        """
        cm = self.SimpleContextManager()
        interpreter.global_env["cm"] = cm

        code = """
outer = 'outer'
with cm:
    inner = 'inner'
    combined = outer + inner

result = inner  # Should raise NameError
"""
        tree = parser.parse(code)

        with pytest.raises(NameError):
            interpreter.execute(tree)

        assert "outer" in interpreter.local_env
        assert "inner" not in interpreter.local_env

    def test_with_statement_global_access(self, parser, interpreter):
        """Test global variable access within with statements.

        Tests:
        1. Global variable declaration inside with block
        2. Global variable modification inside with block
        3. Global variable persistence after with block
        """
        cm = self.SimpleContextManager()
        interpreter.global_env["cm"] = cm

        code = """
global x
x = 'global'
with cm:
    global x
    x = 'modified'
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        assert interpreter.global_env["x"] == "modified"

    def test_with_statement_nonlocal_access(self, parser, interpreter):
        """Test nonlocal variable access within with statements.

        Tests:
        1. Nonlocal variable access in nested function scope
        2. Nonlocal variable modification inside with block
        3. Nonlocal variable persistence after with block
        """
        cm = self.SimpleContextManager()
        interpreter.global_env["cm"] = cm

        code = """
def outer():
    x = 'outer'
    with cm:
        nonlocal x
        x = 'modified'
    return x

result = outer()
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        assert interpreter.get_name_value("result") == "modified"

    def test_nested_with_statements_scope(self, parser, interpreter):
        """Test scope handling in nested with statements.

        Tests:
        1. Multiple context manager nesting
        2. Variable access across nested scopes
        3. Variable cleanup after nested blocks
        """
        cm1 = self.SimpleContextManager()
        cm2 = self.SimpleContextManager()
        interpreter.global_env["cm1"] = cm1
        interpreter.global_env["cm2"] = cm2

        code = """
with cm1 as v1:
    x = 'outer'
    with cm2 as v2:
        y = 'inner'
        z = x + y
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        # After execution, variables should not be accessible
        assert "x" not in interpreter.local_env
        assert "y" not in interpreter.local_env
        assert "z" not in interpreter.local_env

    def test_with_statement_exception_scope(self, parser, interpreter):
        """Test scope handling during exception scenarios.

        Tests:
        1. Variable scope during exception handling
        2. Variable cleanup after exception
        3. Exception propagation through context manager
        """

        class ExceptionContextManager:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False  # Don't suppress the exception

        cm = ExceptionContextManager()
        interpreter.global_env["cm"] = cm

        code = """
try:
    with cm:
        x = 'inside'
        raise ValueError('test error')
except ValueError:
    pass

result = x  # Should raise NameError
"""
        tree = parser.parse(code)
        with pytest.raises(NameError):
            interpreter.execute(tree)

        assert "x" not in interpreter.local_env

    def test_multiple_context_managers_scope(self, parser, interpreter):
        """Test scope handling with multiple context managers in single with.

        Tests:
        1. Multiple context manager initialization
        2. Variable access to multiple context values
        3. Variable cleanup after with block
        """
        cm1 = self.SimpleContextManager()
        cm2 = self.SimpleContextManager()
        interpreter.global_env["cm1"] = cm1
        interpreter.global_env["cm2"] = cm2

        code = """
with cm1 as v1, cm2 as v2:
    x = v1
    y = v2
    combined = x + y
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        # Variables should not be accessible after with block
        assert "x" not in interpreter.local_env
        assert "y" not in interpreter.local_env
        assert "combined" not in interpreter.local_env

    def test_with_statement_variable_lifetime(self, parser, interpreter):
        """Test variable lifetime and visibility in with statements.

        Tests:
        1. Variable visibility before, during, and after with block
        2. Variable modification persistence
        3. Temporary variable cleanup
        """
        cm = self.SimpleContextManager()
        interpreter.global_env["cm"] = cm

        code = """
before = 'visible'
with cm:
    during = 'hidden'
    before = 'modified'
after = 'visible'
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        assert "before" in interpreter.local_env
        assert "after" in interpreter.local_env
        assert "during" not in interpreter.local_env
        assert interpreter.get_name_value("before") == "modified"

    def test_with_statement_nested_function_scope(self, parser, interpreter):
        """Test interaction between with statements and function scopes.

        Tests:
        1. Function definition inside with block
        2. Nonlocal variable access in nested function
        3. Function execution within with block context
        """
        cm = self.SimpleContextManager()
        interpreter.global_env["cm"] = cm

        code = """
def outer():
    x = 'function'
    with cm:
        def inner():
            nonlocal x
            return x
        return inner()
result = outer()
"""
        tree = parser.parse(code)
        interpreter.execute(tree)

        assert interpreter.get_name_value("result") == "function"
