#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import pytest

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
)


class TestWithStatement:
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
        """Test that variables in with block are properly scoped"""
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
        """Test accessing global variables from within with block"""
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
        """Test accessing nonlocal variables from within with block"""
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

        assert interpreter.local_env["result"] == "modified"

    def test_nested_with_statements_scope(self, parser, interpreter):
        """Test scope handling in nested with statements"""
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
        """Test scope handling when exceptions occur"""

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
        """Test scope handling with multiple context managers"""
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
        """Test that variables defined in with block don't persist"""
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
        assert interpreter.local_env["before"] == "modified"

    def test_with_statement_nested_function_scope(self, parser, interpreter):
        """Test interaction between with statement and function scopes"""
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

        assert interpreter.local_env["result"] == "function"
