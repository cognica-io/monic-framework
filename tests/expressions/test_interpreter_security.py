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


def test_forbidden_function_call():
    """Test that calling a forbidden function raises a SecurityError."""
    parser = ExpressionParser()
    tree = parser.parse("time.sleep(1)")
    interpreter = ExpressionInterpreter()
    with pytest.raises(SecurityError):
        interpreter.execute(tree)
