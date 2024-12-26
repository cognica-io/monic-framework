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
)


def test_forbidden_function_call():
    """Test that calling a forbidden function raises a SecurityError."""
    parser = ExpressionsParser()
    tree = parser.parse("time.sleep(1)")
    interpreter = ExpressionsInterpreter()
    with pytest.raises(SecurityError):
        interpreter.execute(tree)
