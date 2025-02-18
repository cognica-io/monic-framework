#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
)


def test_return_at_top_level():
    """Test return statement at the top level."""
    parser = ExpressionsParser()
    context = ExpressionsContext(allow_return_at_top_level=True)
    interpreter = ExpressionsInterpreter(context)

    code = """
        return 1
    """
    tree = parser.parse(code)

    result = interpreter.execute(tree)
    assert result == 1
