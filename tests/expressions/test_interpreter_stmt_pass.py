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


def test_pass_standalone():
    """Test pass statement as a standalone statement."""
    parser = ExpressionsParser()
    context = ExpressionsContext()
    interpreter = ExpressionsInterpreter(context)

    pass_code = parser.parse(
        """
pass
"""
    )

    # Should execute without raising any exceptions
    interpreter.execute(pass_code)


def test_pass_in_function():
    """Test pass statement inside a function definition."""
    parser = ExpressionsParser()
    context = ExpressionsContext()
    interpreter = ExpressionsInterpreter(context)

    pass_in_function_code = parser.parse(
        """
def dummy_function():
    pass

dummy_function()
"""
    )

    # Should define and call the function without any issues
    interpreter.execute(pass_in_function_code)


def test_pass_in_conditional():
    """Test pass statement in an if-else block."""
    parser = ExpressionsParser()
    context = ExpressionsContext()
    interpreter = ExpressionsInterpreter(context)

    pass_in_conditional_code = parser.parse(
        """
x = 10
if x > 5:
    pass
else:
    x = 0
x
"""
    )

    # Execute the code
    result = interpreter.execute(pass_in_conditional_code)

    # Verify that x remains 10 after the pass statement
    assert result == 10


def test_pass_in_loop():
    """Test pass statement in a loop."""
    parser = ExpressionsParser()
    context = ExpressionsContext()
    interpreter = ExpressionsInterpreter(context)

    pass_in_loop_code = parser.parse(
        """
result = []
for i in range(3):
    if i == 1:
        pass
    result.append(i)
result
"""
    )

    # Execute the code
    result = interpreter.execute(pass_in_loop_code)

    # Verify the result list
    assert result == [0, 1, 2]


def test_multiple_pass_statements():
    """Test multiple pass statements in sequence."""
    parser = ExpressionsParser()
    context = ExpressionsContext()
    interpreter = ExpressionsInterpreter(context)

    multiple_pass_code = parser.parse(
        """
pass
x = 5
pass
y = x + 1
pass
"""
    )

    # Execute the code
    interpreter.execute(multiple_pass_code)

    # Verify that variable assignments still work
    assert interpreter.get_name_value("x") == 5
    assert interpreter.get_name_value("y") == 6
