#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import pytest

from monic.expressions import ExpressionsParser, ExpressionsInterpreter


def test_awaitable_value():
    """Test that AwaitableValue.__await__ works correctly."""
    # Create a simple async function that returns a value
    code = """
async def test_func():
    return 42

await test_func()
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Execute the code
    result = interpreter.execute(parser.parse(code))

    # The result should be 42
    assert result == 42


def test_awaitable_value_with_error():
    """Test that AwaitableValue.__await__ handles errors correctly."""
    # Create an async function that raises an error
    code = """
async def test_func():
    raise ValueError("test error")

await test_func()
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # The execution should raise ValueError
    with pytest.raises(ValueError, match="test error"):
        interpreter.execute(parser.parse(code))


def test_awaitable_value_return():
    """Test that AwaitableValue.__await__ returns the correct value."""
    # Create an async function that returns a value through StopIteration
    code = """
async def test_func():
    # This will make __await__ return the value through StopIteration
    return await test_func() if False else 42

await test_func()
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Execute the code
    result = interpreter.execute(parser.parse(code))

    # The result should be 42
    assert result == 42


def test_awaitable_value_direct_return():
    """Test that AwaitableValue.__await__ directly returns its value."""
    # Create an async function that returns a value directly
    code = """
async def test_func():
    # This will make __await__ directly return its value
    return 42

# Call the function multiple times to ensure the value is returned correctly
result1 = await test_func()
result2 = await test_func()
result1 + result2  # This should be 84
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Execute the code
    result = interpreter.execute(parser.parse(code))

    # The result should be 84 (42 + 42)
    assert result == 84


def test_awaitable_value_direct():
    """
    Test that AwaitableValue.__await__ directly returns its value by creating
    an awaitable value directly.
    """
    code = """
async def create_awaitable():
    # This will create an awaitable value that will be awaited directly
    return 42

# Get the awaitable value
awaitable = create_awaitable()
# Await it directly
result1 = await awaitable
# Await it again to ensure it works multiple times
result2 = await awaitable
result1 + result2  # This should be 84
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Execute the code
    result = interpreter.execute(parser.parse(code))

    # The result should be 84 (42 + 42)
    assert result == 84


def test_awaitable_value_complex():
    """
    Test that AwaitableValue.__await__ returns its value in a more complex
    scenario.
    """
    code = """
async def nested_awaitable():
    # This will create a chain of awaitable values
    async def inner():
        return 42

    # First await will yield the value
    value = await inner()
    # Second await will return the value
    return await inner()

# Execute the nested awaitable
result = await nested_awaitable()
result  # This should be 42
"""

    parser = ExpressionsParser()
    interpreter = ExpressionsInterpreter()

    # Execute the code
    result = interpreter.execute(parser.parse(code))

    # The result should be 42
    assert result == 42
