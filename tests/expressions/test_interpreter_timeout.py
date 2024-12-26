#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import time

import pytest

from monic.expressions import (
    ExpressionsContext,
    ExpressionsParser,
    ExpressionsInterpreter,
)


def test_timeout_infinite_loop():
    """Test that an infinite loop raises a TimeoutError."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=0.1)
    interpreter = ExpressionsInterpreter(context)

    infinite_loop_code = parser.parse(
        """
while True:
    pass
"""
    )

    start_time = time.time()
    with pytest.raises(TimeoutError):
        interpreter.execute(infinite_loop_code)

    # Verify the timeout occurred within a reasonable time frame
    elapsed_time = time.time() - start_time
    assert (
        0.09 < elapsed_time < 0.5
    ), f"Timeout occurred in {elapsed_time} seconds, expected around 0.1"


def test_timeout_long_computation():
    """Test that a long-running computation raises a TimeoutError."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=0.1)
    interpreter = ExpressionsInterpreter(context)

    long_computation_code = parser.parse(
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Intentionally computationally expensive
result = fibonacci(35)
"""
    )

    start_time = time.time()
    with pytest.raises(TimeoutError):
        interpreter.execute(long_computation_code)

    # Verify the timeout occurred within a reasonable time frame
    elapsed_time = time.time() - start_time
    assert (
        0.09 < elapsed_time < 0.5
    ), f"Timeout occurred in {elapsed_time} seconds, expected around 0.1"


def test_no_timeout_short_computation():
    """Test that a short computation does not raise a TimeoutError."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=1.0)
    interpreter = ExpressionsInterpreter(context)

    short_computation_code = parser.parse(
        """
result = sum(range(1000))
result
"""
    )

    start_time = time.time()
    result = interpreter.execute(short_computation_code)
    elapsed_time = time.time() - start_time

    # Verify the computation finished in a reasonable time
    assert elapsed_time < 1.0, f"Computation took {elapsed_time} seconds"

    # Verify the result is correct
    assert result == 499500


def test_timeout_none():
    """Test that when timeout is None, no timeout occurs."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=None)
    interpreter = ExpressionsInterpreter(context)

    long_computation_code = parser.parse(
        """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Intentionally computationally expensive
result = fibonacci(28)
"""
    )

    start_time = time.time()
    interpreter.execute(long_computation_code)
    elapsed_time = time.time() - start_time

    # This is a legitimately long computation, but should complete
    assert (
        elapsed_time > 3.0
    ), "Computation should take a considerable amount of time"


def test_timeout_repeated_method_calls():
    """Test timeout mechanism works across multiple method calls."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=0.1)
    interpreter = ExpressionsInterpreter(context)

    repeated_calls_code = parser.parse(
        """
def slow_function():
    total = 0
    for _ in range(1000000):
        total += 1
    return total

result1 = slow_function()
result2 = slow_function()
"""
    )

    start_time = time.time()
    with pytest.raises(TimeoutError):
        interpreter.execute(repeated_calls_code)

    # Verify the timeout occurred within a reasonable time frame
    elapsed_time = time.time() - start_time
    assert (
        0.09 < elapsed_time < 0.5
    ), f"Timeout occurred in {elapsed_time} seconds, expected around 0.1"


# Performance test with parametrization
@pytest.mark.parametrize("timeout", [0.05, 0.1, 0.2])
def test_timeout_variability(timeout):
    """Test timeout mechanism with different timeout values."""
    parser = ExpressionsParser()
    context = ExpressionsContext(timeout=timeout)
    interpreter = ExpressionsInterpreter(context)

    long_computation_code = parser.parse(
        """
def busy_wait():
    total = 0
    start_time = time.time()
    while time.time() - start_time < 1:  # Busy wait for 1 second
        total += 1
    return total

result = busy_wait()
"""
    )

    start_time = time.time()
    with pytest.raises(TimeoutError):
        interpreter.execute(long_computation_code)

    # Verify the timeout occurred within a reasonable time frame
    elapsed_time = time.time() - start_time
    assert (
        elapsed_time < timeout + 0.1
    ), f"Timeout occurred in {elapsed_time} seconds, expected around {timeout}"
