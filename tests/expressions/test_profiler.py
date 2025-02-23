#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=protected-access

import ast

from monic.expressions import (
    CPUProfiler,
    CPUProfileRecord,
    ExpressionsParser,
    ExpressionsContext,
    ExpressionsInterpreter,
)


def test_cpu_profiler_initialization():
    profiler = CPUProfiler()
    assert profiler._root.node_type == "Root"
    assert len(profiler._stack) == 0
    assert len(profiler._records) == 0


def test_cpu_profiler_reset():
    profiler = CPUProfiler()

    # Add some records
    profiler.begin_record(ast.Name(id="TestNode"), "TestNode", 1, 1, 0, 10)
    profiler.end_record()

    profiler.reset()
    assert len(profiler._stack) == 0
    assert len(profiler._records) == 0
    assert profiler._root.node_type == "Root"


def test_cpu_profiler_record_tracking():
    profiler = CPUProfiler()

    # Test single record
    profiler.begin_record(
        ast.Name(id="FunctionDef"), "FunctionDef", 1, 5, 0, 20
    )
    profiler.end_record()

    report = profiler.get_report()
    assert len(report) == 1
    assert report[0].node_type == "FunctionDef"
    assert report[0].call_count == 1


def test_cpu_profiler_nested_records():
    profiler = CPUProfiler()

    # Create nested structure
    profiler.begin_record(
        ast.Name(id="FunctionDef"), "FunctionDef", 1, 10, 0, 20
    )
    profiler.begin_record(ast.Name(id="For"), "For", 2, 5, 4, 15)
    profiler.end_record()  # End For
    profiler.end_record()  # End FunctionDef

    report = profiler.get_report()
    assert len(report) == 1
    assert len(report[0].children) == 1
    assert report[0].children[0].node_type == "For"


def test_cpu_profiler_with_code_snippets():
    code = """
    def example():
        for i in range(10):
            print(i)
    """
    profiler = CPUProfiler()

    profiler.begin_record(
        ast.Name(id="FunctionDef"), "FunctionDef", 2, 4, 4, 15
    )
    profiler.end_record()

    report = profiler.get_report(code=code)
    assert report[0].snippet is not None
    assert "def example()" in report[0].snippet


def test_cpu_profiler_threshold_filtering():
    profiler = CPUProfiler(cpu_threshold=1.0)  # 1 second threshold

    profiler.begin_record(
        ast.Name(id="SlowOperation"), "SlowOperation", 1, 1, 0, 10
    )
    profiler.end_record()

    report = profiler.get_report()
    assert (
        len(report) == 0
    )  # Should be filtered out as it's faster than threshold


def test_cpu_profiler_report_formats():
    profiler = CPUProfiler()

    profiler.begin_record(ast.Name(id="TestNode"), "TestNode", 1, 1, 0, 10)
    profiler.end_record()

    # Test dictionary format
    dict_report = profiler.get_report_as_dict()
    assert isinstance(dict_report, list)
    assert isinstance(dict_report[0], dict)
    assert dict_report[0]["node_type"] == "TestNode"

    # Test string format
    str_report = profiler.get_report_as_string()
    assert isinstance(str_report, str)
    assert "TestNode" in str_report
    assert "total=" in str_report
    assert "self=" in str_report


def test_cpu_profiler_top_n_limiting():
    profiler = CPUProfiler()

    # Add multiple records
    for i in range(5):
        profiler.begin_record(
            ast.Name(id=f"Node{i}"), f"Node{i}", i + 1, i + 1, 0, 10
        )
        profiler.end_record()

    report = profiler.get_report(top_n=3)
    assert len(report) == 3


def test_cpu_profiler_empty_report():
    profiler = CPUProfiler()

    report = profiler.get_report()
    assert len(report) == 0

    str_report = profiler.get_report_as_string()
    assert "CPU Profiling Report:" in str_report


def test_profiler_basic_execution():
    # Setup
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    code = """
    result = 0
    for i in range(100):
        result += i
    return result
    """

    # Execute
    tree = parser.parse(code)
    result = interpreter.execute(tree)

    # Verify execution result
    assert result == sum(range(100))

    # Verify profiler exists and has records
    assert interpreter.cpu_profiler is not None
    report = interpreter.cpu_profiler.get_report()
    assert len(report) > 0


def test_profiler_function_calls():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)

    result = factorial(5)
    return result
    """

    # Execute
    tree = parser.parse(code)
    result = interpreter.execute(tree)

    # Verify execution result
    assert result == 120

    # Get profiler report
    report = interpreter.cpu_profiler.get_report()

    def get_node_by_lineno(
        node: CPUProfileRecord, name: str, lineno: int
    ) -> list[CPUProfileRecord]:
        result: list[CPUProfileRecord] = []
        if node.node_type == name and node.lineno == lineno:
            result.append(node)
        for child in node.children:
            result.extend(get_node_by_lineno(child, name, lineno))
        return result

    # Verify function calls are tracked
    nodes = [n for r in report for n in get_node_by_lineno(r, "Call", 5)]
    assert len(nodes) > 0
    call_count = sum(n.call_count for n in nodes)
    assert call_count > 1


def test_profiler_report_formatting():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    def calculate():
        sum = 0
        for i in range(100):
            sum += i
        return sum

    result = calculate()
    return result
    """

    # Execute
    tree = parser.parse(code)
    interpreter.execute(tree)

    # Test different report formats
    dict_report = interpreter.cpu_profiler.get_report_as_dict(code=code)
    str_report = interpreter.cpu_profiler.get_report_as_string(code=code)

    assert isinstance(dict_report, list)
    assert isinstance(str_report, str)
    assert "CPU Profiling Report:" in str_report
    assert "calculate" in str_report


def test_profiler_with_cpu_threshold():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=1.0,
    )  # 1 second threshold
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    # Quick operation
    result = 1 + 1
    return result
    """

    # Execute
    tree = parser.parse(code)
    interpreter.execute(tree)

    # Should have no records due to threshold
    report = interpreter.cpu_profiler.get_report()
    assert len(report) == 0


def test_profiler_nested_operations():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    def outer():
        def inner():
            sum = 0
            for i in range(50):
                sum += i
            return sum
        return inner()

    result = outer()
    return result
    """

    # Execute
    tree = parser.parse(code)
    result = interpreter.execute(tree)

    # Verify execution result
    assert result == sum(range(50))

    # Get profiler report
    report = interpreter.cpu_profiler.get_report()

    seen = set()

    def get_node_by_lineno(
        node: CPUProfileRecord, lineno: int
    ) -> list[CPUProfileRecord]:
        result: list[CPUProfileRecord] = []
        key = f"{node.node_type}:{node.lineno}:{node.col_offset}"
        if key in seen:
            return result
        seen.add(key)

        if node.lineno == lineno:
            result.append(node)
        for child in node.children:
            result.extend(get_node_by_lineno(child, lineno))
        return result

    # Verify nested function structure
    nodes = [n for r in report for n in get_node_by_lineno(r, 10)]
    nodes = [n for n in nodes if n is not None and n.node_type == "Call"]
    assert len(nodes) > 0
    assert len(nodes[0].children) > 0


def test_profiler_reset():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    # First execution
    code1 = "result = 1 + 1"
    tree1 = parser.parse(code1)
    interpreter.execute(tree1)

    # Get initial report
    initial_report = interpreter.cpu_profiler.get_report()
    assert len(initial_report) > 0

    # Reset profiler
    interpreter.cpu_profiler.reset()

    # Get report after reset
    empty_report = interpreter.cpu_profiler.get_report()
    assert len(empty_report) == 0

    # Second execution
    code2 = "result = 2 + 2"
    tree2 = parser.parse(code2)
    interpreter.execute(tree2)

    # Verify new records are created
    new_report = interpreter.cpu_profiler.get_report()
    assert len(new_report) > 0


def test_profiler_top_n_limiting():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    def func1(): return 1
    def func2(): return 2
    def func3(): return 3
    def func4(): return 4
    def func5(): return 5

    result = func1() + func2() + func3() + func4() + func5()
    return result
    """

    # Execute
    tree = parser.parse(code)
    interpreter.execute(tree)

    # Get reports with different top_n values
    full_report = interpreter.cpu_profiler.get_report()
    top_3_report = interpreter.cpu_profiler.get_report(top_n=3)

    assert len(top_3_report) <= 3
    assert len(full_report) >= len(top_3_report)


def test_profiler_code_snippets():
    parser = ExpressionsParser()
    context = ExpressionsContext(
        allow_return_at_top_level=True,
        enable_cpu_profiling=True,
        cpu_threshold=None,
    )
    interpreter = ExpressionsInterpreter(context)

    assert interpreter.cpu_profiler is not None

    code = """
    def example():
        x = 1
        y = 2
        return x + y

    result = example()
    return result
    """

    # Execute
    tree = parser.parse(code)
    interpreter.execute(tree)

    # Get report with code snippets
    report = interpreter.cpu_profiler.get_report(code=code)

    # Verify snippets are included
    for record in report:
        if record.node_type == "FunctionDef":
            assert record.snippet is not None
            assert "def example" in record.snippet
