#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=too-many-instance-attributes,too-many-arguments

import textwrap
import time

from dataclasses import dataclass, field


@dataclass
class CPUProfileRecord:
    node_type: str
    lineno: int = 0
    end_lineno: int = 0
    col_offset: int = 0
    end_col_offset: int = 0
    total_time: float = 0.0
    call_count: int = 0
    snippet: str | None = None
    children: dict[str, "CPUProfileRecord"] = field(default_factory=dict)


class CPUProfiler:
    def __init__(self, cpu_threshold: float | None = None) -> None:
        self._stack: list[CPUProfileRecord] = []
        self._root = CPUProfileRecord("Root")
        self._current = self._root
        self._records: dict[str, CPUProfileRecord] = {}
        self._start_time = time.perf_counter_ns()
        self._cpu_threshold = cpu_threshold

    def reset(self) -> None:
        self._stack = []
        self._root = CPUProfileRecord("Root")
        self._current = self._root
        self._records = {}
        self._start_time = time.perf_counter_ns()

    def begin_record(
        self,
        node_type: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ) -> None:
        # Create a key for the record
        location = f"{node_type}:{lineno}:{col_offset}:{end_col_offset}"

        # Reuse existing record or create new one
        if location in self._records:
            record = self._records[location]
        else:
            record = CPUProfileRecord(
                node_type, lineno, end_lineno, col_offset, end_col_offset
            )
            self._records[location] = record

        # Initialize stack with root if empty
        if not self._stack:
            self._stack = [self._root]
            self._current = self._root

        # Set parent-child relationship
        parent = self._stack[-1]
        if location not in parent.children:
            parent.children[location] = record

        self._stack.append(record)
        self._current = record
        self._start_time = time.perf_counter_ns()

    def end_record(self) -> None:
        if len(self._stack) <= 1:
            return

        record = self._stack.pop()
        elapsed_time = time.perf_counter_ns() - self._start_time

        # Accumulate time
        record.total_time += elapsed_time
        record.call_count += 1

        self._current = self._stack[-1]

    def get_report(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> list[CPUProfileRecord]:
        if code:
            code_lines = textwrap.dedent(code).splitlines()
        else:
            code_lines = None

        # Apply CPU threshold filtering
        filtered_records = [
            record
            for record in self._stack[0].children.values()
            if self._cpu_threshold is None
            or record.total_time / 1_000_000_000 >= self._cpu_threshold
        ]

        # Sort records by execution time
        records = sorted(
            filtered_records, key=lambda x: x.total_time, reverse=True
        )

        def set_snippets_recursively(record: CPUProfileRecord) -> None:
            if code_lines and 0 <= record.lineno - 1 < len(code_lines):
                record.snippet = (
                    code_lines[record.lineno - 1].split("#")[0].rstrip()
                )

            for child in record.children.values():
                set_snippets_recursively(child)

        if code:
            for record in records:
                set_snippets_recursively(record)

        if top_n:
            records = records[:top_n]

        return records

    def get_report_as_string(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> str:
        report = ["CPU Profiling Report:\n"]
        records = self.get_report(code=code, top_n=top_n)

        def format_record(
            record: CPUProfileRecord, depth: int = 0
        ) -> list[str]:
            lines = []
            indent = "  " * depth
            location = f"{record.lineno}:{record.col_offset}"

            # Print basic information
            lines.append(
                f"{indent}{record.node_type:15} {location:<8} "
                f"{record.total_time / 1_000_000:.6f}ms "
                f"({record.call_count} calls)"
            )

            # Print code snippet
            if record.snippet:
                lines.append(f"{indent}  - {record.snippet}")
                adjustment = self._get_marker_adjustment(record)
                marker_length = len(record.snippet.strip()[0:adjustment])
                lines.append(
                    f"{indent}    "
                    + (" " * (record.col_offset))
                    + ("^" * marker_length)
                )

            # Print child nodes
            for child in record.children.values():
                lines.extend(format_record(child, depth + 1))

            return lines

        for record in records:
            report.extend(format_record(record))

        return "\n".join(report)

    def _get_marker_adjustment(self, record: CPUProfileRecord) -> int:
        adjustment: dict[str, int] = {
            "FunctionDef": -1,
            "If": -1,
            "While": -1,
            "For": -1,
            "Subscript": -1,
        }

        return adjustment.get(
            record.node_type, record.end_col_offset - record.col_offset
        )
