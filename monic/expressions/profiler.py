#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=too-many-instance-attributes,too-many-arguments

import ast
import textwrap
import time
import typing as t

from dataclasses import asdict, dataclass, field


@dataclass
class CPUProfileRecord:
    node_type: str
    depth: int = 0
    lineno: int = 0
    end_lineno: int = 0
    col_offset: int = 0
    end_col_offset: int = 0
    total_time_ns: int = 0
    self_time_ns: int = 0
    call_count: int = 0
    snippet: str | None = None
    children: list["CPUProfileRecord"] = field(default_factory=list)

    def to_dict(self) -> dict[str, t.Any]:
        return asdict(self)


@dataclass
class CPUProfileRecordInternal:
    node_type: str
    depth: int = 0
    lineno: int = 0
    end_lineno: int = 0
    col_offset: int = 0
    end_col_offset: int = 0
    total_time_ns: int = 0
    self_time_ns: int = 0
    call_count: int = 0
    snippet: str | None = None
    children: dict[str, "CPUProfileRecordInternal"] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, t.Any]:
        return asdict(self)


class CPUProfiler:
    def __init__(self, cpu_threshold: float | None = None) -> None:
        self._stack: list[CPUProfileRecordInternal] = []
        self._root = CPUProfileRecordInternal("Root")
        self._current = self._root
        self._records: dict[str, CPUProfileRecordInternal] = {}
        self._start_time = time.process_time_ns()
        self._cpu_threshold = cpu_threshold

    def reset(self) -> None:
        self._stack = []
        self._root = CPUProfileRecordInternal("Root")
        self._current = self._root
        self._records = {}
        self._start_time = time.process_time_ns()

    def begin_record(
        self,
        node: ast.AST,
        node_type: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ) -> None:
        # Create a key for the record
        location = (
            f"{id(node)}/{node_type}:{lineno}:{col_offset}:{end_col_offset}"
        )
        for record in self._stack:
            parent_location = (
                f"{record.node_type}:{record.lineno}:{record.col_offset}"
                f":{record.end_col_offset}"
            )
            location = f"{location}/{parent_location}"

        # Reuse existing record or create new one
        if location in self._records:
            record = self._records[location]
        else:
            depth = len(self._stack)
            record = CPUProfileRecordInternal(
                node_type,
                depth,
                lineno,
                end_lineno,
                col_offset,
                end_col_offset,
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
        self._start_time = time.process_time_ns()

    def end_record(self) -> None:
        if not self._stack:
            return

        record = self._stack.pop()
        elapsed_time = time.process_time_ns() - self._start_time

        # Accumulate time for current record
        record.total_time_ns += elapsed_time
        record.self_time_ns += elapsed_time
        record.call_count += 1

        # Accumulate time for all parents in the stack
        for parent in self._stack:
            parent.total_time_ns += elapsed_time

        # Update current record to the last record in the stack
        self._current = self._stack[-1] if self._stack else self._root

    def get_report(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> list[CPUProfileRecord]:
        # If no records, return empty list
        if not self._stack:
            return []

        # If code is provided, get code lines
        if code:
            code_lines = textwrap.dedent(code).splitlines()
        else:
            code_lines = None

        # Apply CPU threshold filtering
        filtered_records = [
            record
            for record in self._stack[0].children.values()
            if record.depth <= 1
            and (
                self._cpu_threshold is None
                or record.total_time_ns / 1_000_000_000 >= self._cpu_threshold
            )
        ]

        # Convert internal records to external records
        def convert_record_recursively(
            record: CPUProfileRecordInternal,
        ) -> CPUProfileRecord:
            return CPUProfileRecord(
                node_type=record.node_type,
                depth=record.depth,
                lineno=record.lineno,
                end_lineno=record.end_lineno,
                col_offset=record.col_offset,
                end_col_offset=record.end_col_offset,
                total_time_ns=record.total_time_ns,
                self_time_ns=record.self_time_ns,
                call_count=record.call_count,
                snippet=record.snippet,
                children=[
                    convert_record_recursively(child)
                    for child in record.children.values()
                ],
            )

        converted_records = [
            convert_record_recursively(record) for record in filtered_records
        ]

        # Sort records by execution time recursively
        def sort_records_recursively(
            records: list[CPUProfileRecord],
        ) -> list[CPUProfileRecord]:
            sorted_records = sorted(
                records, key=lambda x: x.total_time_ns, reverse=True
            )
            for record in sorted_records:
                record.children = sort_records_recursively(record.children)

            return sorted_records

        sorted_records = sort_records_recursively(converted_records)

        # If code is provided, set snippets for all records
        def set_snippets_recursively(record: CPUProfileRecord) -> None:
            if code_lines and 0 <= record.lineno - 1 < len(code_lines):
                record.snippet = (
                    code_lines[record.lineno - 1].split("#")[0].rstrip()
                )

            for child in record.children:
                set_snippets_recursively(child)

        # If code is provided, set snippets for all records
        if code_lines:
            for record in sorted_records:
                set_snippets_recursively(record)

        # If top_n is provided, limit the number of records
        if top_n:
            sorted_records = sorted_records[:top_n]

        return sorted_records

    def get_report_as_dict(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> list[dict[str, t.Any]]:
        records = self.get_report(code=code, top_n=top_n)

        return [record.to_dict() for record in records]

    def get_report_as_string(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> str:
        records = self.get_report(code=code, top_n=top_n)

        def format_record(
            record: CPUProfileRecord, depth: int = 0
        ) -> list[str]:
            lines = []
            indent = "│ " * (depth - 1) + ("└── " if depth > 0 else "")
            total_time = record.total_time_ns / 1_000_000_000
            self_time = record.self_time_ns / 1_000_000_000

            # Print basic information with tree structure
            lines.append(
                f"{indent}{record.node_type} "
                f"[{record.lineno}:{record.col_offset}] "
                f"(total={total_time:.6f}s, self={self_time:.6f}s, "
                f"calls={record.call_count})"
            )

            # Print code snippet with improved markers
            if record.snippet:
                snippet_indent = "│ " * depth + "│ "
                lines.append(f"{snippet_indent}▶ {record.snippet}")
                marker_indent = " " * (record.col_offset + 2)
                adjustment = self._get_marker_adjustment(record)
                marker_length = len(record.snippet.strip()[0:adjustment])
                lines.append(
                    f"{snippet_indent}{marker_indent}{'~' * marker_length}"
                )

            # Print child nodes
            for child in record.children:
                lines.extend(format_record(child, depth + 1))

            return lines

        report = ["CPU Profiling Report:\n"]
        for record in records:
            report.extend(format_record(record))

        return "\n".join(report)

    def _get_marker_adjustment(self, record: CPUProfileRecord) -> int:
        adjustment: dict[str, int] = {
            "FunctionDef": -1,
            "If": -1,
            "While": -1,
            "For": -1,
        }

        return adjustment.get(
            record.node_type, record.end_col_offset - record.col_offset
        )
