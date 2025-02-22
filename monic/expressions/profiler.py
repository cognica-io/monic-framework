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
    children: list["CPUProfileRecord"] = field(default_factory=list)


class CPUProfiler:
    def __init__(self, cpu_threshold: float | None = None) -> None:
        self._stack: list[CPUProfileRecord] = []
        self._root = CPUProfileRecord("Root")
        self._current = self._root
        self._records: dict[str, CPUProfileRecord] = {}
        self._start_time = time.perf_counter()
        self._cpu_threshold = cpu_threshold

    def reset(self) -> None:
        self._stack = []
        self._root = CPUProfileRecord("Root")
        self._current = self._root
        self._records = {}
        self._start_time = time.perf_counter()

    def begin_record(
        self,
        node_type: str,
        lineno: int,
        end_lineno: int,
        col_offset: int,
        end_col_offset: int,
    ) -> None:
        if not self._stack:
            self._stack.append(self._root)

        record = CPUProfileRecord(
            node_type,
            lineno,
            end_lineno,
            col_offset,
            end_col_offset,
        )
        self._current.children.append(record)
        self._stack.append(record)
        self._current = record

        self._start_time = time.perf_counter()

    def end_record(self) -> None:
        if len(self._stack) > 1:
            record = self._stack.pop()
            record.total_time = time.perf_counter() - self._start_time
            # If the record is below the threshold, don't add it to the records.
            if self._cpu_threshold and record.total_time < self._cpu_threshold:
                return

            self._current = self._stack[-1]

            key = f"{record.node_type}:{record.lineno}:{record.col_offset}"
            if key not in self._records:
                self._records[key] = record
                self._records[key].call_count += 1
            else:
                self._records[key].total_time += record.total_time
                self._records[key].call_count += 1

    def get_report(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> list[CPUProfileRecord]:
        if code:
            code_lines = textwrap.dedent(code).splitlines()
        else:
            code_lines = None

        records = []
        for _, record in sorted(
            self._records.items(),
            key=lambda x: x[1].total_time,
            reverse=True,
        ):
            if code_lines:
                snippet = code_lines[record.lineno - 1]
                record.snippet = snippet.split("#")[0].rstrip()
            records.append(record)

        if top_n:
            records = records[:top_n]

        return records

    def get_report_as_string(
        self, *, code: str | None = None, top_n: int | None = None
    ) -> str:
        report = ["CPU Profiling Report:\n"]
        records = self.get_report(code=code, top_n=top_n)
        for record in records:
            location = f"{record.lineno}:{record.col_offset}"
            report.append(
                f"{record.node_type:15} {location:<8} {record.total_time:.6f}s"
                f" ({record.call_count} calls)"
            )

            if record.snippet:
                report.append(f"  - {record.snippet}")
                adjustment = self._get_marker_adjustment(record)
                marker_length = len(record.snippet.strip()[0:adjustment])
                report.append(
                    "    " + (" " * (record.col_offset)) + ("^" * marker_length)
                )

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
