#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from dataclasses import dataclass


@dataclass
class ExpressionsContext:
    # Allow return statements at top-level
    allow_return_at_top_level: bool = False

    # The timeout for evaluating the expression in seconds.
    timeout: float | None = 10.0

    # Whether to enable CPU profiling.
    enable_cpu_profiling: bool = False
    # The CPU threshold for evaluating the expression in seconds.
    cpu_threshold: float | None = 0.000001  # 1 microsecond
