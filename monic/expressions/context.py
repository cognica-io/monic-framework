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
