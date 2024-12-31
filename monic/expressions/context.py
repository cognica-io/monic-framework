#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from dataclasses import dataclass


@dataclass
class ExpressionsContext:
    # The timeout for evaluating the expression in seconds.
    timeout: float | None = 10.0
