#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

from dataclasses import dataclass


@dataclass
class ExpressionContext:
    # The timeout for evaluating the expression in seconds.
    timeout: float = 10.0
