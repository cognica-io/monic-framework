#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import typing as t
from dataclasses import dataclass


@dataclass
class ExpressionContext:
    # The timeout for evaluating the expression in seconds.
    timeout: t.Optional[float] = 10.0
