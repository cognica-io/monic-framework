#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import time

from monic.expressions.registry import monic_bind_default


@monic_bind_default("time.time")
def time_time() -> float:
    return time.time()


@monic_bind_default("time.monotonic")
def time_monotonic() -> float:
    return time.monotonic()
