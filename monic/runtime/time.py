#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import time

from monic.expressions.registry import monic_bind_default


@monic_bind_default("time.time")
def time_time() -> float:
    return time.time()


@monic_bind_default("time.monotonic")
def time_monotonic() -> float:
    return time.monotonic()


@monic_bind_default("time.is_available")
def time_is_available() -> bool:
    return True
