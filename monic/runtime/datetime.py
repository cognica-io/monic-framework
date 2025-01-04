#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions.registry import (
    monic_bind_default,
    monic_bind_default_module,
)


monic_bind_default_module("datetime", "datetime")


@monic_bind_default("datetime.is_available")
def datetime_is_available() -> bool:
    return True
