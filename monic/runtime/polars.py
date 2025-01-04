#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions.registry import (
    monic_bind_default,
    monic_bind_default_module,
)


try:
    monic_bind_default_module("polars", "pl")
    _has_polars = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_polars = False  # pragma: no cover


@monic_bind_default("pl.is_available")
def pl_is_available() -> bool:
    return _has_polars
