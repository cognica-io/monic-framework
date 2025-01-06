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
    monic_bind_default_module("pyarrow", "pa")
    _has_arrow = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_arrow = False  # pragma: no cover

try:
    monic_bind_default_module("pyarrow.compute", "pc")
    _has_arrow_compute = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_arrow_compute = False  # pragma: no cover

try:
    monic_bind_default_module("pyarrow.parquet", "pq")
    _has_arrow_parquet = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_arrow_parquet = False  # pragma: no cover


@monic_bind_default("pa.is_available")
def pa_is_available() -> bool:
    return _has_arrow


@monic_bind_default("pc.is_available")
def pc_is_available() -> bool:
    return _has_arrow_compute


@monic_bind_default("pq.is_available")
def pq_is_available() -> bool:
    return _has_arrow_parquet
