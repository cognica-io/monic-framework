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
    monic_bind_default_module("pandas", "pd")
    _has_pandas = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_pandas = False  # pragma: no cover


@monic_bind_default("pd.is_available")
def pd_is_available() -> bool:
    return _has_pandas
