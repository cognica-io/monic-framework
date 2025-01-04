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
    monic_bind_default_module("numpy", "np")
    _has_numpy = True  # pragma: no cover
except ImportError:  # pragma: no cover
    _has_numpy = False  # pragma: no cover


@monic_bind_default("np.is_available")
def np_is_available() -> bool:
    return _has_numpy
