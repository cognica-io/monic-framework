#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions.registry import monic_bind_default_module


try:
    monic_bind_default_module("polars", "pl")
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
