#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

from monic.expressions.registry import monic_bind_default_module


try:
    monic_bind_default_module("polars", "pl")
except ImportError:
    pass