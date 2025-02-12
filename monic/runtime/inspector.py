#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import inspect
import types
import typing as t

from monic.expressions.registry import (
    NamespaceProxy,
    monic_bind_default,
)


@monic_bind_default("inspector.signature")
def inspector_signature(
    obj: t.Callable[..., t.Any] | NamespaceProxy | types.ModuleType
) -> str:
    if isinstance(obj, (NamespaceProxy, types.ModuleType)):
        raise ValueError("Object is not callable.")

    if hasattr(obj, "__expressions_name__"):
        name = obj.__expressions_name__
    else:
        name = f"{obj.__module__}.{obj.__name__}"

    sig = inspect.signature(obj)

    return f"{name}{sig}"


@monic_bind_default("inspector.is_available")
def inspector_is_available() -> bool:
    return True
