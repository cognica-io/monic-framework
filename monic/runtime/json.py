#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import json
import typing as t

from monic.expressions.registry import monic_bind_default


@monic_bind_default("json.dumps")
def json_dumps(obj, *args, **kwargs) -> str:
    return json.dumps(obj, *args, **kwargs)


@monic_bind_default("json.loads")
def json_loads(s, *args, **kwargs) -> t.Any:
    return json.loads(s, *args, **kwargs)
