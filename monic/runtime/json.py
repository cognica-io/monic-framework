#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import json

from monic.expressions.registry import monic_bind_default


@monic_bind_default("json.dumps")
def json_dumps(obj, *args, **kwargs):
    return json.dumps(obj, *args, **kwargs)


@monic_bind_default("json.loads")
def json_loads(s, *args, **kwargs):
    return json.loads(s, *args, **kwargs)
