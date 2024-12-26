#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import ast
import textwrap


class ExpressionsParser:
    def __init__(self) -> None:
        pass

    def parse(self, expression: str) -> ast.Module:
        expression = textwrap.dedent(expression)
        return ast.parse(expression, mode="exec", type_comments=True)
