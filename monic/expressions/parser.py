#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import ast


class ExpressionParser:
    def __init__(self):
        pass

    def parse(self, expression: str) -> ast.Module:
        return ast.parse(expression, mode="exec", type_comments=True)
