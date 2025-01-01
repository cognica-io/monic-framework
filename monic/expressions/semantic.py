#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import ast


class SemanticAnalyzer(ast.NodeVisitor):
    """Semantic analyzer for Monic expressions.

    This class performs semantic analysis on the AST before interpretation.
    It checks for various semantic rules that cannot be enforced by the parser alone.
    """

    def visit_Match(self, node: ast.Match) -> None:
        """Check semantic rules for match statements.

        Args:
            node: Match AST node

        Raises:
            SyntaxError: If pattern matching syntax is invalid
        """
        for case in node.cases:
            if isinstance(case.pattern, ast.MatchSequence):
                star_count = sum(
                    1
                    for p in case.pattern.patterns
                    if isinstance(p, ast.MatchStar)
                )
                if star_count > 1:
                    raise SyntaxError(
                        "multiple starred expressions in sequence pattern"
                    )
        # Continue visiting child nodes
        self.generic_visit(node)

    def analyze(self, node: ast.AST) -> None:
        """Analyze the AST for semantic correctness.

        Args:
            node: The root AST node

        Raises:
            SyntaxError: If semantic rules are violated
        """
        self.visit(node)
