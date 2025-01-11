#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import ast

from monic.expressions.exceptions import SecurityError


class SecurityChecker(ast.NodeVisitor):
    """Security checker for Monic expressions.

    This class performs security checks on the AST before interpretation.
    It ensures that no dangerous operations are allowed.
    """

    # List of forbidden functions and modules
    FORBIDDEN_NAMES = {
        # Built-in functions
        "eval",
        "exec",
        "compile",
        "execfile",
        "open",
        "globals",
        "locals",
        "vars",
        "__import__",
        # Module functions
        "time.sleep",
    }

    # List of forbidden attribute accesses
    FORBIDDEN_ATTRS = {
        # Module attributes
        "__builtins__",
        "__loader__",
        "__spec__",
        # Class and instance attributes
        "__code__",
        "__globals__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__qualname__",
        "__traceback__",
    }

    def check_attribute_security(self, attr_name: str) -> None:
        """Check if attribute access is allowed.

        Args:
            attr_name: Name of the attribute to check

        Raises:
            SecurityError: If attribute access is not allowed
        """
        if attr_name in self.FORBIDDEN_ATTRS:
            raise SecurityError(
                f"Access to '{attr_name}' attribute is not allowed"
            )

    def visit_Name(self, node: ast.Name) -> None:
        """Check for forbidden names."""
        if node.id in self.FORBIDDEN_NAMES:
            raise SecurityError(f"Call to builtin '{node.id}' is not allowed")
        if node.id == "__builtins__":
            raise SecurityError(
                "Access to '__builtins__' attribute is not allowed"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for forbidden attribute access."""
        # First check if this is a forbidden attribute
        self.check_attribute_security(node.attr)

        # Check if this is a forbidden function call
        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{node.attr}"
            if full_name in self.FORBIDDEN_NAMES:
                raise SecurityError(f"Call to '{full_name}' is not allowed")

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Check for import statements."""
        raise SecurityError("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for import from statements."""
        raise SecurityError("Import statements are not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        """Check for security violations in function calls.

        Args:
            node: The Call AST node

        Raises:
            SecurityError: If a forbidden operation is attempted
        """
        # Check if this is a getattr call
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
        ):
            # Check if the second argument (attribute name) is a string literal
            attr_arg = node.args[1]
            if isinstance(attr_arg, ast.Constant) and isinstance(
                attr_arg.value, str
            ):
                # Check if the attribute is forbidden
                self.check_attribute_security(attr_arg.value)

        self.generic_visit(node)

    def check(self, node: ast.AST) -> None:
        """Check the AST for security violations.

        Args:
            node: The root AST node

        Raises:
            SecurityError: If security rules are violated
        """
        self.visit(node)
