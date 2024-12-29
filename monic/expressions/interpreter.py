#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=no-else-break,no-else-return,no-else-raise,broad-except
# pylint: disable=too-many-branches,too-many-return-statements,too-many-locals
# pylint: disable=too-many-public-methods,too-many-instance-attributes
# pylint: disable=too-many-statements,too-many-nested-blocks,too-many-lines
# pylint: disable=unnecessary-dunder-call

import ast
import datetime
import operator
import sys
import time
import types
import typing as t

from dataclasses import dataclass, field

from monic.expressions.context import ExpressionsContext
from monic.expressions.exceptions import (
    SecurityError,
    UnsupportedUnpackingError,
)
from monic.expressions.registry import registry


class ReturnValue(Exception):
    """Raised to return a value from a function."""

    def __init__(self, value):
        self.value = value


class BreakLoop(Exception):
    """Raised to break out of a loop."""


class ContinueLoop(Exception):
    """Raised to continue to the next iteration of a loop."""


@dataclass
class Scope:
    # Names declared as global
    globals: t.Set[str] = field(default_factory=set)
    # Names declared as nonlocal
    nonlocals: t.Set[str] = field(default_factory=set)
    # Names assigned in current scope
    locals: t.Set[str] = field(default_factory=set)


@dataclass
class ControlFlow:
    """Record for tracking control flow state."""

    function_depth: int = 0
    loop_depth: int = 0


class ExpressionsInterpreter(ast.NodeVisitor):
    def __init__(self, context: t.Optional[ExpressionsContext] = None) -> None:
        """Initialize the interpreter.

        Args:
            context: Optional context for execution
        """
        self.started_at = time.monotonic()

        self.context = context or ExpressionsContext()
        self.scope_stack: t.List[Scope] = [Scope()]  # Track scopes
        self.control: ControlFlow = ControlFlow()

        # Initialize with built-in environment
        self.global_env: t.Dict[str, t.Any] = {
            # Built-in functions
            "print": print,
            "len": len,
            "range": range,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "zip": zip,
            "enumerate": enumerate,
            "filter": filter,
            "map": map,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "issubclass": issubclass,
            # Built-in types
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "tuple": tuple,
            "set": set,
            "dict": dict,
            # Constants
            "None": None,
            "True": True,
            "False": False,
            # Exceptions
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "NameError": NameError,
            "IndexError": IndexError,
            "KeyError": KeyError,
            "ZeroDivisionError": ZeroDivisionError,
            "StopIteration": StopIteration,
            "TimeoutError": TimeoutError,
            "RuntimeError": RuntimeError,
            "SecurityError": SecurityError,
            "UnsupportedUnpackingError": UnsupportedUnpackingError,
        }

        # Add registered objects to global environment
        self.global_env.update(registry.get_all())

        # Add built-in modules
        self.global_env.update(
            {
                "datetime": datetime,
                "time": time,
            }
        )

        # Add built-in decorators
        self.global_env.update(
            {
                "classmethod": classmethod,
                "staticmethod": staticmethod,
                "property": property,
            }
        )

        self.local_env: t.Dict[str, t.Any] = {}

        # Initialize last result storage
        self.global_env["_"] = None

        # List of forbidden functions and modules
        self.FORBIDDEN_NAMES = {
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
        self.FORBIDDEN_ATTRS = {
            "__code__",
            "__globals__",
            "__dict__",
            "__class__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__qualname__",
        }

    @property
    def current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def execute(self, tree: ast.AST) -> t.Any:
        """Execute an AST."""
        # Perform security check
        self._check_security(tree)

        # Reset the timer for timeout tracking
        self.started_at = time.monotonic()

        try:
            # Handle expression statements specially to capture their value
            if isinstance(tree, ast.Expression):
                result = self.visit(tree)
                self.global_env["_"] = result
                return result
            elif isinstance(tree, ast.Module):
                result = None
                for stmt in tree.body:
                    if isinstance(stmt, ast.Expr):
                        # For expression statements, capture the value
                        result = self.visit(stmt.value)
                        self.global_env["_"] = result
                    else:
                        # For other statements, just execute them
                        self.visit(stmt)
                return result
            else:
                result = self.visit(tree)
                self.global_env["_"] = result
                return result
        except TimeoutError as e:
            raise e
        except Exception as e:
            raise type(e)(str(e)) from e

    def get_name_value(self, name: str) -> t.Any:
        """Get the value of a name in the current scope."""
        return self._get_name_value(name)

    def visit(self, node: ast.AST) -> t.Any:
        """Visit a node and check for timeout."""
        # Check for timeout if one is set
        if self.context.timeout is not None:
            elapsed = time.monotonic() - self.started_at
            if elapsed > self.context.timeout:
                raise TimeoutError(
                    "Execution exceeded timeout of "
                    f"{self.context.timeout} seconds"
                )

        # Get the visitor method for this node type
        visitor = getattr(
            self, f"visit_{type(node).__name__}", self.generic_visit
        )
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Called if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def _check_security(self, node: ast.AST) -> None:
        """Check for potentially dangerous operations in the AST.

        Args:
            node: AST node to check

        Raises:
            SecurityError: If dangerous operations are detected
        """
        for op in ast.walk(node):
            # Check for forbidden function calls
            if isinstance(op, ast.Name) and op.id in self.FORBIDDEN_NAMES:
                raise SecurityError(f"Call to builtin '{op.id}' is not allowed")

            # Check for forbidden attribute access
            if isinstance(op, ast.Attribute):
                # Check for direct forbidden attribute access
                if op.attr in self.FORBIDDEN_ATTRS:
                    raise SecurityError(
                        f"Access to '{op.attr}' attribute is not allowed"
                    )

                # Check for forbidden function calls like time.sleep
                if isinstance(op.value, ast.Name):
                    full_name = f"{op.value.id}.{op.attr}"
                    if full_name in self.FORBIDDEN_NAMES:
                        raise SecurityError(
                            f"Call to '{full_name}' is not allowed"
                        )

            # Check for __builtins__ access
            if isinstance(op, ast.Name) and op.id == "__builtins__":
                raise SecurityError(
                    "Access to '__builtins__' attribute is not allowed"
                )

            # Check for import statements
            if isinstance(op, (ast.Import, ast.ImportFrom)):
                raise SecurityError("Import statements are not allowed")

    def visit_Global(self, node: ast.Global) -> None:
        """Handle global declarations."""
        for name in node.names:
            self.current_scope.globals.add(name)
            # Remove from locals if present
            self.current_scope.locals.discard(name)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """
        Handle 'nonlocal' statements, e.g.:
            nonlocal x, y

        In Python, if a variable is declared 'nonlocal', it must exist in
        at least one enclosing (function) scope. If not found, raise
        SyntaxError as in the standard Python behavior.
        """
        if len(self.scope_stack) < 2:
            raise SyntaxError(
                "nonlocal declaration not allowed at module level"
            )

        for name in node.names:
            # Mark this name as nonlocal in the current scope
            self.current_scope.nonlocals.add(name)

            found = False
            # Check all outer scopes (excluding the current scope)
            for scope in reversed(self.scope_stack[:-1]):
                # If already local or already marked nonlocal there, consider
                # it found
                if (
                    name in scope.locals
                    or name in scope.nonlocals
                    or name in self.local_env
                ):
                    found = True
                    break

            if not found:
                # If it's not in any enclosing scope, Python raises SyntaxError
                raise SyntaxError(
                    f"No binding for nonlocal '{name}' found in outer scopes"
                )

    def visit_Constant(self, node: ast.Constant) -> t.Any:
        return node.value

    def _get_name_value(self, name: str) -> t.Any:
        """Get value of a name considering scope declarations."""
        # Fast path for common case
        if name in self.local_env:
            return self.local_env[name]

        # Check current scope declarations
        current = self.current_scope
        if name in current.globals:
            if name in self.global_env:
                return self.global_env[name]
            raise NameError(f"Global name '{name}' is not defined")

        if name in current.nonlocals:
            # Use reversed list slice for faster iteration
            for scope in reversed(self.scope_stack[:-1]):
                if name in scope.locals:
                    return self.local_env[name]
            raise NameError(f"Nonlocal name '{name}' is not defined")

        if name in self.global_env:
            return self.global_env[name]

        raise NameError(f"Name '{name}' is not defined")

    def _set_name_value(self, name: str, value: t.Any) -> None:
        """
        Set the value of a name, considering 'global' and 'nonlocal'
        declarations.
        """
        # If declared global in the current scope:
        if name in self.current_scope.globals:
            self.global_env[name] = value
            return

        # If declared nonlocal in the current scope:
        if name in self.current_scope.nonlocals:
            # Walk backward through scopes to find the correct one
            for i in range(len(self.scope_stack) - 2, -1, -1):
                scope = self.scope_stack[i]
                if (
                    name in scope.locals
                    or name in scope.nonlocals
                    or name in self.local_env
                ):
                    # Found the appropriate scope, set value in local_env
                    self.local_env[name] = value
                    return
            raise NameError(f"Nonlocal name '{name}' not found in outer scopes")

        # Otherwise, treat it as a local assignment
        self.current_scope.locals.add(name)
        self.local_env[name] = value

    def _del_name_value(self, name: str) -> None:
        """Delete a name from the appropriate scope."""
        if name == "_":
            raise SyntaxError("Cannot delete special variable '_'")

        if name in self.current_scope.globals:
            if name in self.global_env:
                del self.global_env[name]
            else:
                raise NameError(f"Global name '{name}' is not defined")
        elif name in self.current_scope.nonlocals:
            # Search for name in outer scopes
            found = False
            for scope in reversed(self.scope_stack[:-1]):
                if name in scope.locals:
                    found = True
                    if name in self.local_env:
                        del self.local_env[name]
                        scope.locals.remove(name)
                    break
            if not found:
                raise NameError(f"Nonlocal name '{name}' is not defined")
        else:
            # Try to delete from current scope
            if name in self.current_scope.locals:
                del self.local_env[name]
                self.current_scope.locals.remove(name)
            elif name in self.global_env:
                del self.global_env[name]
            else:
                raise NameError(f"Name '{name}' is not defined")

    def visit_Name(self, node: ast.Name) -> t.Any:
        """Visit a Name node, handling variable lookup according to scope rules.

        Args:
            node: The Name AST node

        Returns:
            The value of the name in the appropriate scope

        Raises:
            NameError: If the name cannot be found in any accessible scope
            SyntaxError: If attempting to modify special variable '_'
            NotImplementedError: If the context type is not supported
        """
        # Handle special underscore variable
        if node.id == "_":
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                op = "delete" if isinstance(node.ctx, ast.Del) else "assign to"
                raise SyntaxError(f"Cannot {op} special variable '_'")
            return self.global_env.get("_")

        # Handle different contexts
        if isinstance(node.ctx, ast.Store):
            return node.id
        elif isinstance(node.ctx, ast.Load):
            # If the name is declared global or nonlocal in the current scope,
            # skip the registry fallback entirely so we preserve the correct
            # error.
            if (
                node.id in self.current_scope.globals
                or node.id in self.current_scope.nonlocals
            ):
                return self._get_name_value(node.id)

            try:
                return self._get_name_value(node.id)
            except NameError:
                # If not found in current scope, try the registry
                try:
                    return registry.get(node.id)
                except KeyError as e:
                    raise NameError(f"Name '{node.id}' is not defined") from e
        elif isinstance(node.ctx, ast.Del):
            self._del_name_value(node.id)
        else:
            raise NotImplementedError(
                f"Unsupported context type: {type(node.ctx).__name__}"
            )

    _AUG_OP_MAP: t.Dict[t.Type[ast.operator], t.Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle augmented assignment with proper scope handling."""
        op_func = self._AUG_OP_MAP.get(type(node.op))
        if not op_func:
            raise NotImplementedError(
                "Unsupported augmented assignment operator: "
                f"{type(node.op).__name__}"
            )

        # Get the current value
        if isinstance(node.target, ast.Name):
            target_value = self._get_name_value(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            obj = self.visit(node.target.value)
            target_value = getattr(obj, node.target.attr)
        elif isinstance(node.target, ast.Subscript):
            container = self.visit(node.target.value)
            index = self.visit(node.target.slice)
            target_value = container[index]
        else:
            raise NotImplementedError(
                "Unsupported augmented assignment target: "
                f"{type(node.target).__name__}"
            )

        # Compute the new value
        right_value = self.visit(node.value)
        result = op_func(target_value, right_value)

        # Store the result
        if isinstance(node.target, ast.Name):
            self._set_name_value(node.target.id, result)
        elif isinstance(node.target, ast.Attribute):
            setattr(obj, node.target.attr, result)
        elif isinstance(node.target, ast.Subscript):
            container[index] = result

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.visit(node.value)
        # Handle multiple targets
        if len(node.targets) > 1:
            # Multiple target assignment: a = b = 10
            for target in node.targets:
                self._handle_unpacking_target(target, value)
        else:
            # Single target assignment
            target = node.targets[0]
            self._handle_unpacking_target(target, value)

    def _handle_unpacking_target(self, target: ast.AST, value: t.Any) -> None:
        """
        Handle different types of unpacking targets.

        Args:
            target: AST node representing the unpacking target
            value: The value being assigned

        Raises:
            UnsupportedUnpackingError: If an unsupported unpacking pattern is
            encountered
        """
        if isinstance(target, ast.Name):
            # Simple name assignment with scope handling
            self._set_name_value(target.id, value)
        elif isinstance(target, ast.Attribute):
            # Handle attribute assignment (e.g., self.x = value)
            obj = self.visit(target.value)
            setattr(obj, target.attr, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Tuple or list unpacking
            try:
                # Unpack the value
                if not hasattr(value, "__iter__"):
                    raise ValueError("Cannot unpack non-iterable value")

                iter_value = iter(value)

                # Check for starred expressions (extended unpacking)
                starred_indices = [
                    i
                    for i, elt in enumerate(target.elts)
                    if isinstance(elt, ast.Starred)
                ]

                if len(starred_indices) > 1:
                    raise UnsupportedUnpackingError(
                        "Cannot use multiple starred expressions in assignment"
                    )

                if starred_indices:
                    # Handle starred unpacking
                    star_index = starred_indices[0]
                    starred_target = t.cast(
                        ast.Starred, target.elts[star_index]
                    )

                    # Handle elements before the starred expression
                    before_elements = target.elts[:star_index]
                    for tgt in before_elements:
                        try:
                            self._handle_unpacking_target(tgt, next(iter_value))
                        except StopIteration as e:
                            raise ValueError(
                                "Not enough values to unpack"
                            ) from e

                    # Collect remaining elements for the starred target
                    starred_values = list(iter_value)

                    # Calculate how many elements should be in the starred part
                    # after the current group
                    after_star_count = len(target.elts) - star_index - 1

                    # If there are more elements after the starred part
                    if after_star_count > 0:
                        # Make sure there are enough elements
                        if len(starred_values) < after_star_count:
                            raise ValueError("Not enough values to unpack")

                        # Separate starred values
                        starred_list = starred_values[:-after_star_count]
                        after_star_values = starred_values[-after_star_count:]

                        # Assign starred target
                        if isinstance(starred_target.value, ast.Name):
                            self._set_name_value(
                                starred_target.value.id, starred_list
                            )

                        # Assign elements after starred
                        after_elements = target.elts[star_index + 1 :]
                        for tgt, val in zip(after_elements, after_star_values):
                            self._handle_unpacking_target(tgt, val)
                    else:
                        # If no elements after starred, just assign the rest
                        # to the starred target
                        if isinstance(starred_target.value, ast.Name):
                            self._set_name_value(
                                starred_target.value.id, starred_values
                            )
                else:
                    # Standard unpacking without starred expression
                    value_list = list(value)
                    if len(value_list) < len(target.elts):
                        raise ValueError("Not enough values to unpack")
                    elif len(value_list) > len(target.elts):
                        raise ValueError("Too many values to unpack")

                    # Unpack each element
                    for tgt, val in zip(target.elts, value):
                        self._handle_unpacking_target(tgt, val)
            except (TypeError, ValueError) as e:
                raise UnsupportedUnpackingError(str(e)) from e
        else:
            raise UnsupportedUnpackingError(
                f"Unsupported unpacking target type: {type(target).__name__}"
            )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> t.Any:
        """Handle named expressions (walrus operator).

        Example: (x := 1) assigns 1 to x and returns 1
        """
        value = self.visit(node.value)

        # The target should be a Name node
        if not isinstance(node.target, ast.Name):
            raise SyntaxError("Invalid target for named expression")

        # Named expressions bind in the containing scope
        if len(self.scope_stack) > 1:
            # If we're in a nested scope, add to the parent scope
            parent_scope = self.scope_stack[-2]
            parent_scope.locals.add(node.target.id)
        else:
            # In the global scope, add to current scope
            self.current_scope.locals.add(node.target.id)

        # Set the value in the current environment
        self.local_env[node.target.id] = value
        return value

    def visit_BoolOp(self, node: ast.BoolOp) -> t.Any:
        """Handle logical AND and OR with Python's short-circuit behavior."""
        if isinstance(node.op, ast.And):
            # "and" should return the first falsy value, or the last value if
            # all are truthy
            result = True
            for value_node in node.values:
                result = self.visit(value_node)
                if not result:
                    return result  # Short-circuit on falsy
            return result
        elif isinstance(node.op, ast.Or):
            # "or" should return the first truthy value, or the last value if
            # all are falsy
            result = False
            for value_node in node.values:
                result = self.visit(value_node)
                if result:
                    return result  # Short-circuit on truthy
            return result
        else:
            raise NotImplementedError(
                f"Unsupported BoolOp operator: {type(node.op).__name__}"
            )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> t.Any:
        operand = self.visit(node.operand)

        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.Not):
            return not operand
        elif isinstance(node.op, ast.Invert):
            return ~operand
        else:
            raise NotImplementedError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )

    def visit_BinOp(self, node: ast.BinOp) -> t.Any:
        left = self.visit(node.left)
        right = self.visit(node.right)

        try:
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
            elif isinstance(node.op, ast.FloorDiv):
                return left // right
            elif isinstance(node.op, ast.Mod):
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left**right
            else:
                raise NotImplementedError(
                    f"Unsupported binary operator: {type(node.op).__name__}"
                )
        except (ZeroDivisionError, TypeError, ValueError) as e:
            raise type(e)(str(e)) from e

    _COMPARE_OP_MAP: t.Dict[t.Type[ast.cmpop], t.Callable] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
    }

    def visit_Compare(self, node: ast.Compare) -> bool:
        try:
            left = self.visit(node.left)

            for op, comparator in zip(node.ops, node.comparators):
                right = self.visit(comparator)
                op_func = self._COMPARE_OP_MAP.get(type(op))
                if op_func is None:
                    raise NotImplementedError(
                        f"Unsupported comparison operator: {type(op).__name__}"
                    )

                if not op_func(left, right):
                    return False
                left = right

            return True
        except TypeError as e:
            raise TypeError(f"Invalid comparison: {str(e)}") from e

    def visit_Try(self, node: ast.Try) -> None:
        try:
            for stmt in node.body:
                self.visit(stmt)
        except Exception as e:
            handled = False
            for handler in node.handlers:
                if handler.type is None:
                    exc_class = Exception
                else:
                    exc_class = self._get_exception_class(handler.type)

                if isinstance(e, exc_class):
                    handled = True
                    if handler.name is not None:
                        self.local_env[handler.name] = e
                    for stmt in handler.body:
                        self.visit(stmt)
                    if handler.name is not None:
                        del self.local_env[handler.name]
                    break

            if not handled:
                raise e
        else:
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
        finally:
            if node.finalbody:
                for stmt in node.finalbody:
                    self.visit(stmt)

    def _get_exception_class(self, node: ast.expr) -> t.Type[Exception]:
        if isinstance(node, ast.Name):
            class_name = node.id
            if class_name in globals()["__builtins__"]:
                exc_class = globals()["__builtins__"][class_name]
                if isinstance(exc_class, type) and issubclass(
                    exc_class, Exception
                ):
                    return exc_class
            raise NameError(
                f"Name '{class_name}' is not defined or is not an exception "
                "class"
            )
        elif isinstance(node, ast.Attribute):
            value = self.visit(node.value)
            attr = node.attr
            if hasattr(value, attr):
                exc_class = getattr(value, attr)
                if isinstance(exc_class, type) and issubclass(
                    exc_class, Exception
                ):
                    return exc_class
            raise NameError(f"'{attr}' is not a valid exception class")
        else:
            raise TypeError(
                f"Invalid exception class specification: {ast.dump(node)}"
            )

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is None:
            raise RuntimeError("No active exception to re-raise")

        exc = self.visit(node.exc)
        if isinstance(exc, type) and issubclass(exc, Exception):
            if node.cause:
                cause = self.visit(node.cause)
                raise exc from cause
            raise exc()
        else:
            if isinstance(exc, BaseException):
                raise exc
            else:
                raise TypeError(
                    f"Expected an exception instance, got {type(exc).__name__}"
                )

    def visit_With(self, node: ast.With) -> None:
        """Execute a with statement, properly handling scopes and context
        managers.

        Args:
            node: The With statement AST node

        The with statement creates a new scope for its body and properly manages
        multiple context managers, handling any exceptions that may occur during
        execution.
        """
        # Push a new scope for the with block
        new_scope = Scope()
        self.scope_stack.append(new_scope)

        # Save current environment state
        saved_env = self.local_env.copy()
        with_env = {}  # Environment for variables defined in this with block

        # List to track context managers and their values
        context_managers = []

        try:
            # Enter all context managers in order
            for item in node.items:
                try:
                    # Evaluate the context manager expression
                    context_manager = self.visit(item.context_expr)

                    try:
                        # Enter the context manager
                        value = context_manager.__enter__()
                        context_managers.append((context_manager, value))

                        # Handle the optional 'as' variable if present
                        if item.optional_vars is not None:
                            # Add the 'as' variable to the with block's
                            # environment
                            name = self.visit(item.optional_vars)
                            with_env[name] = value
                            self.current_scope.locals.add(name)
                    except Exception as enter_exc:
                        # If __enter__ fails, properly clean up previous context
                        # managers
                        for mgr, _ in reversed(context_managers[:-1]):
                            try:
                                mgr.__exit__(None, None, None)
                            except Exception:
                                # Ignore any cleanup exceptions
                                pass
                        raise enter_exc
                except Exception as ctx_exc:
                    # Clean up any successfully entered context managers
                    self._exit_context_managers(context_managers, ctx_exc)
                    raise ctx_exc

            # Update local environment with with block's environment
            self.local_env = {**saved_env, **with_env}

            try:
                # Execute the body of the with statement
                for stmt in node.body:
                    self.visit(stmt)
            except Exception as body_exc:
                # Handle any exception from the body
                if not self._exit_context_managers(context_managers, body_exc):
                    raise body_exc
            else:
                # No exception occurred, exit context managers normally
                self._exit_context_managers(context_managers, None)
        finally:
            # Restore the original environment and pop the scope
            # Only keep variables that were in the original environment
            self.local_env = {
                k: v
                for k, v in self.local_env.items()
                if k in saved_env or k not in self.current_scope.locals
            }
            self.scope_stack.pop()

    def _exit_context_managers(
        self,
        context_managers: list[tuple[t.Any, t.Any]],
        exc_info: t.Optional[Exception],
    ) -> bool:
        """Exit a list of context managers, handling any exceptions.

        Args:
            context_managers: List of (context_manager, value) pairs to exit
            exc_info: The exception that occurred, if any

        Returns:
            bool: True if any context manager suppressed the exception
        """
        # Track if any context manager suppresses the exception
        suppressed = False

        if exc_info is not None:
            exc_type = type(exc_info)
            exc_value = exc_info
            exc_tb = exc_info.__traceback__
        else:
            exc_type = None
            exc_value = None
            exc_tb = None

        # Exit context managers in reverse order
        for cm, _ in reversed(context_managers):
            try:
                if cm.__exit__(exc_type, exc_value, exc_tb):
                    suppressed = True
                    exc_type = None
                    exc_value = None
                    exc_tb = None
            except Exception as exit_exc:
                # If __exit__ raises an exception, update the exception info
                exc_type = type(exit_exc)
                exc_value = exit_exc
                exc_tb = exit_exc.__traceback__
                suppressed = False

        return suppressed

    def visit_If(self, node: ast.If) -> None:
        if self.visit(node.test):
            for stmt in node.body:
                self.visit(stmt)
        elif node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_IfExp(self, node: ast.IfExp) -> t.Any:
        # Ternary expression: <body> if <test> else <orelse>
        condition = self.visit(node.test)
        if condition:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Pass(
        self, node: ast.Pass  # pylint: disable=unused-argument
    ) -> None:
        """
        Handle the Pass statement.

        The Pass statement is a no-operation statement that does nothing.
        It's used as a placeholder when syntactically a statement is required
        but no action is desired.

        Args:
            node (ast.Pass): The Pass statement AST node

        Returns:
            None
        """
        # Do nothing, which is exactly what Pass is supposed to do
        return None

    def visit_Break(
        self, node: ast.Break  # pylint: disable=unused-argument
    ) -> None:
        """Handle break statement."""
        if self.control.loop_depth == 0:
            raise SyntaxError("'break' outside loop")
        raise BreakLoop()

    def visit_Continue(
        self, node: ast.Continue  # pylint: disable=unused-argument
    ) -> None:
        """Handle continue statement."""
        if self.control.loop_depth == 0:
            raise SyntaxError("'continue' outside loop")
        raise ContinueLoop()

    def visit_While(self, node: ast.While) -> None:
        self.control.loop_depth += 1

        try:
            while True:
                test_result = self.visit(node.test)  # Evaluate test first
                if not test_result:
                    break

                try:
                    for stmt in node.body:
                        try:
                            self.visit(stmt)
                        except ContinueLoop:
                            break
                    else:
                        # This else block is executed if no break occurred
                        continue
                except BreakLoop:
                    break
                except ReturnValue as rv:
                    raise rv
                except Exception as e:
                    if node.orelse:
                        for stmt in node.orelse:
                            self.visit(stmt)
                    raise e
        finally:
            self.control.loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self.control.loop_depth += 1

        iter_value = self.visit(node.iter)

        try:
            for item in iter_value:
                # Use the unpacking method to handle the target
                self._handle_unpacking_target(node.target, item)

                try:
                    for stmt in node.body:
                        try:
                            self.visit(stmt)
                        except ContinueLoop:
                            break
                    else:
                        # This else block is executed if no break occurred
                        continue
                except BreakLoop:
                    break
                except ReturnValue as rv:
                    raise rv
        except Exception as e:
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
            raise e
        finally:
            self.control.loop_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Handle function definition with support for named parameters, defaults,
        keyword-only, *args, and **kwargs.
        """
        def_scope = Scope()
        self.scope_stack.append(def_scope)

        try:
            # Validate nonlocal declarations at function definition time
            for stmt in node.body:
                if isinstance(stmt, ast.Nonlocal):
                    if len(self.scope_stack) < 2:
                        raise SyntaxError(
                            "nonlocal declaration not allowed at module level"
                        )
                    # For non-nested functions, check bindings at definition
                    # time
                    if len(self.scope_stack) == 2:  # Only one outer scope
                        for name in stmt.names:
                            found = False
                            # Check all outer scopes
                            # (excluding the current scope)
                            for scope in reversed(self.scope_stack[:-1]):
                                if (
                                    name in scope.locals
                                    or name in scope.nonlocals
                                    or name in self.local_env
                                ):
                                    found = True
                                    break
                            if not found:
                                raise SyntaxError(
                                    f"No binding for nonlocal '{name}' "
                                    "found in outer scopes"
                                )

            closure_env: t.Dict[str, t.Any] = {}
            outer_env: t.Dict[str, t.Any] = self.local_env

            # Precompute default values for positional and kw-only
            defaults = [self.visit(d) for d in node.args.defaults]
            kw_defaults = [
                None if d is None else self.visit(d)
                for d in node.args.kw_defaults
            ]

            # e.g. if we have 3 positional params and 1 default
            # => required_count=2
            required_count = len(node.args.args) - len(defaults)

            def func(*call_args, **call_kwargs):
                # Create a new execution scope
                func_scope = Scope()
                self.scope_stack.append(func_scope)

                prev_env = self.local_env
                # Build local env from outer + closure
                self.local_env = {**outer_env, **closure_env}

                self.control.function_depth += 1

                try:
                    # Register the function name itself for recursion
                    self.local_env[node.name] = func

                    # Process nonlocal declarations at function execution time
                    for stmt in node.body:
                        if isinstance(stmt, ast.Nonlocal):
                            for name in stmt.names:
                                found = False
                                # Check all outer scopes
                                # (excluding the current scope)
                                for scope in reversed(self.scope_stack[:-1]):
                                    if (
                                        name in scope.locals
                                        or name in scope.nonlocals
                                        or name in self.local_env
                                    ):
                                        found = True
                                        break
                                if not found:
                                    raise SyntaxError(
                                        f"No binding for nonlocal '{name}' "
                                        "found in outer scopes"
                                    )
                                # Mark this name as nonlocal in the current
                                # scope
                                self.current_scope.nonlocals.add(name)

                    ###############################
                    # 1) Bind the "regular" positional params
                    ###############################
                    positional_params = node.args.args
                    bound_args_count = min(
                        len(call_args), len(positional_params)
                    )

                    # First, match positional_args with call_args 1:1 from the
                    # front
                    for i in range(bound_args_count):
                        param = positional_params[i]
                        self._set_name_value(param.arg, call_args[i])

                    # Then, for leftover positional params, we check defaults
                    # or keywords
                    for i in range(bound_args_count, len(positional_params)):
                        param = positional_params[i]
                        param_name = param.arg

                        if i < required_count:
                            # This param must be provided either by leftover
                            # call_args (already exhausted) or by a keyword
                            if param_name in call_kwargs:
                                self._set_name_value(
                                    param_name, call_kwargs.pop(param_name)
                                )
                            else:
                                raise TypeError(
                                    f"{node.name}() missing required positional"
                                    f" argument: '{param_name}'"
                                )
                        else:
                            # This param has a default
                            default_index = i - required_count
                            if param_name in call_kwargs:
                                # Use the user-provided keyword
                                self._set_name_value(
                                    param_name, call_kwargs.pop(param_name)
                                )
                            else:
                                # Use the default
                                self._set_name_value(
                                    param_name, defaults[default_index]
                                )

                    ###############################
                    # 2) Handle keyword-only params
                    ###############################
                    # e.g. def func(x, *, y=10, z)
                    kwonly_params = node.args.kwonlyargs
                    for i, kw_param in enumerate(kwonly_params):
                        pname = kw_param.arg
                        if pname in call_kwargs:
                            self._set_name_value(pname, call_kwargs.pop(pname))
                        else:
                            # if there's a default => use it; else error
                            if kw_defaults[i] is not None:
                                self._set_name_value(pname, kw_defaults[i])
                            else:
                                raise TypeError(
                                    f"{node.name}() missing required "
                                    f"keyword-only argument: '{pname}'"
                                )

                    ###############################
                    # 3) *args (vararg)
                    ###############################
                    if node.args.vararg:
                        vararg_name = node.args.vararg.arg
                        leftover = call_args[len(positional_params) :]
                        self._set_name_value(vararg_name, leftover)
                    else:
                        # If no vararg, but user gave extra positional => error
                        if len(call_args) > len(positional_params):
                            raise TypeError(
                                f"{node.name}() takes {len(positional_params)} "
                                f"positional arguments but {len(call_args)} "
                                f"were given"
                            )

                    ###############################
                    # 4) **kwargs
                    ###############################
                    if node.args.kwarg:
                        kwarg_name = node.args.kwarg.arg
                        self._set_name_value(kwarg_name, call_kwargs)
                    else:
                        if call_kwargs:
                            first_unexpected = next(iter(call_kwargs))
                            raise TypeError(
                                f"{node.name}() got an unexpected keyword "
                                f"argument: '{first_unexpected}'"
                            )

                    ###############################
                    # 5) Execute function body
                    ###############################
                    try:
                        for stmt in node.body:
                            self.visit(stmt)
                        return None
                    except ReturnValue as rv:
                        return rv.value
                finally:
                    self.control.function_depth -= 1

                    # Update nonlocals
                    for name in func_scope.nonlocals:
                        if name in self.local_env:
                            closure_env[name] = self.local_env[name]
                            outer_env[name] = self.local_env[name]

                    self.local_env = prev_env
                    self.scope_stack.pop()

            # Register the function in the current scope
            self._set_name_value(node.name, func)
        finally:
            self.scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> t.Callable:
        closure_env: t.Dict[str, t.Any] = {}
        outer_env: t.Dict[str, t.Any] = self.local_env

        defaults = [self.visit(d) for d in node.args.defaults]
        kw_defaults = [
            None if d is None else self.visit(d) for d in node.args.kw_defaults
        ]
        required_count = len(node.args.args) - len(defaults)

        def lambda_func(*call_args, **call_kwargs):
            lambda_scope = Scope()
            self.scope_stack.append(lambda_scope)

            prev_env = self.local_env
            self.local_env = {**outer_env, **closure_env}

            try:
                # 1) Bind positional
                positional_params = node.args.args
                bound_args_count = min(len(call_args), len(positional_params))
                for i in range(bound_args_count):
                    param = positional_params[i]
                    self._set_name_value(param.arg, call_args[i])

                # leftover positional -> defaults or error
                for i in range(bound_args_count, len(positional_params)):
                    param = positional_params[i]
                    pname = param.arg
                    if i < required_count:
                        # must come from keyword
                        if pname in call_kwargs:
                            self._set_name_value(pname, call_kwargs.pop(pname))
                        else:
                            raise TypeError(
                                "Lambda missing required positional argument: "
                                f"'{pname}'"
                            )
                    else:
                        # default
                        d_index = i - required_count
                        if pname in call_kwargs:
                            self._set_name_value(pname, call_kwargs.pop(pname))
                        else:
                            self._set_name_value(pname, defaults[d_index])

                # 2) kw-only
                kwonly_params = node.args.kwonlyargs
                for i, kwp in enumerate(kwonly_params):
                    pname = kwp.arg
                    if pname in call_kwargs:
                        self._set_name_value(pname, call_kwargs.pop(pname))
                    else:
                        if kw_defaults[i] is not None:
                            self._set_name_value(pname, kw_defaults[i])
                        else:
                            raise TypeError(
                                "Lambda missing required keyword-only "
                                f"argument: '{pname}'"
                            )

                # 3) *args
                if node.args.vararg:
                    vararg_name = node.args.vararg.arg
                    leftover = call_args[len(positional_params) :]
                    self._set_name_value(vararg_name, leftover)
                else:
                    if len(call_args) > len(positional_params):
                        raise TypeError(
                            f"Lambda takes {len(positional_params)} positional "
                            f"arguments but {len(call_args)} were given"
                        )

                # 4) **kwargs
                if node.args.kwarg:
                    kwarg_name = node.args.kwarg.arg
                    self._set_name_value(kwarg_name, call_kwargs)
                else:
                    if call_kwargs:
                        first_unexpected = next(iter(call_kwargs))
                        raise TypeError(
                            "Lambda got an unexpected keyword argument: "
                            f"'{first_unexpected}'"
                        )

                # 5) Evaluate the body
                result = self.visit(node.body)

                # Update nonlocals
                for name in lambda_scope.nonlocals:
                    if name in self.local_env:
                        closure_env[name] = self.local_env[name]
                        outer_env[name] = self.local_env[name]

                return result
            finally:
                self.local_env = prev_env
                self.scope_stack.pop()

        return lambda_func

    def visit_Return(self, node: ast.Return) -> None:
        if self.control.function_depth == 0:
            raise SyntaxError("'return' outside function")

        value = None if node.value is None else self.visit(node.value)
        raise ReturnValue(value)

    def visit_Call(self, node: ast.Call) -> t.Any:
        """
        Handle function calls, including positional args, keyword args, and
        **kwargs.
        """
        # Evaluate the function object
        func = self.visit(node.func)
        pos_args = [self.visit(arg) for arg in node.args]

        # Evaluate keyword arguments
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                # This is the case of f(**some_dict)
                dict_val = self.visit(kw.value)
                if not isinstance(dict_val, dict):
                    raise TypeError(
                        "Argument after ** must be a dict, got "
                        f"{type(dict_val).__name__}"
                    )
                # Merge into our kwargs
                for k, v in dict_val.items():
                    if not isinstance(k, str):
                        raise TypeError("Keywords must be strings")
                    kwargs[k] = v
            else:
                # Normal keyword argument f(key=value)
                key_name = kw.arg
                value = self.visit(kw.value)
                kwargs[key_name] = value

        # Check if the function is callable
        if not callable(func):
            raise TypeError(f"'{type(func).__name__}' object is not callable")

        # Handle registered functions
        if registry.is_registered(func):
            return func(*pos_args, **kwargs)

        # Handle bound methods
        if isinstance(func, types.MethodType):
            return func(*pos_args, **kwargs)

        # Handle normal functions
        return func(*pos_args, **kwargs)

    def visit_List(self, node: ast.List) -> list:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Set(self, node: ast.Set) -> set:
        return {self.visit(elt) for elt in node.elts}

    def visit_Dict(self, node: ast.Dict) -> dict:
        return {
            self.visit(key) if key is not None else None: self.visit(value)
            for key, value in zip(node.keys, node.values)
        }

    def visit_ListComp(self, node: ast.ListComp) -> list:
        return self._handle_comprehension(node, list)

    def visit_SetComp(self, node: ast.SetComp) -> set:
        return self._handle_comprehension(node, set)

    def visit_DictComp(self, node: ast.DictComp) -> dict:
        # Preserve the current local environment
        outer_env = self.local_env.copy()
        self.local_env = self.local_env.copy()

        try:
            result = {}
            for generator in node.generators:
                iter_obj = self.visit(generator.iter)

                for item in iter_obj:
                    try:
                        self._handle_unpacking_target(generator.target, item)
                    except UnsupportedUnpackingError:
                        # If unpacking fails, fallback to simple assignment
                        if isinstance(generator.target, ast.Name):
                            self.local_env[generator.target.id] = item
                        else:
                            raise

                    # Check if all conditions are met
                    if all(
                        self.visit(if_clause) for if_clause in generator.ifs
                    ):
                        # Evaluate key and value
                        key = self.visit(node.key)
                        value = self.visit(node.value)
                        result[key] = value

            return result
        finally:
            # Restore the original local environment
            self.local_env = outer_env

    T = t.TypeVar("T", list, set)

    def _handle_comprehension(
        self, node: t.Union[ast.ListComp, ast.SetComp], result_type: t.Type[T]
    ) -> T:
        # Create new scope for the comprehension
        comp_scope = Scope()
        self.scope_stack.append(comp_scope)

        # Copy the outer environment
        outer_env = self.local_env
        self.local_env = outer_env.copy()

        try:
            result = []

            def process_generator(generators: list, index: int = 0):
                if index >= len(generators):
                    # Base case: all generators processed, evaluate element
                    value = self.visit(node.elt)
                    result.append(value)
                    return

                generator = generators[index]
                iter_obj = self.visit(generator.iter)

                # Save the current environment before processing this generator
                current_env = self.local_env.copy()

                for item in iter_obj:
                    # Restore environment from before this generator's loop
                    self.local_env = current_env.copy()

                    try:
                        self._handle_unpacking_target(generator.target, item)
                    except UnsupportedUnpackingError:
                        if isinstance(generator.target, ast.Name):
                            self._set_name_value(generator.target.id, item)
                        else:
                            raise

                    # Check if conditions
                    # Named expressions in conditions should affect outer scope
                    if all(
                        self.visit(if_clause) for if_clause in generator.ifs
                    ):
                        # Process next generator or append result
                        process_generator(generators, index + 1)

                    # Update outer environment with any named expression
                    # bindings
                    for name, value in self.local_env.items():
                        if name not in current_env:
                            outer_env[name] = value

            # Start processing generators recursively
            process_generator(node.generators)
            return result_type(result)
        finally:
            # Restore the outer environment and pop the scope
            self.local_env = outer_env
            self.scope_stack.pop()

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(self._format_value(value))
            else:
                raise NotImplementedError(
                    f"Unsupported node type in f-string: {type(value).__name__}"
                )
        return "".join(parts)

    def _format_value(self, node: ast.FormattedValue) -> str:
        """Format a single value in an f-string.

        Args:
            node: FormattedValue AST node

        Returns:
            Formatted string representation of the value

        Raises:
            NotImplementedError: If unsupported conversion or format_spec is
            used
        """
        # Evaluate the expression
        value = self.visit(node.value)

        # Handle conversion specifier (s, r, a)
        if node.conversion == -1:  # No conversion
            converted = value
        elif node.conversion == 115:  # 's' for str()
            converted = str(value)
        elif node.conversion == 114:  # 'r' for repr()
            converted = repr(value)
        elif node.conversion == 97:  # 'a' for ascii()
            converted = ascii(value)
        else:
            raise NotImplementedError(
                f"Unsupported conversion type in f-string: {node.conversion}"
            )

        # Handle format specification
        if node.format_spec is None:
            format_spec = ""
        else:
            # Format spec can itself be an f-string
            format_spec = self.visit(node.format_spec)

        try:
            # Apply the format specification
            if format_spec:
                result = format(converted, format_spec)
            else:
                result = format(converted)
            return result
        except ValueError as e:
            raise ValueError(
                f"Invalid format specification '{format_spec}' "
                f"for value {repr(value)} of type {type(value).__name__}"
            ) from e

    def visit_Attribute(self, node: ast.Attribute) -> t.Any:
        """Visit an attribute access node.

        Args:
            node: The Attribute AST node

        Returns:
            The value of the attribute

        Raises:
            SecurityError: If accessing a forbidden attribute
            AttributeError: If the attribute doesn't exist
        """
        if node.attr in self.FORBIDDEN_ATTRS:
            raise SecurityError(
                f"Access to '{node.attr}' attribute is not allowed"
            )

        value = self.visit(node.value)

        # Handle normal attribute access
        try:
            attr = getattr(value, node.attr)
        except AttributeError as e:
            raise AttributeError(
                f"'{type(value).__name__}' object has no attribute "
                f"'{node.attr}'"
            ) from e

        # If this is a function defined in our namespace, bind it to the
        # instance
        if isinstance(attr, types.FunctionType):
            # Check if this is a registered function
            if registry.is_registered(attr):
                return attr

            # Check if this is a static method
            if isinstance(value, type):
                # If accessed on class, check for static marker
                if getattr(value, f"__static_{node.attr}", False):
                    return attr
            else:
                # If accessed on instance, check the class
                if getattr(type(value), f"__static_{node.attr}", False):
                    return attr

            # If not a static method, check if it's a module function
            if isinstance(value, types.ModuleType):
                # For module functions, don't bind self
                return attr

            # Check if the function is in the global environment
            for global_value in self.global_env.values():
                if isinstance(global_value, dict):
                    # Check nested dictionaries
                    for nested_value in global_value.values():
                        if attr is nested_value:
                            return attr
                elif attr is global_value:
                    return attr

            # If not in global environment, bind it to the instance
            def bound_method(*args, **kwargs):
                prev_env = self.local_env

                try:
                    # Check if this is a static method - either by flag or by
                    # type
                    has_static_flag = getattr(
                        type(value), f"__static_{node.attr}", False
                    )
                    is_static_method = isinstance(attr, staticmethod)

                    # Check if the method is a function
                    is_function = isinstance(attr, types.FunctionType)

                    if has_static_flag or is_static_method:
                        # For static methods, don't bind to instance
                        self.local_env = prev_env
                        if is_static_method:
                            method = attr.__get__(None, type(value))
                            return method(*args, **kwargs)
                        return attr(*args, **kwargs)
                    elif isinstance(attr, classmethod):
                        self.local_env = prev_env
                        method = attr.__get__(type(value), type(value))
                        return method(*args, **kwargs)
                    elif is_function:
                        # If it's a function, call it directly
                        self.local_env = prev_env
                        return attr(*args, **kwargs)
                    else:
                        self.local_env = prev_env
                        method = attr.__get__(value, type(value))
                        return method(*args, **kwargs)
                finally:
                    self.local_env = prev_env

            return bound_method
        elif isinstance(attr, (classmethod, staticmethod, property)):
            # For decorated methods, get the underlying function
            func = attr.__get__(value, type(value))
            if isinstance(attr, (classmethod, property)):
                # For classmethod and property, we need to set up the
                # environment
                def decorated_method(*args, **kwargs):
                    prev_env = self.local_env
                    self.local_env = {**prev_env, "self": value}
                    try:
                        return func(*args, **kwargs)
                    finally:
                        self.local_env = prev_env

                return decorated_method
            else:
                # For staticmethod, just return the function as is
                return func

        return attr

    def visit_Subscript(self, node: ast.Subscript) -> t.Any:
        """Handle subscript operations with improved slice support.

        Args:
            node: Subscript AST node

        Returns:
            Value from the subscript operation

        Raises:
            TypeError: If subscript operation is invalid
            IndexError: If index is out of range
        """
        value = self.visit(node.value)

        # Handle different slice types
        if isinstance(node.slice, ast.Index):
            # For Python < 3.9 compatibility
            slice_val = self.visit(node.slice)
            return value[slice_val]
        elif isinstance(node.slice, ast.Slice):
            # Handle slice with start:stop:step syntax
            start = (
                self.visit(node.slice.lower)
                if node.slice.lower is not None
                else None
            )
            stop = (
                self.visit(node.slice.upper)
                if node.slice.upper is not None
                else None
            )
            step = (
                self.visit(node.slice.step)
                if node.slice.step is not None
                else None
            )
            return value[start:stop:step]
        else:
            # For Python >= 3.9, node.slice can be other expression nodes
            slice_val = self.visit(node.slice)
            try:
                return value[slice_val]
            except TypeError as e:
                raise TypeError(
                    f"Invalid subscript type: {type(slice_val).__name__}"
                ) from e

    def visit_Expr(self, node: ast.Expr) -> t.Any:
        """Visit an expression statement.

        Args:
            node: The Expr AST node

        Returns:
            The value of the expression
        """
        return self.visit(node.value)

    def visit_Expression(self, node: ast.Expression) -> t.Any:
        result = self.visit(node.body)
        self.global_env["_"] = result
        return result

    def visit_Module(self, node: ast.Module) -> t.Any:
        result = None
        for stmt in node.body:
            if isinstance(stmt, ast.Expr):
                # For expression statements, capture the value
                result = self.visit(stmt.value)
                self.global_env["_"] = result
            else:
                # For other statements, just execute them
                self.visit(stmt)
        return result

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definition with support for inheritance and class body.

        Args:
            node: The ClassDef AST node
        """
        # Create a new scope for the class definition
        class_scope = Scope()
        self.scope_stack.append(class_scope)

        try:
            # Evaluate base classes
            bases = tuple(self.visit(base) for base in node.bases)

            # Create namespace for class attributes
            namespace: t.Dict[str, t.Any] = {}

            # Save current environment
            prev_env = self.local_env
            self.local_env = namespace

            try:
                # Execute the class body
                for stmt in node.body:
                    if isinstance(stmt, ast.FunctionDef):
                        # Handle function definition
                        self.visit(stmt)

                        # Get the function from namespace
                        func = namespace[stmt.name]

                        # Handle decorators in reverse order
                        for decorator in reversed(stmt.decorator_list):
                            # Evaluate the decorator
                            decorator_func = self.visit(decorator)
                            # Apply the decorator
                            func = decorator_func(func)

                            # For static methods, we need to store both the
                            # decorator and the function
                            if decorator_func is staticmethod:
                                namespace[f"__static_{stmt.name}"] = True

                        # Update the function in namespace
                        namespace[stmt.name] = func
                    else:
                        self.visit(stmt)
            finally:
                # Restore the environment
                self.local_env = prev_env

            # Create a custom super implementation for this class
            def custom_super(cls=None, obj_or_type=None):
                if cls is None and obj_or_type is None:
                    # Handle zero-argument super() by finding the calling class
                    # and instance from the current scope
                    if "self" in self.local_env:
                        obj_or_type = self.local_env["self"]
                        cls = class_obj
                    else:
                        raise RuntimeError(
                            "super(): no arguments and no context - unable to "
                            "determine class and instance"
                        )
                elif cls is None:
                    # Handle one-argument super()
                    if obj_or_type is None:
                        raise TypeError("super() argument 1 cannot be None")
                    cls = type(obj_or_type)

                if obj_or_type is None:
                    raise TypeError("super() argument 2 cannot be None")

                # Find the next class in the MRO after cls
                mro = (
                    obj_or_type.__class__.__mro__
                    if isinstance(obj_or_type, object)
                    else obj_or_type.__mro__
                )
                for i, base in enumerate(mro):
                    if base is cls:
                        if i + 1 < len(mro):
                            next_class = mro[i + 1]

                            # Create a bound method that will bind self to the
                            # method
                            def bound_super_method(
                                name, current_class=next_class
                            ):
                                method = getattr(current_class, name)
                                if isinstance(
                                    method, (staticmethod, classmethod)
                                ):
                                    return method.__get__(
                                        obj_or_type, current_class
                                    )
                                else:
                                    return method.__get__(
                                        obj_or_type, current_class
                                    )

                            # Create a new class with a __getattr__ method that
                            # will bind self to the method
                            params = {
                                "__getattr__": (
                                    lambda _, name, method=bound_super_method: (
                                        method(name)
                                    )
                                )
                            }
                            return type("Super", (), params)()
                        break
                raise RuntimeError("super(): bad __mro__")

            # Add custom super to the class namespace
            namespace["super"] = custom_super

            # Set the module name for the class
            namespace["__module__"] = "monic.expressions.registry"

            # Create the class object
            class_obj = types.new_class(
                node.name, bases, {}, lambda ns: ns.update(namespace)
            )

            # Register the class in the current scope
            self._set_name_value(node.name, class_obj)
        finally:
            self.scope_stack.pop()

    if sys.version_info >= (3, 10):

        def visit_Match(self, node: ast.Match) -> None:
            """Handle match-case statements.

            Args:
                node: Match AST node

            Example:
                match value:
                    case 1:
                        ...
                    case [x, y]:
                        ...
                    case {"key": value}:
                        ...
                    case _:
                        ...
            """
            # Evaluate the subject expression
            subject = self.visit(node.subject)

            # Create a new scope for pattern matching
            match_scope = Scope()
            self.scope_stack.append(match_scope)

            try:
                # Try each case in order
                for case in node.cases:
                    pattern = case.pattern

                    # Create a temporary scope for pattern matching
                    temp_scope = Scope()
                    self.scope_stack.append(temp_scope)

                    try:
                        if self._match_pattern(pattern, subject):
                            # If there's a guard, evaluate it
                            if case.guard is not None:
                                guard_result = self.visit(case.guard)
                                if not guard_result:
                                    continue

                            # Copy matched variables from temp scope to match
                            # scope
                            for name in temp_scope.locals:
                                if name in self.local_env:
                                    self._set_name_value(
                                        name, self.local_env[name]
                                    )

                            # Execute the case body
                            for stmt in case.body:
                                self.visit(stmt)
                            return
                    finally:
                        self.scope_stack.pop()
            finally:
                self.scope_stack.pop()

        def _match_pattern(
            self,
            pattern: ast.pattern,
            value: t.Any,
        ) -> bool:
            """Match a pattern against a value.

            Args:
                pattern: Pattern AST node
                value: Value to match against

            Returns:
                bool: Whether the pattern matches the value
            """
            if isinstance(pattern, ast.MatchValue):
                # Literal pattern: case 1: or case "string": etc.
                pattern_value = self.visit(pattern.value)
                return (
                    type(value) is type(pattern_value)
                    and value == pattern_value
                )
            elif isinstance(pattern, ast.MatchSingleton):
                # Singleton pattern: case None: or case True: etc.
                return value is pattern.value
            elif isinstance(pattern, ast.MatchSequence):
                # Sequence pattern: case [x, y]:
                if not isinstance(value, (list, tuple)):
                    return False

                # Handle star patterns
                star_idx = -1
                for i, p in enumerate(pattern.patterns):
                    if isinstance(p, ast.MatchStar):
                        star_idx = i
                        break

                if star_idx == -1:
                    # No star pattern
                    if len(pattern.patterns) != len(value):
                        return False
                    # Match each element
                    for p, v in zip(pattern.patterns, value):
                        if not self._match_pattern(p, v):
                            return False
                    return True
                else:
                    # Has star pattern
                    if len(value) < len(pattern.patterns) - 1:
                        return False
                    # Match patterns before star
                    for p, v in zip(
                        pattern.patterns[:star_idx], value[:star_idx]
                    ):
                        if not self._match_pattern(p, v):
                            return False

                    # Match patterns after star
                    remaining_count = len(pattern.patterns) - star_idx - 1
                    for p, v in zip(
                        pattern.patterns[star_idx + 1 :],
                        value[-remaining_count:] if remaining_count > 0 else [],
                    ):
                        if not self._match_pattern(p, v):
                            return False

                    # Bind star pattern if it has a name
                    star_pattern = pattern.patterns[star_idx]
                    if (
                        isinstance(star_pattern, ast.MatchStar)
                        and star_pattern.name
                    ):
                        star_value = (
                            list(value[star_idx:-remaining_count])
                            if remaining_count > 0
                            else list(value[star_idx:])
                        )
                        self._set_name_value(star_pattern.name, star_value)
                        self.current_scope.locals.add(star_pattern.name)

                    return True
            elif isinstance(pattern, ast.MatchMapping):
                # Mapping pattern: case {"key": value}:
                if not isinstance(value, dict):
                    return False

                # Check if all required keys are present
                for key in pattern.keys:
                    key_value = self.visit(key)
                    if key_value not in value:
                        return False

                # Match each key-pattern pair
                for key, pat in zip(pattern.keys, pattern.patterns):
                    key_value = self.visit(key)
                    if not self._match_pattern(pat, value[key_value]):
                        return False

                # Handle rest pattern if present
                if pattern.rest is not None:
                    rest_dict = {
                        k: v
                        for k, v in value.items()
                        if not any(self.visit(key) == k for key in pattern.keys)
                    }
                    self._set_name_value(pattern.rest, rest_dict)
                    self.current_scope.locals.add(pattern.rest)

                return True
            elif isinstance(pattern, ast.MatchStar):
                # Star pattern: case [x, *rest, y]:
                # This is handled by MatchSequence
                if pattern.name is not None:
                    self._set_name_value(pattern.name, value)
                    self.current_scope.locals.add(pattern.name)
                return True
            elif isinstance(pattern, ast.MatchAs):
                # AS pattern: case x: or case [x] as lst:
                if pattern.pattern is not None:
                    if not self._match_pattern(pattern.pattern, value):
                        return False
                if pattern.name is not None:
                    self._set_name_value(pattern.name, value)
                    self.current_scope.locals.add(pattern.name)
                return True
            elif isinstance(pattern, ast.MatchOr):
                # OR pattern: case 1 | 2 | 3:
                for p in pattern.patterns:
                    # Create a temporary scope for each OR pattern
                    # to avoid variable binding conflicts
                    temp_scope = Scope()
                    self.scope_stack.append(temp_scope)
                    try:
                        if self._match_pattern(p, value):
                            return True
                    finally:
                        self.scope_stack.pop()
                return False
            elif isinstance(pattern, ast.MatchClass):
                # Class pattern: case Point(x, y):
                cls = self.visit(pattern.cls)
                if not isinstance(value, cls):
                    return False

                # Get positional attributes from __match_args__
                match_args = getattr(cls, "__match_args__", ())
                if len(pattern.patterns) > len(match_args):
                    return False

                # Match positional patterns
                for pat, attr_name in zip(pattern.patterns, match_args):
                    if not self._match_pattern(pat, getattr(value, attr_name)):
                        return False

                # Match keyword patterns
                for name, pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
                    if not hasattr(value, name):
                        return False
                    if not self._match_pattern(pat, getattr(value, name)):
                        return False

                return True

            return False
