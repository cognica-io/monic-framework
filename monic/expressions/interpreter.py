#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

# pylint: disable=no-else-return,no-else-raise,broad-except
# pylint: disable=too-many-branches,too-many-return-statements,too-many-locals
# pylint: disable=too-many-public-methods,too-many-instance-attributes

import ast
import datetime
import operator
import time
import typing as t

from dataclasses import dataclass, field

from monic.expressions.context import ExpressionContext
from monic.expressions.exceptions import (
    SecurityError,
    UnsupportedUnpackingError,
)


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

    in_loop: bool = False


class ExpressionInterpreter(ast.NodeVisitor):
    def __init__(self, context: t.Optional[ExpressionContext] = None) -> None:
        self.started_at = time.monotonic()

        self.context = context or ExpressionContext()
        self.scope_stack: t.List[Scope] = [Scope()]  # Track scopes
        self.control: ControlFlow = ControlFlow()

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
            # Modules
            "datetime": datetime,
            "time": time,
        }
        self.local_env: t.Dict[str, t.Any] = {}

        # Initialize last result storage
        self.global_env["_"] = None

        # List of forbidden functions and modules
        self.FORBIDDEN_NAMES = {
            "eval",
            "exec",
            "compile",
            "execfile",
            "open",
            "globals",
            "locals",
            "vars",
            "__import__",
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
            raise type(e)(f"Runtime error: {str(e)}") from e

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
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
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
        for node in ast.walk(node):
            # Check for forbidden function calls
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                raise SecurityError(f"Use of '{node.id}' is not allowed")

            # Check for forbidden attribute access
            if (
                isinstance(node, ast.Attribute)
                and node.attr in self.FORBIDDEN_ATTRS
            ):
                raise SecurityError(
                    f"Access to '{node.attr}' attribute is not allowed"
                )

            # Check for __builtins__ access
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                raise SecurityError(
                    "Access to '__builtins__' attribute is not allowed"
                )

            # Check for import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
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

        This marks the given names as nonlocal for the current scope,
        and ensures the names actually exist in an outer scope.
        """
        for name in node.names:
            # Mark as nonlocal in the current scope
            self.current_scope.nonlocals.add(name)

            # Ensure at least one outer scope has the name
            found = False
            for scope in reversed(self.scope_stack[:-1]):
                # If name is declared local or already nonlocal in that scope
                if (name in scope.locals) or (name in scope.nonlocals):
                    found = True
                    break
            if not found:
                raise SyntaxError(
                    f"No binding for nonlocal '{name}' found in outer scopes"
                )

    def visit_Constant(self, node: ast.Constant) -> t.Any:
        return node.value

    def _get_name_value(self, name: str) -> t.Any:
        """Get value of a name considering scope declarations."""
        if name in self.current_scope.globals:
            if name in self.global_env:
                return self.global_env[name]
            raise NameError(f"Global name '{name}' is not defined")
        elif name in self.current_scope.nonlocals:
            # Search for name in outer scopes
            for scope in reversed(self.scope_stack[:-1]):
                if name in scope.locals:
                    return self.local_env[name]
            raise NameError(f"Nonlocal name '{name}' is not defined")
        elif name in self.local_env:
            return self.local_env[name]
        elif name in self.global_env:
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

        # If declared nonlocal in the current scope, walk backward to find an
        # owner scope:
        if name in self.current_scope.nonlocals:
            for i in range(len(self.scope_stack) - 2, -1, -1):
                s = self.scope_stack[i]
                # If the name is recognized in that scope (locals or nonlocals)
                if name in s.locals or name in s.nonlocals:
                    # Now we need to store it in that scope's environment.
                    # If you track an environment dictionary per scope
                    # explicitly, you'd reference that here. e.g.
                    #   self.env_stack[i][name] = value
                    # If you only use self.local_env for all scopes, a minimal
                    # approach is to store it there:
                    self.local_env[name] = value
                    return

            # If we fail to find a scope with that name, it's an error
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
            return self._get_name_value(node.id)
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
                    if len(list(value)) != len(target.elts):
                        raise ValueError("Not enough values to unpack")

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

    def visit_Compare(self, node: ast.Compare) -> bool:
        try:
            left = self.visit(node.left)

            for op, comparator in zip(node.ops, node.comparators):
                right = self.visit(comparator)

                if isinstance(op, ast.Eq):
                    result = left == right
                elif isinstance(op, ast.NotEq):
                    result = left != right
                elif isinstance(op, ast.Lt):
                    result = left < right
                elif isinstance(op, ast.LtE):
                    result = left <= right
                elif isinstance(op, ast.Gt):
                    result = left > right
                elif isinstance(op, ast.GtE):
                    result = left >= right
                elif isinstance(op, ast.In):
                    result = left in right
                elif isinstance(op, ast.NotIn):
                    result = left not in right
                elif isinstance(op, ast.Is):
                    result = left is right
                elif isinstance(op, ast.IsNot):
                    result = left is not right
                else:
                    raise NotImplementedError(
                        f"Unsupported comparison operator: {type(op).__name__}"
                    )

                if not result:
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
                        value = (
                            context_manager.__enter__()  # pylint: disable=unnecessary-dunder-call
                        )
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
        exc_info: Exception | None,
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
        # Ternary expression:  <body> if <test> else <orelse>
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
        if not self.control.in_loop:
            raise SyntaxError("'break' outside loop")
        raise BreakLoop()

    def visit_Continue(
        self, node: ast.Continue  # pylint: disable=unused-argument
    ) -> None:
        """Handle continue statement."""
        if not self.control.in_loop:
            raise SyntaxError("'continue' outside loop")
        raise ContinueLoop()

    def visit_While(self, node: ast.While) -> None:
        prev_in_loop = self.control.in_loop
        self.control.in_loop = True

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
            self.control.in_loop = prev_in_loop

    def visit_For(self, node: ast.For) -> None:
        prev_in_loop = self.control.in_loop
        self.control.in_loop = True

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
            self.control.in_loop = prev_in_loop

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definition with proper closure environment."""
        # Create a new scope for the function definition
        def_scope = Scope()
        self.scope_stack.append(def_scope)

        try:
            # Capture the current environment for closure
            closure_env = self.local_env

            def func(*args):
                # Create new scope for function execution
                func_scope = Scope()
                self.scope_stack.append(func_scope)

                # Save current environment
                prev_env = self.local_env
                # Set up new environment with access to closure
                self.local_env = closure_env.copy()

                try:
                    # Add function to its environment for recursion
                    self.local_env[node.name] = func

                    # Bind parameters
                    for param, arg in zip(node.args.args, args):
                        self._set_name_value(param.arg, arg)

                    # Execute function body
                    try:
                        for stmt in node.body:
                            self.visit(stmt)
                        return None
                    except ReturnValue as rv:
                        return rv.value
                finally:
                    # If this function modified any nonlocal variables,
                    # update them in the closure environment
                    for name in func_scope.nonlocals:
                        if name in self.local_env:
                            closure_env[name] = self.local_env[name]

                    # Restore previous environment
                    self.local_env = prev_env
                    self.scope_stack.pop()

            # Store function in current scope
            self._set_name_value(node.name, func)
        finally:
            self.scope_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> t.Callable:
        """Handle lambda expression with proper scope creation."""
        # Capture the current environment for closure
        closure_env = self.local_env.copy()

        def lambda_func(*args):
            # Create new scope for lambda
            self.scope_stack.append(Scope())

            # Save previous environment
            previous_env = self.local_env
            self.local_env = closure_env.copy()

            try:
                # Assign parameters in lambda's scope
                for param, arg in zip(node.args.args, args):
                    self._set_name_value(param.arg, arg)

                return self.visit(node.body)
            finally:
                # Restore previous environment and pop scope
                self.local_env = previous_env
                self.scope_stack.pop()

        return lambda_func

    def visit_Return(self, node: ast.Return) -> None:
        value = None if node.value is None else self.visit(node.value)
        raise ReturnValue(value)

    def visit_Call(self, node: ast.Call) -> t.Any:
        # Get function object
        func = self.visit(node.func)

        # Get function name if possible
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check for forbidden functions
        if func_name in self.FORBIDDEN_NAMES:
            raise SecurityError(f"Call to '{func_name}' is not allowed")

        # Check actual function type
        if callable(func):
            func_type = type(func).__name__
            if func_type in {"builtin_function_or_method", "type"}:
                # Additional check for builtin functions
                if func.__name__ in self.FORBIDDEN_NAMES:
                    raise SecurityError(
                        f"Call to builtin '{func.__name__}' is not allowed"
                    )

        args = [self.visit(arg) for arg in node.args]
        return func(*args)

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
        self, node: ast.ListComp | ast.SetComp, result_type: t.Type[T]
    ) -> T:
        # Create new scope for the comprehension
        comp_scope = Scope()
        self.scope_stack.append(comp_scope)

        # Copy the outer environment so that we can track changes
        outer_env = self.local_env
        self.local_env = outer_env.copy()

        try:
            result = []
            # Process all generators (for n in numbers, etc.)
            for generator in node.generators:
                iter_obj = self.visit(generator.iter)
                for item in iter_obj:
                    try:
                        self._handle_unpacking_target(generator.target, item)
                    except UnsupportedUnpackingError:
                        # If unpacking fails, fallback to simple assignment
                        if isinstance(generator.target, ast.Name):
                            self._set_name_value(generator.target.id, item)
                        else:
                            raise

                    # Evaluate ifs
                    if all(
                        self.visit(if_clause) for if_clause in generator.ifs
                    ):
                        # Append the comprehension element
                        result.append(self.visit(node.elt))

            # Build the final list or set
            return result_type(result)
        finally:
            # Merge any new/updated bindings from the comprehension scope back
            # into the parent
            for key, val in self.local_env.items():
                outer_env[key] = val

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
        if node.attr in self.FORBIDDEN_ATTRS:
            raise SecurityError(
                f"Access to '{node.attr}' attribute is not allowed"
            )

        value = self.visit(node.value)
        return getattr(value, node.attr)

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
