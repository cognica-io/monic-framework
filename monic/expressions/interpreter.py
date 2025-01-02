#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

# pylint: disable=no-else-break,no-else-return,no-else-raise,broad-except
# pylint: disable=too-many-branches,too-many-return-statements,too-many-locals
# pylint: disable=too-many-public-methods,too-many-instance-attributes
# pylint: disable=too-many-statements,too-many-nested-blocks,too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=unnecessary-dunder-call

import ast
import operator
import time
import types
import typing as t

from dataclasses import dataclass, field

from monic.expressions.context import ExpressionsContext
from monic.expressions.exceptions import SecurityError
from monic.expressions.registry import registry
from monic.expressions.security import SecurityChecker
from monic.expressions.semantic import SemanticAnalyzer


class ReturnValue(Exception):
    """Raised to return a value from a function."""

    def __init__(self, value):
        self.value = value

class YieldValue(Exception):
    """Raised to yield a value from a generator."""

    def __init__(self, value):
        self.value = value


class YieldFromValue(Exception):
    """Raised to yield an iterator from a generator."""

    def __init__(self, iterator):
        self.iterator = iterator


@dataclass
class Scope:
    # Names declared as global
    globals: t.Set[str] = field(default_factory=set)
    # Names declared as nonlocal
    nonlocals: t.Set[str] = field(default_factory=set)
    # Names assigned in current scope
    locals: t.Set[str] = field(default_factory=set)


class ScopeContext:
    """Context manager for managing scope stack.

    This context manager ensures that scopes are properly pushed and popped
    from the stack, even if an exception occurs.
    """

    def __init__(
        self,
        interpreter: "ExpressionsInterpreter",
        save_env: bool = False,
    ) -> None:
        self.scope = Scope()
        self.interpreter = interpreter
        self.save_env = save_env

        self.saved_env: dict[str, t.Any] = {}
        self.new_env: dict[str, t.Any] = {}

    def __enter__(self) -> tuple[Scope, dict[str, t.Any] | None]:
        self.interpreter.scope_stack.append(self.scope)
        if self.save_env:
            # Save the current environment
            self.saved_env = self.interpreter.local_env.copy()
            # Create a new environment that inherits from the saved one
            self.new_env = {}
            # Copy over values from outer scope that might be needed
            if len(self.interpreter.scope_stack) > 1:
                outer_scope = self.interpreter.scope_stack[-2]
                for name in outer_scope.locals:
                    if name in self.saved_env:
                        self.new_env[name] = self.saved_env[name]
            self.interpreter.local_env = self.new_env
            return self.scope, self.saved_env
        return self.scope, None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.save_env:
            # Update the saved environment with any changes
            # that should persist outside the with block
            updated_env = {}
            for name, value in self.interpreter.local_env.items():
                if name in self.saved_env:
                    # Keep variables that existed in outer scope
                    updated_env[name] = value
                elif (
                    len(self.interpreter.scope_stack) > 1
                    and name in self.interpreter.scope_stack[-2].locals
                ):
                    # Keep nonlocal variables
                    updated_env[name] = value
            # Restore the saved environment with updates
            self.saved_env.update(updated_env)
            self.interpreter.local_env = self.saved_env
        self.interpreter.scope_stack.pop()


@dataclass
class ControlFlow:
    """Record for tracking control flow state."""

    function_depth: int = 0
    loop_depth: int = 0
    break_flag: bool = False
    continue_flag: bool = False


# Type variable for comprehension result types
T = t.TypeVar("T", list, set)


class ExpressionsInterpreter(ast.NodeVisitor):
    """Interpreter for Monic expressions."""

    def __init__(self, context: ExpressionsContext | None = None) -> None:
        """Initialize the interpreter.

        Args:
            context: Optional context for execution
        """
        self.started_at = time.monotonic()

        self.context = context or ExpressionsContext()
        self.scope_stack: list[Scope] = [Scope()]  # Track scopes
        self.control: ControlFlow = ControlFlow()

        # Initialize with built-in environment
        self.global_env: dict[str, t.Any] = {
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
            "getattr": getattr,
            "hasattr": hasattr,
            "iter": iter,
            "next": next,
            "type": type,
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
            "SyntaxError": SyntaxError,
            "IndentationError": IndentationError,
            "AttributeError": AttributeError,
            "SecurityError": SecurityError,
        }

        # Add bound objects in registry to global environment
        self.global_env.update(registry.get_all())

        # Add built-in decorators
        self.global_env.update(
            {
                "classmethod": classmethod,
                "staticmethod": staticmethod,
                "property": property,
            }
        )

        self.local_env: dict[str, t.Any] = {}

        # Initialize last result storage
        self.global_env["_"] = None

    @property
    def current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def execute(self, tree: ast.AST) -> t.Any:
        """Execute an AST."""
        # Perform security check
        checker = SecurityChecker()
        checker.check(tree)

        # Perform semantic analysis
        analyzer = SemanticAnalyzer()
        analyzer.analyze(tree)

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
        # Special case for '_'
        if name == "_":
            self.global_env["_"] = value
            return

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

    def _handle_name_target(self, target: ast.Name, value: t.Any) -> None:
        """Handle simple name assignment with scope handling.

        Args:
            target: Name AST node
            value: Value to assign
        """
        self._set_name_value(target.id, value)

    def _handle_attribute_target(
        self, target: ast.Attribute, value: t.Any
    ) -> None:
        """Handle attribute assignment (e.g., self.x = value).

        Args:
            target: Attribute AST node
            value: Value to assign
        """
        obj = self.visit(target.value)
        setattr(obj, target.attr, value)

    def _handle_subscript_target(
        self, target: ast.Subscript, value: t.Any
    ) -> None:
        """Handle subscript assignment (e.g., lst[0] = value).

        Args:
            target: Subscript AST node
            value: Value to assign
        """
        container = self.visit(target.value)
        index = self.visit(target.slice)
        container[index] = value

    def _handle_unpacking_target(self, target: ast.AST, value: t.Any) -> None:
        """Handle different types of unpacking targets.

        Args:
            target: AST node representing the unpacking target
            value: The value being assigned

        Raises:
            TypeError: If an unsupported unpacking pattern is encountered
        """
        if isinstance(target, ast.Name):
            self._handle_name_target(target, value)
        elif isinstance(target, ast.Attribute):
            self._handle_attribute_target(target, value)
        elif isinstance(target, ast.Subscript):
            self._handle_subscript_target(target, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._handle_sequence_unpacking(target, value)
        else:
            raise TypeError(f"cannot unpack {type(target).__name__}")

    def _handle_sequence_unpacking(
        self,
        target: ast.Tuple | ast.List,
        value: t.Any,
    ) -> None:
        """Handle sequence (tuple/list) unpacking.

        Args:
            target: Tuple or List AST node
            value: Value to unpack

        Raises:
            TypeError: If value cannot be unpacked
            ValueError: If there are too many or too few values to unpack
            SyntaxError: If multiple starred expressions are used
        """
        with ScopeContext(self):
            try:
                if not hasattr(value, "__iter__"):
                    raise TypeError(
                        f"cannot unpack non-iterable {type(value).__name__} "
                        "object"
                    )

                # Check for starred expressions (extended unpacking)
                starred_indices = [
                    i
                    for i, elt in enumerate(target.elts)
                    if isinstance(elt, ast.Starred)
                ]

                if len(starred_indices) > 1:
                    raise SyntaxError(
                        "multiple starred expressions in assignment"
                    )

                if starred_indices:
                    # Handle starred unpacking
                    star_index = starred_indices[0]
                    starred_target = t.cast(
                        ast.Starred, target.elts[star_index]
                    )
                    self._handle_starred_unpacking(
                        target.elts, value, star_index, starred_target
                    )
                else:
                    # Standard unpacking without starred expression
                    value_list = list(value)
                    if len(value_list) < len(target.elts):
                        raise ValueError(
                            "not enough values to unpack (expected "
                            f"{len(target.elts)}, got {len(value_list)})"
                        )
                    elif len(value_list) > len(target.elts):
                        raise ValueError(
                            "too many values to unpack (expected "
                            f"{len(target.elts)})"
                        )

                    # Unpack each element
                    for tgt, val in zip(target.elts, value):
                        self._handle_unpacking_target(tgt, val)
            except (TypeError, ValueError, SyntaxError) as e:
                raise type(e)(str(e)) from e

    def _handle_starred_unpacking(
        self,
        target_elts: list[ast.expr],
        value: t.Any,
        star_index: int,
        starred_target: ast.Starred,
    ) -> None:
        """Handle starred unpacking in sequence assignments.

        Args:
            target_elts: List of target elements
            value: Value being unpacked
            star_index: Index of the starred expression
            starred_target: The starred target node

        Raises:
            ValueError: If there are not enough values to unpack
            TypeError: If target is not a valid unpacking target
        """
        with ScopeContext(self):
            iter_value = iter(value)

            # Handle elements before the starred expression
            before_elements = target_elts[:star_index]
            for tgt in before_elements:
                try:
                    self._handle_unpacking_target(tgt, next(iter_value))
                except StopIteration as e:
                    raise ValueError("not enough values to unpack") from e

            # Collect remaining elements for the starred target
            starred_values = list(iter_value)

            # Calculate how many elements should be in the starred part
            after_star_count = len(target_elts) - star_index - 1

            # If there are more elements after the starred part
            if after_star_count > 0:
                # Make sure there are enough elements
                if len(starred_values) < after_star_count:
                    raise ValueError("not enough values to unpack")

                # Separate starred values
                starred_list = starred_values[:-after_star_count]
                after_star_values = starred_values[-after_star_count:]

                # Assign starred target
                if isinstance(starred_target.value, ast.Name):
                    self._set_name_value(starred_target.value.id, starred_list)
                else:
                    raise TypeError("starred assignment target must be a name")

                # Assign elements after starred
                after_elements = target_elts[star_index + 1 :]
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
                    raise TypeError("starred assignment target must be a name")

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
        """Handle try-except-else-finally statements.

        Args:
            node: Try AST node
        """
        with ScopeContext(self):
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except Exception as e:
                handled = False
                for handler in node.handlers:
                    if handler.type is None:
                        exc_class: t.Type[BaseException] = Exception
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

    def _get_exception_class(self, node: ast.expr) -> t.Type[BaseException]:
        """Resolve the AST node to an actual exception class object."""
        # Use self.visit to evaluate the expression node (which might be a Name
        # like 'ValueError' or user-defined classes like 'ValidationError', or
        # even dotted attributes). This way, if the user code declared:
        #
        #    class ValidationError(InputError): pass
        #
        # then we have that class in self.local_env, and self.visit(node)
        # will retrieve it.
        value = self.visit(node)

        # Make sure it's a subclass of BaseException (i.e. an Exception).
        if isinstance(value, type) and issubclass(value, BaseException):
            return value

        # Otherwise, it's not a valid exception class.
        # If "value" is, for instance, a string or a function, we raise.
        # (If "value" doesn't have a __name__, make a fallback name.)
        value_name = getattr(value, "__name__", repr(value))
        raise NameError(
            f"Name '{value_name}' is not defined or is not an exception class"
        )

    def visit_Raise(self, node: ast.Raise) -> None:
        """
        Handle raise statements that preserve exception chaining.

        - If node.exc is a call (e.g. TypeError("msg")), we create an instance.
        - If node.exc is already an exception instance, we just raise it as-is.
        - If node.cause is specified, we set that as the __cause__.
        """
        if node.exc is None:
            raise RuntimeError("No active exception to re-raise")

        # Evaluate the main 'exc'
        # Could be a class like ValueError, or an instance like ValueError()
        exc_obj = self.visit(node.exc)

        # Evaluate the 'from' cause if present
        cause_obj = self.visit(node.cause) if node.cause else None

        # If exc_obj is already an exception instance
        if isinstance(exc_obj, BaseException):
            if cause_obj is not None:
                if not isinstance(cause_obj, BaseException):
                    raise TypeError(
                        "Expected an exception instance for 'from' cause, "
                        f"got {type(cause_obj).__name__}"
                    )
                # Raise existing exception instance with cause
                raise exc_obj from cause_obj
            else:
                raise exc_obj

        # If exc_obj is a class (like ValueError) rather than an instance
        if isinstance(exc_obj, type) and issubclass(exc_obj, BaseException):
            new_exc = exc_obj()  # Instantiate
            if cause_obj is not None:
                if not isinstance(cause_obj, BaseException):
                    raise TypeError(
                        "Expected an exception instance for 'from' cause, "
                        f"got {type(cause_obj).__name__}"
                    )
                raise new_exc from cause_obj
            else:
                raise new_exc

        # Otherwise it's not a valid exception type or instance
        raise TypeError(
            "Expected an exception instance or class, got "
            f"{type(exc_obj).__name__}"
        )

    def visit_With(self, node: ast.With) -> None:
        """
        Execute a with statement, properly handling scopes and context managers.
        """
        # Create a new scope for the with block
        scope = Scope()

        # Save the current environment
        outer_env = self.local_env

        # Create a new environment for the with block
        self.local_env = {}
        # Copy only non-local variables from outer environment
        for name, value in outer_env.items():
            if name not in scope.locals:
                self.local_env[name] = value

        self.scope_stack.append(scope)

        try:
            # List to track context managers and their values
            context_managers = []

            try:
                # Enter all context managers in order
                for item in node.items:
                    try:
                        # Evaluate the context manager expression using outer
                        # environment
                        prev_env = self.local_env
                        self.local_env = outer_env
                        try:
                            context_manager = self.visit(item.context_expr)
                        finally:
                            self.local_env = prev_env

                        try:
                            # Enter the context manager
                            value = context_manager.__enter__()
                            context_managers.append((context_manager, value))

                            # Handle the optional 'as' variable if present
                            if item.optional_vars is not None:
                                # Add the 'as' variable to the with block's
                                # environment
                                name = self.visit(item.optional_vars)
                                self.local_env[name] = value
                                scope.locals.add(name)
                        except Exception as enter_exc:
                            # If __enter__ fails, properly clean up previous
                            # context managers
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

                try:
                    # Execute the body of the with statement
                    for stmt in node.body:
                        self.visit(stmt)
                        # Track any new variables defined in the body
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    scope.locals.add(target.id)
                except Exception as body_exc:
                    # Handle any exception from the body
                    if not self._exit_context_managers(
                        context_managers, body_exc
                    ):
                        raise body_exc
                else:
                    # No exception occurred, exit context managers normally
                    self._exit_context_managers(context_managers, None)
            finally:
                # Update outer environment with modified variables
                for name, value in self.local_env.items():
                    if name in outer_env:  # Only update existing variables
                        outer_env[name] = value

                # Restore the outer environment
                self.local_env = outer_env
        finally:
            # Pop the scope
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
        self.control.break_flag = True

    def visit_Continue(
        self, node: ast.Continue  # pylint: disable=unused-argument
    ) -> None:
        """Handle continue statement."""
        if self.control.loop_depth == 0:
            raise SyntaxError("'continue' outside loop")
        self.control.continue_flag = True

    def visit_While(self, node: ast.While) -> None:
        self.control.loop_depth += 1

        try:
            while True:
                test_result = self.visit(node.test)  # Evaluate test first
                if not test_result:
                    break

                try:
                    for stmt in node.body:
                        self.visit(stmt)
                        if self.control.break_flag:
                            break
                        if self.control.continue_flag:
                            self.control.continue_flag = False
                            break
                    if self.control.break_flag:
                        break
                    else:
                        # This else block is executed if no break occurred
                        continue
                except ReturnValue as rv:
                    raise rv
                except Exception as e:
                    if node.orelse:
                        for stmt in node.orelse:
                            self.visit(stmt)
                    raise e
        finally:
            self.control.break_flag = False
            self.control.continue_flag = False
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
                        self.visit(stmt)
                        if self.control.break_flag:
                            break
                        if self.control.continue_flag:
                            self.control.continue_flag = False
                            break
                    if self.control.break_flag:
                        break
                    else:
                        # This else block is executed if no break occurred
                        continue
                except ReturnValue as rv:
                    raise rv
            if not self.control.break_flag and node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
        except Exception as e:
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
            raise e
        finally:
            self.control.break_flag = False
            self.control.continue_flag = False
            self.control.loop_depth -= 1

    def _validate_nonlocal_declarations(
        self, body: t.Sequence[ast.stmt], scope_stack: list[Scope]
    ) -> None:
        """Validate nonlocal declarations in function body.

        Args:
            body: Function body AST nodes
            scope_stack: Current scope stack

        Raises:
            SyntaxError: If nonlocal declaration is invalid
        """
        for stmt in body:
            if isinstance(stmt, ast.Nonlocal):
                if len(scope_stack) < 2:
                    raise SyntaxError(
                        "nonlocal declaration not allowed at module level"
                    )

                # For non-nested functions, check bindings at definition time
                if len(scope_stack) == 2:  # Only one outer scope
                    for name in stmt.names:
                        found = False
                        # Check all outer scopes (excluding the current scope)
                        for scope in reversed(scope_stack[:-1]):
                            if (
                                name in scope.locals
                                or name in scope.nonlocals
                                or name in self.local_env
                            ):
                                found = True
                                break
                        if not found:
                            raise SyntaxError(
                                f"No binding for nonlocal '{name}' found "
                                "in outer scopes"
                            )

    def _process_function_parameters(
        self,
        func_name: str,
        call_args: tuple[t.Any, ...],
        call_kwargs: dict[str, t.Any],
        positional_params: list[ast.arg],
        defaults: list[t.Any],
        required_count: int,
        kwonly_params: list[ast.arg],
        kw_defaults: list[t.Any | None],
        vararg: ast.arg | None,
        kwarg: ast.arg | None,
    ) -> None:
        """Process and bind function parameters to arguments.

        Args:
            func_name: Name of the function being called
            call_args: Positional arguments tuple
            call_kwargs: Keyword arguments dictionary
            positional_params: List of positional parameter AST nodes
            defaults: List of default values for positional parameters
            required_count: Number of required positional parameters
            kwonly_params: List of keyword-only parameter AST nodes
            kw_defaults: List of default values for keyword-only parameters
            vararg: *args parameter AST node if present
            kwarg: **kwargs parameter AST node if present

        Raises:
            TypeError: If argument binding fails
        """
        # 1) Bind positional
        bound_args_count = min(len(call_args), len(positional_params))
        for i in range(bound_args_count):
            param = positional_params[i]
            self._set_name_value(param.arg, call_args[i])

        # leftover positional -> defaults or error
        for i in range(bound_args_count, len(positional_params)):
            param = positional_params[i]
            param_name = param.arg

            if i < required_count:
                # This param must be provided either by leftover call_args
                # (already exhausted) or by a keyword
                if param_name in call_kwargs:
                    self._set_name_value(
                        param_name, call_kwargs.pop(param_name)
                    )
                else:
                    raise TypeError(
                        f"{func_name}() missing required positional argument: "
                        f"'{param_name}'"
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
                    self._set_name_value(param_name, defaults[default_index])

        # 2) Handle keyword-only params
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
                        f"{func_name}() missing required keyword-only "
                        f"argument: '{pname}'"
                    )

        # 3) Handle *args
        if vararg:
            vararg_name = vararg.arg
            leftover = call_args[len(positional_params) :]
            self._set_name_value(vararg_name, leftover)
        else:
            # If no vararg, but user gave extra positional => error
            if len(call_args) > len(positional_params):
                raise TypeError(
                    f"{func_name}() takes {len(positional_params)} positional "
                    f"arguments but {len(call_args)} were given"
                )

        # 4) Handle **kwargs
        if kwarg:
            kwarg_name = kwarg.arg
            self._set_name_value(kwarg_name, call_kwargs)
        else:
            if call_kwargs:
                first_unexpected = next(iter(call_kwargs))
                raise TypeError(
                    f"{func_name}() got an unexpected keyword argument: "
                    f"'{first_unexpected}'"
                )

    def _create_function_closure(
        self,
        func_def: ast.FunctionDef,
        outer_env: dict[str, t.Any],
        closure_env: dict[str, t.Any],
        defaults: list[t.Any],
        kw_defaults: list[t.Any | None],
        required_count: int,
    ) -> t.Callable[..., t.Any]:
        """Create a closure for the function definition."""

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
                self.local_env[func_def.name] = func

                # Process nonlocal declarations at function execution time
                for stmt in func_def.body:
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
                                    f"No binding for nonlocal '{name}' found "
                                    "in outer scopes"
                                )
                            # Mark this name as nonlocal in the current scope
                            self.current_scope.nonlocals.add(name)

                # Process function parameters
                self._process_function_parameters(
                    func_def.name,
                    call_args,
                    call_kwargs,
                    func_def.args.args,
                    defaults,
                    required_count,
                    func_def.args.kwonlyargs,
                    kw_defaults,
                    func_def.args.vararg,
                    func_def.args.kwarg,
                )

                # Execute function body
                try:
                    for stmt in func_def.body:
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

        return func

    def _create_generator_closure(
        self,
        func_def: ast.FunctionDef,
        outer_env: dict[str, t.Any],
        closure_env: dict[str, t.Any],
        defaults: list[t.Any],
        kw_defaults: list[t.Any | None],
        required_count: int,
    ) -> t.Callable[..., t.Any]:
        """Create a closure for the generator function definition."""

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
                self.local_env[func_def.name] = func

                # Create a generator iterator
                self.local_env["__iter__"] = iter

                # Process nonlocal declarations at function execution time
                for stmt in func_def.body:
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
                                    f"No binding for nonlocal '{name}' found "
                                    "in outer scopes"
                                )
                            # Mark this name as nonlocal in the current scope
                            self.current_scope.nonlocals.add(name)

                # Process function parameters
                self._process_function_parameters(
                    func_def.name,
                    call_args,
                    call_kwargs,
                    func_def.args.args,
                    defaults,
                    required_count,
                    func_def.args.kwonlyargs,
                    kw_defaults,
                    func_def.args.vararg,
                    func_def.args.kwarg,
                )

                # Execute function body
                try:
                    for stmt in func_def.body:
                        try:
                            self.visit(stmt)
                        except YieldValue as yv:
                            yield yv.value
                        except YieldFromValue as yfv:
                            yield from yfv.iterator
                        except StopIteration:
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

        return func

    def _is_generator(self, node: ast.FunctionDef) -> bool:
        return any(
            isinstance(stmt, (ast.Yield, ast.YieldFrom))
            for stmt in ast.walk(node)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Handle function definition with support for named parameters, defaults,
        keyword-only, *args, and **kwargs.
        """
        def_scope = Scope()
        self.scope_stack.append(def_scope)

        try:
            # Validate nonlocal declarations at function definition time
            self._validate_nonlocal_declarations(node.body, self.scope_stack)

            closure_env: t.Dict[str, t.Any] = {}
            outer_env: t.Dict[str, t.Any] = self.local_env

            # Precompute default values for positional and kw-only
            defaults = [self.visit(d) for d in node.args.defaults]
            kw_defaults = [
                None if d is None else self.visit(d)
                for d in node.args.kw_defaults
            ]

            # e.g. if we have 3 positional params and 1 default then
            # required_count=2
            required_count = len(node.args.args) - len(defaults)

            # Create the function closure
            if self._is_generator(node):
                func = self._create_generator_closure(
                    func_def=node,
                    outer_env=outer_env,
                    closure_env=closure_env,
                    defaults=defaults,
                    kw_defaults=kw_defaults,
                    required_count=required_count,
                )
            else:
                func = self._create_function_closure(
                    func_def=node,
                    outer_env=outer_env,
                    closure_env=closure_env,
                    defaults=defaults,
                    kw_defaults=kw_defaults,
                required_count=required_count,
            )

            # Apply decorators in reverse order
            for decorator in reversed(node.decorator_list):
                decorator_func = self.visit(decorator)
                if decorator_func is not None:
                    func = decorator_func(func)

            # Register the function in the current scope
            self._set_name_value(node.name, func)
        finally:
            self.scope_stack.pop()

    def _create_lambda_closure(
        self,
        node: ast.Lambda,
        outer_env: dict[str, t.Any],
        closure_env: dict[str, t.Any],
        defaults: list[t.Any],
        kw_defaults: list[t.Any | None],
        required_count: int,
    ) -> t.Callable[..., t.Any]:
        """Create a closure for the lambda function.

        Args:
            node: Lambda AST node
            outer_env: Outer environment dictionary
            closure_env: Closure environment dictionary
            defaults: List of default values for positional parameters
            kw_defaults: List of default values for keyword-only parameters
            required_count: Number of required positional parameters

        Returns:
            The created lambda closure
        """

        def lambda_func(*call_args, **call_kwargs):
            lambda_scope = Scope()
            self.scope_stack.append(lambda_scope)

            prev_env = self.local_env
            self.local_env = {**outer_env, **closure_env}

            try:
                # Process function parameters
                self._process_function_parameters(
                    "<lambda>",
                    call_args,
                    call_kwargs,
                    node.args.args,
                    defaults,
                    required_count,
                    node.args.kwonlyargs,
                    kw_defaults,
                    node.args.vararg,
                    node.args.kwarg,
                )

                # Evaluate the body
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

    def visit_Lambda(self, node: ast.Lambda) -> t.Callable[..., t.Any]:
        """Handle lambda function definition.

        Args:
            node: Lambda AST node

        Returns:
            The created lambda function
        """
        closure_env: dict[str, t.Any] = {}
        outer_env: dict[str, t.Any] = self.local_env

        # Precompute default values for positional and kw-only
        defaults = [self.visit(d) for d in node.args.defaults]
        kw_defaults = [
            None if d is None else self.visit(d) for d in node.args.kw_defaults
        ]
        required_count = len(node.args.args) - len(defaults)

        # Create the lambda closure
        return self._create_lambda_closure(
            node=node,
            outer_env=outer_env,
            closure_env=closure_env,
            defaults=defaults,
            kw_defaults=kw_defaults,
            required_count=required_count,
        )

    def visit_Return(self, node: ast.Return) -> None:
        if self.control.function_depth == 0:
            raise SyntaxError("'return' outside function")

        value = None if node.value is None else self.visit(node.value)
        raise ReturnValue(value)

    def visit_Yield(self, node: ast.Yield) -> None:
        value = None if node.value is None else self.visit(node.value)
        raise YieldValue(value)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        iterator = self.visit(node.value)
        raise YieldFromValue(iterator)

    def _evaluate_call_arguments(
        self, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> tuple[list[t.Any], dict[str, t.Any]]:
        """Evaluate function call arguments.

        Args:
            args: List of positional argument AST nodes
            keywords: List of keyword argument AST nodes

        Returns:
            Tuple of (positional args list, keyword args dict)

        Raises:
            TypeError: If keyword argument is invalid
        """
        # Evaluate positional arguments
        pos_args = [self.visit(arg) for arg in args]

        # Evaluate keyword arguments
        kwargs = {}
        for kw in keywords:
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

        return pos_args, kwargs

    def _call_function(
        self, func: t.Any, pos_args: list[t.Any], kwargs: dict[str, t.Any]
    ) -> t.Any:
        """Call a function with the given arguments.

        Args:
            func: Function object to call
            pos_args: List of positional arguments
            kwargs: Dictionary of keyword arguments

        Returns:
            Result of the function call

        Raises:
            TypeError: If the function is not callable
        """
        # Check if the function is callable
        if not callable(func):
            raise TypeError(f"'{type(func).__name__}' object is not callable")

        # Handle bound functions
        if registry.is_bound(func):
            return func(*pos_args, **kwargs)

        # Handle bound methods
        if isinstance(func, types.MethodType):
            return func(*pos_args, **kwargs)

        # Handle normal functions
        return func(*pos_args, **kwargs)

    def visit_Call(self, node: ast.Call) -> t.Any:
        """
        Handle function calls, including positional args, keyword args, and
        **kwargs.

        Args:
            node: Call AST node

        Returns:
            Result of the function call

        Raises:
            TypeError: If the function is not callable or arguments are invalid
        """
        # Evaluate the function object
        func = self.visit(node.func)

        # Evaluate arguments
        pos_args, kwargs = self._evaluate_call_arguments(
            node.args, node.keywords
        )

        # Call the function
        return self._call_function(func, pos_args, kwargs)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> t.Any:
        """
        Handle generator expressions with closure semantics.

        Capture the current local_env when the generator expression is created
        so that references to local variables (like 'y=2') remain valid even
        after returning from the enclosing function.

        Example user code that requires closure:
            def gen():
                y = 2
                return (x + y + i for i in range(3))

        Without closure capturing, "y" would appear undefined once 'gen'
        returns.
        """
        # Copy (snapshot) the current local environment for closure
        closure_env = dict(self.local_env)

        def _generator_expression_runner():
            """
            A helper that reinstalls the 'closure_env' as 'self.local_env'
            whenever we iterate the generator, mimicking how Python closures
            keep references to their defining scopes.
            """
            prev_env = self.local_env
            self.local_env = closure_env
            try:
                # Actually produce the items by calling the comprehension logic
                yield from self._evaluate_generator_exp(node)
            finally:
                self.local_env = prev_env

        return _generator_expression_runner()

    def _evaluate_generator_exp(
        self, node: ast.GeneratorExp
    ) -> t.Iterator[t.Any]:
        """
        Evaluate an already-constructed generator expression node under the
        current (already snapshotted) self.local_env.
        """

        def inner():
            # The logic below closely matches typical generator expression
            # evaluation: we handle node.generators (the "for" clauses, possibly
            # with if-filters) and node.elt (the yield expression).
            generators = node.generators

            # We can reuse your existing "process_generator" or whatever method
            # you have. The snippet below shows a direct approach:
            def process(gens, index=0):
                """
                Recursively handle each comprehension generator in 'gens'.
                Once we exhaust them, evaluate node.elt and yield it.
                """
                if index >= len(gens):
                    # If no more loops, the expression is node.elt
                    yield self.visit(node.elt)
                    return

                gen = gens[index]
                iter_obj = self.visit(gen.iter)

                try:
                    iterator = iter(iter_obj)
                except TypeError as e:
                    # If not iterable
                    raise TypeError(
                        f"object is not iterable: {iter_obj}"
                    ) from e

                for item in iterator:
                    # Assign the loop target
                    self._assign_comprehension_target(gen.target, item)
                    # Check any if-conditions
                    if gen.ifs:
                        skip = False
                        for if_test in gen.ifs:
                            condition = self.visit(if_test)
                            if not condition:
                                skip = True
                                break
                        if skip:
                            continue

                    # Recurse to handle the next generator
                    yield from process(gens, index + 1)

            yield from process(generators)

        yield from inner()

    def _assign_comprehension_target(
        self, target: ast.expr, value: t.Any
    ) -> None:
        """
        Assign 'value' to the target node within a comprehension. This can be
        something like 'for i in range(5)', or 'for k, v in items'.
        If it's just a Name node, we store in self.local_env. If it's a tuple
        or list, we destructure.
        """
        if isinstance(target, ast.Name):
            # This is a simple Name like: for i in ...
            self._set_name_value(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Destructure the value into the sub-targets
            if not isinstance(value, (tuple, list)):
                raise TypeError(f"cannot unpack non-iterable value {value}")
            if len(value) != len(target.elts):
                raise ValueError(
                    f"cannot unpack {len(value)} values into "
                    f"{len(target.elts)} targets"
                )
            for t_elt, v_elt in zip(target.elts, value):
                self._assign_comprehension_target(t_elt, v_elt)
        else:
            # For something more complex, handle similarly or raise an error
            raise NotImplementedError(
                f"unsupported comprehension target type: {type(target)}"
            )

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

    def _setup_comprehension_scope(self) -> tuple[Scope, dict[str, t.Any]]:
        """Set up a new scope for comprehension execution.

        Returns:
            Tuple of (new scope, outer environment)
        """
        # Create new scope for the comprehension
        comp_scope = Scope()
        self.scope_stack.append(comp_scope)

        # Copy the outer environment
        outer_env = self.local_env
        self.local_env = outer_env.copy()

        return comp_scope, outer_env

    def _process_generator_item(
        self,
        generator: ast.comprehension,
        item: t.Any,
        current_env: dict[str, t.Any],
        outer_env: dict[str, t.Any],
    ) -> bool:
        """Process a single item in a generator.

        Args:
            generator: Generator AST node
            item: Current item being processed
            current_env: Current environment dictionary
            outer_env: Outer environment dictionary

        Returns:
            bool: Whether all conditions are met

        Raises:
            TypeError: If value is not iterable
            ValueError: If target unpacking fails
            SyntaxError: If multiple starred expressions are used
        """
        # Restore environment from before this generator's loop
        self.local_env = current_env.copy()

        try:
            self._handle_unpacking_target(generator.target, item)
        except (TypeError, ValueError, SyntaxError) as e:
            if isinstance(generator.target, ast.Name):
                self._set_name_value(generator.target.id, item)
            else:
                raise e

        # Check if all conditions are met
        conditions_met = all(
            self.visit(if_clause) for if_clause in generator.ifs
        )

        # Update outer environment with any named expression bindings
        for name, value in self.local_env.items():
            if name not in current_env:
                outer_env[name] = value

        return conditions_met

    def _handle_comprehension(
        self, node: ast.ListComp | ast.SetComp, result_type: type[T]
    ) -> T:
        """Handle list and set comprehensions.

        Args:
            node: ListComp or SetComp AST node
            result_type: Type of the result (list or set)

        Returns:
            The evaluated comprehension result
        """
        _, outer_env = self._setup_comprehension_scope()

        try:
            result: list[t.Any] = []

            def process_generator(generators: list, index: int = 0) -> None:
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
                    if self._process_generator_item(
                        generator, item, current_env, outer_env
                    ):
                        # Process next generator or append result
                        process_generator(generators, index + 1)

            # Start processing generators recursively
            process_generator(node.generators)
            return result_type(result)
        finally:
            # Restore the outer environment and pop the scope
            self.local_env = outer_env
            self.scope_stack.pop()

    def visit_ListComp(self, node: ast.ListComp) -> list:
        return self._handle_comprehension(node, list)

    def visit_SetComp(self, node: ast.SetComp) -> set:
        return self._handle_comprehension(node, set)

    def visit_DictComp(self, node: ast.DictComp) -> dict[t.Any, t.Any]:
        """Handle dictionary comprehensions."""
        _, outer_env = self._setup_comprehension_scope()

        try:
            result: dict[t.Any, t.Any] = {}

            def process_generator(generators: list, index: int = 0) -> None:
                if index >= len(generators):
                    # Base case: all generators processed, evaluate key-value
                    # pair
                    key = self.visit(node.key)
                    value = self.visit(node.value)
                    result[key] = value
                    return

                generator = generators[index]
                iter_obj = self.visit(generator.iter)

                # Save the current environment before processing this generator
                current_env = self.local_env.copy()

                for item in iter_obj:
                    if self._process_generator_item(
                        generator, item, current_env, outer_env
                    ):
                        # Process next generator or evaluate key-value pair
                        process_generator(generators, index + 1)

            # Start processing generators recursively
            process_generator(node.generators)
            return result
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
        """Format a value according to format spec.

        Args:
            node: The FormattedValue AST node

        Returns:
            The formatted string

        Raises:
            ValueError: If format specification is invalid
        """
        value = self.visit(node.value)
        format_spec = self.visit(node.format_spec) if node.format_spec else ""

        try:
            if node.conversion == -1:  # No conversion
                converted = value
            elif node.conversion == 115:  # str
                converted = str(value)
            elif node.conversion == 114:  # repr
                converted = repr(value)
            elif node.conversion == 97:  # ascii
                converted = ascii(value)
            else:
                raise ValueError(f"Unknown conversion code {node.conversion}")

            if not format_spec:
                result = str(converted)
            else:
                result = format(converted, format_spec)
            return result
        except ValueError as e:
            raise ValueError(
                f"Invalid format specification '{format_spec}' "
                f"for value {repr(value)} of type {type(value).__name__}"
            ) from e

    def _get_attribute_safely(self, value: t.Any, attr_name: str) -> t.Any:
        """Get attribute value with proper error handling.

        Args:
            value: Object to get attribute from
            attr_name: Name of the attribute to get

        Returns:
            The attribute value

        Raises:
            AttributeError: If attribute doesn't exist
        """
        try:
            return getattr(value, attr_name)
        except AttributeError as e:
            if isinstance(value, type):
                raise AttributeError(
                    f"type object '{value.__name__}' has no attribute "
                    f"'{attr_name}'"
                ) from e
            else:
                raise AttributeError(
                    f"'{type(value).__name__}' object has no attribute "
                    f"'{attr_name}'"
                ) from e

    def visit_Attribute(self, node: ast.Attribute) -> t.Any:
        """Visit an attribute access node.

        Args:
            node: The Attribute AST node

        Returns:
            The value of the attribute

        Raises:
            AttributeError: If the attribute doesn't exist
            SecurityError: If attribute access is not allowed
        """
        # Check if this is a forbidden attribute
        if node.attr in SecurityChecker.FORBIDDEN_ATTRS:
            raise SecurityError(
                f"Access to '{node.attr}' attribute is not allowed"
            )

        # Get the base value
        value = self.visit(node.value)

        # Get the attribute safely
        attr = self._get_attribute_safely(value, node.attr)

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
        elif (
            isinstance(node.slice, ast.Constant)
            and node.slice.value is Ellipsis
        ):
            # Handle ellipsis subscript (lst[...]) by returning the entire list
            if isinstance(value, list):
                return value[:]
            else:
                raise TypeError(
                    f"'{type(value).__name__}' object does not support "
                    "ellipsis indexing"
                )
        else:
            # For Python >= 3.9, node.slice can be other expression nodes
            slice_val = self.visit(node.slice)
            try:
                return value[slice_val]
            except TypeError as e:
                if isinstance(value, list):
                    raise TypeError(
                        "list indices must be integers or slices, not "
                        f"{type(slice_val).__name__}"
                    ) from e
                elif isinstance(value, dict):
                    raise TypeError(
                        "unhashable type: " f"{type(slice_val).__name__}"
                    ) from e
                else:
                    raise TypeError(
                        f"{type(value).__name__} indices must be integers"
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

    def _create_custom_super(
        self,
        class_obj: type | None,
    ) -> t.Callable[..., t.Any]:
        """Create a custom super implementation for a class."""

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

                        def bound_super_method(name, current_class=next_class):
                            method = getattr(current_class, name)
                            if isinstance(method, (staticmethod, classmethod)):
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

        return custom_super

    def _create_class_namespace(self, node: ast.ClassDef) -> dict[str, t.Any]:
        """Create and populate the class namespace.

        Args:
            node: ClassDef AST node

        Returns:
            The populated class namespace
        """
        namespace: dict[str, t.Any] = {}

        # Save current environment
        prev_env = self.local_env
        self.local_env = namespace

        try:
            # Execute the class body
            for stmt in node.body:
                self.visit(stmt)
        finally:
            # Restore the environment
            self.local_env = prev_env

        return namespace

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definition with support for inheritance and class body.

        Args:
            node: The ClassDef AST node
        """
        # Create a new scope for the class definition
        with ScopeContext(self, save_env=True) as (_, saved_env):
            # Evaluate base classes
            bases = tuple(self.visit(base) for base in node.bases)

            # Create and populate the class namespace
            namespace = self._create_class_namespace(node)

            # Add custom super to the class namespace (will be updated after
            # class creation)
            namespace["super"] = self._create_custom_super(class_obj=None)

            # Set the module name for the class
            namespace["__module__"] = "monic.expressions.__namespace__"

            # Create the class object
            class_obj = types.new_class(
                node.name, bases, {}, lambda ns: ns.update(namespace)
            )

            # Update super with the actual class object (after class creation)
            namespace["super"] = self._create_custom_super(class_obj=class_obj)

            # Register the class in the current scope
            self._set_name_value(node.name, class_obj)

            # Also register the class in the global environment
            self.global_env[node.name] = class_obj

            # Also register the class in the outer scope if we're in a method
            # or if we have a saved environment
            if saved_env is not None:
                saved_env[node.name] = class_obj
                # Also add to outer scope's locals
                if len(self.scope_stack) > 1:
                    self.scope_stack[-2].locals.add(node.name)

    def _match_value_pattern(
        self,
        pattern: ast.MatchValue,
        value: t.Any,
    ) -> bool:
        """Match a literal value pattern.

        Args:
            pattern: MatchValue AST node
            value: Value to match against

        Returns:
            Whether the pattern matches the value
        """
        pattern_value = self.visit(pattern.value)
        return type(value) is type(pattern_value) and value == pattern_value

    def _match_sequence_pattern(
        self,
        pattern: ast.MatchSequence,
        value: t.Any,
        pattern_vars: set[str],
    ) -> bool:
        """Match a sequence pattern.

        Args:
            pattern: MatchSequence AST node
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
        if not isinstance(value, (list, tuple)):
            return False

        # Find star pattern index if exists
        star_idx = -1
        for i, p in enumerate(pattern.patterns):
            if isinstance(p, ast.MatchStar):
                star_idx = i
                break

        if star_idx == -1:
            return self._match_fixed_sequence(
                pattern.patterns, value, pattern_vars
            )
        else:
            return self._match_star_sequence(
                pattern.patterns, value, star_idx, pattern_vars
            )

    def _match_fixed_sequence(
        self,
        patterns: list[ast.pattern],
        value: t.Any,
        pattern_vars: set[str],
    ) -> bool:
        """Match a sequence pattern without star expressions.

        Args:
            patterns: List of pattern AST nodes
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
        if len(patterns) != len(value):
            return False
        return all(
            self._match_pattern(p, v, pattern_vars)
            for p, v in zip(patterns, value)
        )

    def _match_star_sequence(
        self,
        patterns: list[ast.pattern],
        value: t.Any,
        star_idx: int,
        pattern_vars: set[str],
    ) -> bool:
        """Match a sequence pattern with a star expression.

        Args:
            patterns: List of pattern AST nodes
            value: Value to match against
            star_idx: Index of the star pattern
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
        if len(value) < len(patterns) - 1:
            return False

        # Match patterns before star
        for p, v in zip(patterns[:star_idx], value[:star_idx]):
            if not self._match_pattern(p, v, pattern_vars):
                return False

        # Calculate remaining elements after star
        remaining_count = len(patterns) - star_idx - 1

        # Match patterns after star
        for p, v in zip(
            patterns[star_idx + 1 :],
            value[-remaining_count:] if remaining_count > 0 else [],
        ):
            if not self._match_pattern(p, v, pattern_vars):
                return False

        # Bind star pattern if it has a name
        star_pattern = patterns[star_idx]
        if isinstance(star_pattern, ast.MatchStar) and star_pattern.name:
            star_value = (
                list(value[star_idx:-remaining_count])
                if remaining_count > 0
                else list(value[star_idx:])
            )
            self._set_name_value(star_pattern.name, star_value)
            self.current_scope.locals.add(star_pattern.name)

        return True

    def _match_mapping_pattern(
        self,
        pattern: ast.MatchMapping,
        value: t.Any,
        pattern_vars: set[str],
    ) -> bool:
        """Match a mapping pattern.

        Args:
            pattern: MatchMapping AST node
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
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
            if not self._match_pattern(pat, value[key_value], pattern_vars):
                return False

        # Handle rest pattern if present
        if pattern.rest is not None:
            if pattern.rest in pattern_vars:
                raise SyntaxError(
                    f"multiple assignments to name '{pattern.rest}' in pattern"
                )
            pattern_vars.add(pattern.rest)
            rest_dict = {
                k: v
                for k, v in value.items()
                if not any(self.visit(key) == k for key in pattern.keys)
            }
            self._set_name_value(pattern.rest, rest_dict)
            self.current_scope.locals.add(pattern.rest)

        return True

    def _match_as_pattern(
        self,
        pattern: ast.MatchAs,
        value: t.Any,
        pattern_vars: set[str],
    ) -> bool:
        """Match an AS pattern.

        Args:
            pattern: MatchAs AST node
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
        if pattern.pattern is not None:
            if not self._match_pattern(pattern.pattern, value, pattern_vars):
                return False
        if pattern.name is not None:
            if pattern.name in pattern_vars:
                raise SyntaxError(
                    f"multiple assignments to name '{pattern.name}' in pattern"
                )
            pattern_vars.add(pattern.name)
            self._set_name_value(pattern.name, value)
            self.current_scope.locals.add(pattern.name)
        return True

    def _match_star_pattern(
        self,
        pattern: ast.MatchStar,
        value: t.Any,
    ) -> bool:
        """Match a star pattern.

        Args:
            pattern: MatchStar AST node
            value: Value to match against

        Returns:
            Whether the pattern matches the value
        """
        if pattern.name is not None:
            self._set_name_value(pattern.name, value)
            self.current_scope.locals.add(pattern.name)
        return True

    def _match_pattern(
        self,
        pattern: ast.pattern,
        value: t.Any,
        pattern_vars: set[str] | None = None,
    ) -> bool:
        """Match a pattern against a value.

        Args:
            pattern: Pattern AST node
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value

        Raises:
            SyntaxError: If a pattern variable is bound multiple times
        """
        # Initialize pattern_vars on first call
        if pattern_vars is None:
            pattern_vars = set()

        if isinstance(pattern, ast.MatchValue):
            return self._match_value_pattern(pattern, value)
        elif isinstance(pattern, ast.MatchSingleton):
            return value is pattern.value
        elif isinstance(pattern, ast.MatchSequence):
            return self._match_sequence_pattern(pattern, value, pattern_vars)
        elif isinstance(pattern, ast.MatchMapping):
            return self._match_mapping_pattern(pattern, value, pattern_vars)
        elif isinstance(pattern, ast.MatchStar):
            return self._match_star_pattern(pattern, value)
        elif isinstance(pattern, ast.MatchAs):
            return self._match_as_pattern(pattern, value, pattern_vars)
        elif isinstance(pattern, ast.MatchOr):
            # Try each alternative pattern
            matched_vars = None
            for p in pattern.patterns:
                alt_vars: set[str] = set()
                if self._match_pattern(p, value, alt_vars):
                    if matched_vars is None:
                        matched_vars = alt_vars
                    elif matched_vars != alt_vars:
                        raise SyntaxError(
                            "alternative patterns bind different names"
                        )
                    pattern_vars.update(alt_vars)
                    return True
            return False
        elif isinstance(pattern, ast.MatchClass):
            return self._match_class_pattern(pattern, value, pattern_vars)

        return False

    def _match_class_pattern(
        self,
        pattern: ast.MatchClass,
        value: t.Any,
        pattern_vars: set[str],
    ) -> bool:
        """Match a class pattern.

        Args:
            pattern: MatchClass AST node
            value: Value to match against
            pattern_vars: Set of pattern variables already bound in this pattern

        Returns:
            Whether the pattern matches the value
        """
        cls = self.visit(pattern.cls)
        if not isinstance(value, cls):
            return False

        # Get positional attributes from __match_args__
        match_args = getattr(cls, "__match_args__", ())
        if len(pattern.patterns) > len(match_args):
            return False

        # Match positional patterns
        for pat, attr_name in zip(pattern.patterns, match_args):
            if not self._match_pattern(
                pat, getattr(value, attr_name), pattern_vars
            ):
                return False

        # Match keyword patterns
        for name, pat in zip(pattern.kwd_attrs, pattern.kwd_patterns):
            if not hasattr(value, name):
                return False
            if not self._match_pattern(pat, getattr(value, name), pattern_vars):
                return False

        return True

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
        with ScopeContext(self):
            # Try each case in order
            for case in node.cases:
                pattern = case.pattern

                # Create a temporary scope for pattern matching
                with ScopeContext(self):
                    # Try to match the pattern with a new pattern_vars set
                    pattern_vars: set[str] = set()
                    if not self._match_pattern(pattern, subject, pattern_vars):
                        # If no match, continue to the next case
                        continue

                    # If there's a guard, evaluate it
                    if case.guard is not None:
                        # Evaluate the guard expression
                        guard_result = self.visit(case.guard)
                        if not guard_result:
                            # If the guard fails, continue to the next case
                            continue

                    # Copy matched variables from temp scope to match scope
                    for name in self.current_scope.locals:
                        if name in self.local_env:
                            self._set_name_value(name, self.local_env[name])

                    # Execute the case body
                    for stmt in case.body:
                        self.visit(stmt)

                    # Return the match statement since we found a match
                    return

    def visit_Global(self, node: ast.Global) -> None:
        """Handle global declarations."""
        for name in node.names:
            # Check if the name is already declared as nonlocal
            if name in self.current_scope.nonlocals:
                raise SyntaxError(f"name '{name}' is nonlocal and global")
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
            # Check if the name is already declared as global
            if name in self.current_scope.globals:
                raise SyntaxError(f"name '{name}' is nonlocal and global")

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
        """Visit a constant value node.

        Args:
            node: The Constant AST node

        Returns:
            The constant value
        """
        return node.value
