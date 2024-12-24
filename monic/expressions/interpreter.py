#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import ast
import typing as t


class SecurityError(Exception):
    """Raised when dangerous operations are detected."""


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


class ExpressionInterpreter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.global_env: t.Dict[str, t.Any] = {
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
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "tuple": tuple,
            "set": set,
            "dict": dict,
            "None": None,
            "True": True,
            "False": False,
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

    def execute(self, tree: ast.AST) -> t.Any:
        # Perform security check
        self._check_security(tree)

        try:
            # Handle expression statements specially to capture their value
            if isinstance(tree, ast.Expression):
                result = self.visit(tree)
                self.global_env["_"] = result
                return result
            elif isinstance(tree, ast.Module):
                last_expr = None
                for stmt in tree.body:
                    if isinstance(stmt, ast.Expr):
                        # For expression statements, capture the value
                        last_expr = self.visit(stmt.value)
                        self.global_env["_"] = last_expr
                    else:
                        # For other statements, just execute them
                        self.visit(stmt)
                return last_expr
            else:
                result = self.visit(tree)
                self.global_env["_"] = result
                return result
        except Exception as e:
            raise type(e)(f"Runtime error: {str(e)}") from e

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
                raise SecurityError(f"Access to '{node.attr}' is not allowed")

            # Check for __builtins__ access
            if isinstance(node, ast.Name) and node.id == "__builtins__":
                raise SecurityError("Access to '__builtins__' is not allowed")

            # Check for import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise SecurityError("Import statements are not allowed")

    def visit_Constant(self, node: ast.Constant) -> t.Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> t.Any:
        if isinstance(node.ctx, ast.Store):
            return node.id

        if node.id == "_":
            return self.global_env.get("_")

        if node.id in self.local_env:
            return self.local_env[node.id]
        if node.id in self.global_env:
            return self.global_env[node.id]

        raise NameError(f"name '{node.id}' is not defined")

    def visit_Assign(self, node: ast.Assign) -> None:
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_env[target.id] = value
            else:
                raise NotImplementedError(
                    "Only simple assignments are supported"
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
        except Exception as e:  # pylint: disable=broad-except
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
            if class_name in globals():
                exc_class = globals()[class_name]
                if isinstance(exc_class, type) and issubclass(
                    exc_class, Exception
                ):
                    return exc_class
            raise NameError(
                f"name '{class_name}' is not defined or is not an exception class"
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

    def visit_If(self, node: ast.If) -> None:
        if self.visit(node.test):
            for stmt in node.body:
                self.visit(stmt)
        elif node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        while self.visit(node.test):
            try:
                for stmt in node.body:
                    self.visit(stmt)
            except ReturnValue as rv:
                raise rv
            except Exception as e:
                if node.orelse:
                    for stmt in node.orelse:
                        self.visit(stmt)
                raise e

    def visit_For(self, node: ast.For) -> None:
        iter_value = self.visit(node.iter)
        try:
            for item in iter_value:
                if isinstance(node.target, ast.Name):
                    self.local_env[node.target.id] = item
                else:
                    raise NotImplementedError(
                        "Only simple for loop targets are supported"
                    )

                try:
                    for stmt in node.body:
                        self.visit(stmt)
                except ReturnValue as rv:
                    raise rv
        except Exception as e:
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
            raise e

    def visit_Lambda(self, node: ast.Lambda) -> t.Callable:
        closure_locals = self.local_env.copy()
        closure_globals = self.global_env

        def lambda_func(*args):
            # Save current environments
            prev_local = self.local_env
            prev_global = self.global_env

            try:
                # Create new environments
                self.local_env = closure_locals.copy()
                self.global_env = closure_globals

                # Assign arguments to parameters
                for param, arg in zip(node.args.args, args):
                    self.local_env[param.arg] = arg

                # Execute function body
                return self.visit(node.body)
            finally:
                # Restore environments
                self.local_env = prev_local
                self.global_env = prev_global

        return lambda_func

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        def func(*args):
            # Create new local environment
            previous_env = self.local_env.copy()
            self.local_env = {}

            try:
                # Assign arguments to parameters
                for param, arg in zip(node.args.args, args):
                    self.local_env[param.arg] = arg

                # Execute function body
                try:
                    for stmt in node.body:
                        self.visit(stmt)
                except ReturnValue as rv:
                    return rv.value

                return None
            finally:
                # Restore local environment
                self.local_env = previous_env

        # Store function in global environment
        self.global_env[node.name] = func

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
        outer_env = self.local_env.copy()
        self.local_env = self.local_env.copy()

        try:
            result = {}
            for generator in node.generators:
                iter_obj = self.visit(generator.iter)

                for item in iter_obj:
                    if isinstance(generator.target, ast.Name):
                        self.local_env[generator.target.id] = item
                    else:
                        raise NotImplementedError(
                            "Only simple targets are supported in comprehensions"
                        )

                    if all(
                        self.visit(if_clause) for if_clause in generator.ifs
                    ):
                        key = self.visit(node.key)
                        value = self.visit(node.value)
                        result[key] = value

            return result
        finally:
            self.local_env = outer_env

    T = t.TypeVar("T", list, set)

    def _handle_comprehension(
        self, node: ast.ListComp | ast.SetComp, result_type: t.Type[T]
    ) -> T:
        outer_env = self.local_env.copy()
        self.local_env = self.local_env.copy()

        try:
            result = []
            for generator in node.generators:
                iter_obj = self.visit(generator.iter)

                for item in iter_obj:
                    if isinstance(generator.target, ast.Name):
                        self.local_env[generator.target.id] = item
                    else:
                        raise NotImplementedError(
                            "Only simple targets are supported in comprehensions"
                        )

                    if all(
                        self.visit(if_clause) for if_clause in generator.ifs
                    ):
                        result.append(self.visit(node.elt))

            return result_type(result)
        finally:
            self.local_env = outer_env

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
            NotImplementedError: If unsupported conversion or format_spec is used
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
