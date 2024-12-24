#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import ast
import typing as t


class InterpreterError(Exception):
    pass


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


class ExpressionInterpreter(ast.NodeVisitor):
    def __init__(self) -> None:
        # Environment for storing global and local variables
        self.global_env: t.Dict[str, t.Any] = {}
        self.local_env: t.Dict[str, t.Any] = {}
        # Exception handling stack
        self.exception_stack: t.List[t.Dict] = []

    def execute(self, tree: ast.AST) -> t.Any:
        """Parse and execute the code"""
        try:
            return self.visit(tree)
        except Exception as e:
            raise type(e)(f"Runtime error: {str(e)}")

    def visit_Constant(self, node: ast.Constant) -> t.Any:
        """Handle constant nodes"""
        return node.value

    def visit_Name(self, node: ast.Name) -> t.Any:
        """Handle variable name nodes"""
        if isinstance(node.ctx, ast.Store):
            return node.id
        # Check local environment first, then global for variable lookup
        if node.id in self.local_env:
            return self.local_env[node.id]
        if node.id in self.global_env:
            return self.global_env[node.id]
        raise NameError(f"name '{node.id}' is not defined")

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle assignment nodes"""
        value = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                # By default, assign to local environment
                self.local_env[target.id] = value
            else:
                raise NotImplementedError(
                    "Only simple assignments are supported"
                )

    def visit_BinOp(self, node: ast.BinOp) -> t.Any:
        """Handle binary operator nodes"""
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
            raise type(e)(str(e))

    def visit_Compare(self, node: ast.Compare) -> bool:
        """Handle comparison operator nodes"""
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
        """Handle try-except-else-finally blocks"""
        # Save current exception handling state
        exc_info = {"exc_type": None, "exc_value": None, "exc_traceback": None}
        self.exception_stack.append(exc_info)

        try:
            # Execute try block
            for stmt in node.body:
                try:
                    self.visit(stmt)
                except Exception as e:  # pylint: disable=broad-except
                    # Check if exception matches any handlers
                    handled = False
                    for handler in node.handlers:
                        if handler.type is None or isinstance(
                            e, self._get_exception_class(handler.type)
                        ):
                            # Exception matches this handler
                            if handler.name:
                                self.local_env[handler.name] = e
                            try:
                                for stmt in handler.body:
                                    self.visit(stmt)
                            finally:
                                if handler.name:
                                    del self.local_env[handler.name]
                            handled = True
                            break
                    if not handled:
                        raise
            else:
                # If no exception occurred, execute else block
                if node.orelse:
                    for stmt in node.orelse:
                        self.visit(stmt)
        finally:
            # Always execute finally block
            if node.finalbody:
                for stmt in node.finalbody:
                    self.visit(stmt)
            # Restore previous exception handling state
            self.exception_stack.pop()

    def _get_exception_class(self, node: ast.Name) -> t.Type[Exception]:
        """Get the exception class from a node"""
        class_name = node.id
        if class_name in globals():
            exc_class = globals()[class_name]
            if isinstance(exc_class, type) and issubclass(exc_class, Exception):
                return exc_class
        raise NameError(
            f"name '{class_name}' is not defined or is not an exception class"
        )

    def visit_Raise(self, node: ast.Raise) -> None:
        """Handle raise statements"""
        if node.exc is None:
            if (
                not self.exception_stack
                or self.exception_stack[-1]["exc_type"] is None
            ):
                raise RuntimeError("No active exception to re-raise")
            raise self.exception_stack[-1]["exc_value"]

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
        """Handle if statement nodes"""
        if self.visit(node.test):
            for stmt in node.body:
                self.visit(stmt)
        elif node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        """Handle while loop nodes"""
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
        """Handle for loop nodes"""
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle function definition nodes"""

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
        """Handle return statement nodes"""
        value = None if node.value is None else self.visit(node.value)
        raise ReturnValue(value)

    def visit_Call(self, node: ast.Call) -> t.Any:
        """Handle function call nodes"""
        try:
            func = self.visit(node.func)
            args = [self.visit(arg) for arg in node.args]
            return func(*args)
        except Exception as e:
            raise type(e)(f"Error in function call: {str(e)}")

    def visit_List(self, node: ast.List) -> list:
        """Handle list nodes"""
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        """Handle tuple nodes"""
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Dict(self, node: ast.Dict) -> dict:
        """Handle dictionary nodes"""
        return {
            self.visit(key) if key is not None else None: self.visit(value)
            for key, value in zip(node.keys, node.values)
        }

    def visit_Attribute(self, node: ast.Attribute) -> t.Any:
        """Handle attribute access nodes"""
        try:
            value = self.visit(node.value)
            return getattr(value, node.attr)
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {str(e)}") from e

    def visit_Module(self, node: ast.Module) -> None:
        """Handle module nodes"""
        for stmt in node.body:
            self.visit(stmt)
