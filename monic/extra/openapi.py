#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

"""OpenAPI schema generation for bound functions in the registry."""

import inspect
import typing as t
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin

from monic.expressions.registry import Registry


@dataclass
class OpenAPIParameter:
    """Represents an OpenAPI parameter."""

    name: str
    type: str
    required: bool = True
    description: str | None = None
    default: Any | None = None


@dataclass
class OpenAPIFunction:
    """Represents a function with OpenAPI metadata."""

    name: str
    path: str  # dot-separated path like "math.sqrt"
    description: str | None = None
    parameters: list[OpenAPIParameter] = field(default_factory=list)
    return_type: str | None = None
    deprecated: bool = False


class OpenAPIGenerator:
    """Generates OpenAPI 3.0 schemas from registry-bound functions."""

    # Python type -> OpenAPI type mapping
    TYPE_MAPPING = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        tuple: "array",
        set: "array",
        None: "null",
        type(None): "null",
    }

    # Namespace display names for known libraries and modules
    NAMESPACE_DISPLAY_NAMES = {
        # Data science libraries
        "np": "NumPy",
        "pd": "Pandas",
        "pl": "Polars",
        "pa": "PyArrow",
        "pq": "Parquet",
        "pc": "PyArrow Compute",
        # Common modules
        "json": "JSON",
        "datetime": "DateTime",
        "time": "Time",
        "inspector": "Inspector",
        # Common acronyms
        "api": "API",
        "http": "HTTP",
        "https": "HTTPS",
        "xml": "XML",
        "sql": "SQL",
        "csv": "CSV",
        "html": "HTML",
        "url": "URL",
        "uri": "URI",
        "uuid": "UUID",
        "jwt": "JWT",
        "db": "DB",
        "ui": "UI",
        "io": "IO",
    }

    def __init__(self, registry: Registry):
        """Initialize the generator.

        Args:
            registry: The registry instance to generate schema from
        """
        self.registry = registry

    @staticmethod
    def _format_namespace_title(namespace: str) -> str:
        """Format namespace for display in documentation.

        Handles both known library names and generic namespace formatting.

        Args:
            namespace: Raw namespace string (e.g., "np", "math", "api")

        Returns:
            Properly formatted display name

        Examples:
            >>> OpenAPIGenerator._format_namespace_title("np")
            'NumPy'
            >>> OpenAPIGenerator._format_namespace_title("math")
            'Math'
            >>> OpenAPIGenerator._format_namespace_title("api")
            'API'
            >>> OpenAPIGenerator._format_namespace_title("my_utils")
            'My_Utils'
        """
        # 1. Check known mappings first
        if namespace in OpenAPIGenerator.NAMESPACE_DISPLAY_NAMES:
            return OpenAPIGenerator.NAMESPACE_DISPLAY_NAMES[namespace]

        # 2. Heuristic for short acronyms (<=3 chars, all lowercase)
        #    e.g., "db" -> "DB", "ui" -> "UI"
        if len(namespace) <= 3 and namespace.islower() and namespace.isalpha():
            return namespace.upper()

        # 3. Title case for regular words
        #    Better than capitalize() for multi-word: "my_utils" -> "My_Utils"
        return namespace.title()

    def _python_type_to_openapi(self, py_type: Any) -> dict[str, Any]:
        """Convert Python type annotation to OpenAPI schema.

        Args:
            py_type: Python type annotation

        Returns:
            OpenAPI schema dictionary
        """
        # Handle None/NoneType
        if py_type is None or py_type is type(None):
            return {"type": "null"}

        # Handle basic types
        if py_type in self.TYPE_MAPPING:
            return {"type": self.TYPE_MAPPING[py_type]}

        # Handle generic types
        origin = get_origin(py_type)
        args = get_args(py_type)

        # Handle Optional[T] = Union[T, None]
        if origin is t.Union:
            # Filter out NoneType
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # Optional[T]
                schema = self._python_type_to_openapi(non_none_args[0])
                schema["nullable"] = True
                return schema
            else:
                # Union of multiple types
                return {
                    "oneOf": [self._python_type_to_openapi(arg) for arg in args]
                }

        # Handle List[T], Sequence[T], etc.
        if origin in (list, t.List, t.Sequence):
            if args:
                return {
                    "type": "array",
                    "items": self._python_type_to_openapi(args[0]),
                }
            return {"type": "array"}

        # Handle Dict[K, V], Mapping[K, V]
        if origin in (dict, t.Dict, t.Mapping):
            if len(args) >= 2:
                return {
                    "type": "object",
                    "additionalProperties": self._python_type_to_openapi(args[1]),
                }
            return {"type": "object"}

        # Handle Tuple[T1, T2, ...]
        if origin in (tuple, t.Tuple):
            if args:
                return {
                    "type": "array",
                    "items": {
                        "oneOf": [self._python_type_to_openapi(arg) for arg in args]
                    },
                    "minItems": len(args),
                    "maxItems": len(args),
                }
            return {"type": "array"}

        # Handle Literal[value1, value2, ...]
        if origin is t.Literal:
            return {"enum": list(args)}

        # Handle Any
        if py_type is t.Any:
            return {}  # No type constraint

        # Fallback: treat as object
        return {"type": "object"}

    def _extract_function_info(self, name: str, func: Callable) -> OpenAPIFunction:
        """Extract OpenAPI info from a Python function.

        Args:
            name: The bound name of the function
            func: The function object

        Returns:
            OpenAPIFunction with extracted metadata
        """
        # Get signature
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            # Built-in functions without signature
            return OpenAPIFunction(
                name=func.__name__,
                path=name,
                description=f"Built-in function: {name}",
            )

        # Extract docstring
        doc = inspect.getdoc(func)
        description = doc.split("\n")[0] if doc else None

        # Extract parameters
        parameters: list[OpenAPIParameter] = []
        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ("self", "cls"):
                continue

            # Get type annotation
            param_type = param.annotation
            if param_type is inspect.Parameter.empty:
                openapi_type = "object"  # Unknown type
            else:
                openapi_type_dict = self._python_type_to_openapi(param_type)
                openapi_type = openapi_type_dict.get("type", "object")

            # Check if required
            has_default = param.default is not inspect.Parameter.empty
            required = not has_default

            # Create parameter
            parameters.append(
                OpenAPIParameter(
                    name=param_name,
                    type=openapi_type,
                    required=required,
                    default=param.default if has_default else None,
                    description=None,  # Could extract from docstring
                )
            )

        # Extract return type
        return_annotation = sig.return_annotation
        if return_annotation is inspect.Signature.empty:
            return_type_str = None
        else:
            return_type_dict = self._python_type_to_openapi(return_annotation)
            return_type_str = return_type_dict.get("type")

        return OpenAPIFunction(
            name=func.__name__,
            path=name,
            description=description,
            parameters=parameters,
            return_type=return_type_str,
        )

    def _collect_functions(
        self, namespace: dict[str, Any], prefix: str = ""
    ) -> list[OpenAPIFunction]:
        """Recursively collect functions from namespace.

        Args:
            namespace: The namespace dictionary
            prefix: Path prefix for nested names

        Returns:
            List of OpenAPIFunction objects
        """
        functions: list[OpenAPIFunction] = []

        for name, value in namespace.items():
            full_name = f"{prefix}{name}" if prefix else name

            if isinstance(value, dict):
                # Nested namespace
                functions.extend(
                    self._collect_functions(value, prefix=f"{full_name}.")
                )
            elif callable(value):
                # Function
                try:
                    func_info = self._extract_function_info(full_name, value)
                    functions.append(func_info)
                except Exception:
                    # Skip functions that can't be introspected
                    pass

        return functions

    def generate_functions_list(self) -> list[OpenAPIFunction]:
        """Generate list of all bound functions with metadata.

        Returns:
            List of OpenAPIFunction objects
        """
        functions: list[OpenAPIFunction] = []

        # Get user-defined functions
        functions.extend(self._collect_functions(self.registry._objects))

        # Get default/built-in functions
        functions.extend(self._collect_functions(self.registry._default_objects))

        return functions

    def generate_openapi_schema(
        self,
        title: str = "Monic Framework API",
        version: str = "1.0.0",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Generate OpenAPI 3.0 schema for all bound functions.

        Args:
            title: API title
            version: API version
            description: API description

        Returns:
            OpenAPI 3.0 schema as dictionary
        """
        functions = self.generate_functions_list()

        # Build paths
        paths: dict[str, Any] = {}
        for func in functions:
            # Create path: /functions/{function_path}
            # Replace dots with slashes for REST-like paths
            path = f"/functions/{func.path.replace('.', '/')}"

            # Build parameters schema
            parameters_schema = {}
            required_params = []
            for param in func.parameters:
                param_schema = {"type": param.type}
                if param.description:
                    param_schema["description"] = param.description
                if param.default is not None:
                    param_schema["default"] = param.default

                parameters_schema[param.name] = param_schema
                if param.required:
                    required_params.append(param.name)

            # Build request body
            request_body = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": parameters_schema,
                        }
                    }
                },
            }

            if required_params:
                request_body["content"]["application/json"]["schema"][
                    "required"
                ] = required_params

            # Build response
            response_schema = {"type": func.return_type or "object"}
            responses = {
                "200": {
                    "description": "Successful response",
                    "content": {
                        "application/json": {
                            "schema": response_schema,
                        }
                    },
                }
            }

            # Create operation
            operation = {
                "summary": func.description or f"Execute {func.name}",
                "operationId": func.path.replace(".", "_"),
                "tags": [func.path.split(".")[0]] if "." in func.path else ["default"],
                "requestBody": request_body,
                "responses": responses,
            }

            # Add to paths
            paths[path] = {"post": operation}

        # Build complete schema
        schema = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": description
                or "API for Monic Framework bound functions",
            },
            "paths": paths,
            "components": {
                "schemas": {},  # Could add reusable schemas here
            },
        }

        return schema

    def generate_markdown_doc(self) -> str:
        """Generate Markdown documentation for all bound functions.

        Returns:
            Markdown formatted documentation
        """
        functions = self.generate_functions_list()

        # Group by namespace
        grouped: dict[str, list[OpenAPIFunction]] = {}
        for func in functions:
            namespace = func.path.split(".")[0] if "." in func.path else "builtin"
            if namespace not in grouped:
                grouped[namespace] = []
            grouped[namespace].append(func)

        # Generate markdown
        lines = ["# Monic Framework - Bound Functions", "", ""]

        for namespace in sorted(grouped.keys()):
            funcs = grouped[namespace]
            lines.append(f"## {self._format_namespace_title(namespace)}")
            lines.append("")

            for func in sorted(funcs, key=lambda f: f.path):
                # Function signature
                params_str = ", ".join(
                    f"{p.name}: {p.type}"
                    + (f" = {p.default}" if not p.required else "")
                    for p in func.parameters
                )
                return_str = f" -> {func.return_type}" if func.return_type else ""

                lines.append(f"### `{func.name}({params_str}){return_str}`")
                lines.append("")

                # Description
                if func.description:
                    lines.append(func.description)
                    lines.append("")

                # Parameters table
                if func.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    lines.append("| Name | Type | Required | Default | Description |")
                    lines.append("|------|------|----------|---------|-------------|")
                    for param in func.parameters:
                        default = str(param.default) if param.default else "-"
                        required = "✓" if param.required else ""
                        desc = param.description or "-"
                        lines.append(
                            f"| `{param.name}` | `{param.type}` | {required} | {default} | {desc} |"
                        )
                    lines.append("")

                # Return type
                if func.return_type:
                    lines.append(f"**Returns:** `{func.return_type}`")
                    lines.append("")

                lines.append("---")
                lines.append("")

        return "\n".join(lines)


def generate_openapi_spec(
    registry: Registry | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Convenience function to generate OpenAPI spec.

    Args:
        registry: Registry instance. If None, uses the global registry
                 (includes all @monic_bind decorated functions).
        **kwargs: Additional arguments for generate_openapi_schema

    Returns:
        OpenAPI 3.0 schema

    Examples:
        # Use global registry (default)
        >>> from monic.expressions.registry import monic_bind
        >>> @monic_bind("math.add")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>> spec = generate_openapi_spec()  # Includes add()

        # Use custom registry
        >>> from monic.expressions.registry import Registry
        >>> my_reg = Registry()
        >>> @my_reg.bind("custom.func")
        ... def custom_func(): ...
        >>> spec = generate_openapi_spec(registry=my_reg)
    """
    if registry is None:
        # Import global registry
        from monic.expressions.registry import registry as global_registry

        registry = global_registry

    generator = OpenAPIGenerator(registry)
    return generator.generate_openapi_schema(**kwargs)


def generate_markdown_docs(registry: Registry | None = None) -> str:
    """Convenience function to generate Markdown documentation.

    Args:
        registry: Registry instance. If None, uses the global registry
                 (includes all @monic_bind decorated functions).

    Returns:
        Markdown documentation

    Examples:
        # Use global registry (default)
        >>> docs = generate_markdown_docs()

        # Use custom registry
        >>> my_reg = Registry()
        >>> docs = generate_markdown_docs(registry=my_reg)
    """
    if registry is None:
        # Import global registry
        from monic.expressions.registry import registry as global_registry

        registry = global_registry

    generator = OpenAPIGenerator(registry)
    return generator.generate_markdown_doc()
