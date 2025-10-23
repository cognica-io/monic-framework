#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

"""
Type Hints Showcase Example

Demonstrates comprehensive type hint support in OpenAPI generation,
including complex generic types, unions, optionals, and more.
"""

from typing import Any, Literal

from monic.expressions.registry import monic_bind
from monic.extra.openapi import generate_openapi_spec


# Basic types
@monic_bind("types.basic")
def basic_types(
    integer: int,
    floating: float,
    text: str,
    flag: bool,
) -> dict[str, Any]:
    """Demonstrate basic Python types.

    Args:
        integer: An integer value
        floating: A floating point number
        text: A string value
        flag: A boolean flag

    Returns:
        Dictionary containing all input values
    """
    return {
        "integer": integer,
        "floating": floating,
        "text": text,
        "flag": flag,
    }


# Optional types
@monic_bind("types.optional")
def optional_params(
    required: str,
    optional: int | None = None,
    with_default: str = "default",
) -> dict[str, Any]:
    """Demonstrate optional parameters.

    Args:
        required: Required parameter
        optional: Optional integer (can be None)
        with_default: Parameter with default value

    Returns:
        Dictionary of parameters
    """
    return {
        "required": required,
        "optional": optional,
        "with_default": with_default,
    }


# Collection types
@monic_bind("types.collections")
def collection_types(
    int_list: list[int],
    str_dict: dict[str, str],
    mixed_tuple: tuple[int, str, float],
) -> list[Any]:
    """Demonstrate collection types.

    Args:
        int_list: List of integers
        str_dict: Dictionary with string keys and values
        mixed_tuple: Tuple with mixed types

    Returns:
        Combined list of all inputs
    """
    return [int_list, str_dict, mixed_tuple]


# Union types
@monic_bind("types.union")
def union_types(
    int_or_str: int | str,
    multi_union: int | str | float,
) -> str:
    """Demonstrate union types.

    Args:
        int_or_str: Either an integer or string
        multi_union: Can be int, str, or float

    Returns:
        String representation of inputs
    """
    return f"{int_or_str} - {multi_union}"


# Literal types
@monic_bind("types.literal")
def literal_types(
    mode: Literal["read", "write", "append"],
    level: Literal[1, 2, 3],
) -> str:
    """Demonstrate literal types (enums).

    Args:
        mode: Operation mode (limited to specific values)
        level: Priority level (1, 2, or 3)

    Returns:
        Status message
    """
    return f"Mode: {mode}, Level: {level}"


# Nested generics
@monic_bind("types.nested")
def nested_generics(
    list_of_dicts: list[dict[str, int]],
    dict_of_lists: dict[str, list[float]],
) -> dict[str, Any]:
    """Demonstrate nested generic types.

    Args:
        list_of_dicts: List containing dictionaries
        dict_of_lists: Dictionary containing lists

    Returns:
        Processed data structure
    """
    return {
        "list_count": len(list_of_dicts),
        "dict_keys": list(dict_of_lists.keys()),
    }


# Complex return types
@monic_bind("types.complex_return")
def complex_return(data: str) -> dict[str, list[int]]:
    """Demonstrate complex return type.

    Args:
        data: Input data string

    Returns:
        Dictionary mapping strings to lists of integers
    """
    return {
        "lengths": [len(word) for word in data.split()],
        "indices": list(range(len(data))),
    }


# Optional with union
@monic_bind("types.optional_union")
def optional_union(
    value: int | str | None = None,
) -> str:
    """Demonstrate optional union type.

    Args:
        value: Can be int, str, or None

    Returns:
        Type description of the value
    """
    if value is None:
        return "None"
    return type(value).__name__


def main():
    """Generate OpenAPI spec showcasing all type conversions."""
    print("Monic Framework - Type Hints Showcase")
    print("=" * 70)
    print()

    # Generate spec
    spec = generate_openapi_spec(
        title="Type Hints Showcase API",
        version="1.0.0",
        description="Comprehensive demonstration of Python type hints in OpenAPI",
    )

    print("Type Conversions Demonstrated:")
    print()

    # Extract and display type mappings
    for path, definition in spec["paths"].items():
        func_name = path.split("/")[-1]
        operation = definition["post"]

        print(f"Function: {func_name}")
        print(f"  Summary: {operation['summary']}")

        # Show parameter types
        schema = operation["requestBody"]["content"]["application/json"][
            "schema"
        ]
        if "properties" in schema:
            print("  Parameters:")
            for param_name, param_schema in schema["properties"].items():
                param_type = param_schema.get("type", "complex")
                nullable = " (nullable)" if param_schema.get("nullable") else ""
                default = (
                    f" = {param_schema['default']}"
                    if "default" in param_schema
                    else ""
                )
                print(f"    - {param_name}: {param_type}{nullable}{default}")

        # Show return type
        return_schema = operation["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        return_type = return_schema.get("type", "complex")
        print(f"  Returns: {return_type}")
        print()

    # Save output
    import json

    with open("openapi_types.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print("✓ Saved: openapi_types.json")

    print()
    print("Key Takeaways:")
    print("  • Python type hints → OpenAPI types automatically")
    print("  • Optional types handled with nullable flag")
    print("  • Union types converted to oneOf schemas")
    print("  • Literal types become enum constraints")
    print("  • Nested generics fully supported")


if __name__ == "__main__":
    main()
