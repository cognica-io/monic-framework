#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

"""
Basic OpenAPI Generation Example

This example demonstrates the fundamental usage of OpenAPI generation
from Monic Framework bound functions.
"""

from monic.expressions.registry import monic_bind
from monic.extra.openapi import generate_openapi_spec, generate_markdown_docs


# Define functions using @monic_bind decorator
@monic_bind("math.add")
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        Sum of a and b
    """
    return a + b


@monic_bind("math.multiply")
def multiply(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Product of x and y
    """
    return x * y


@monic_bind("string.concat")
def concat(text1: str, text2: str, separator: str = " ") -> str:
    """Concatenate two strings with a separator.

    Args:
        text1: First string
        text2: Second string
        separator: Separator between strings (default: space)

    Returns:
        Concatenated string
    """
    return text1 + separator + text2


def main():
    """Generate OpenAPI specification and documentation."""
    print("Monic Framework - Basic OpenAPI Example")
    print("=" * 70)
    print()

    # Generate OpenAPI 3.0 specification
    # No registry argument needed - uses global registry automatically
    spec = generate_openapi_spec(
        title="Math & String API",
        version="1.0.0",
        description="Simple API for mathematical and string operations",
    )

    print(f"Generated OpenAPI spec with {len(spec['paths'])} endpoints")
    print()

    # Show available endpoints
    print("Available endpoints:")
    for path in spec["paths"]:
        print(f"  POST {path}")
    print()

    # Generate Markdown documentation
    markdown = generate_markdown_docs()

    # Save outputs
    import json

    with open("openapi_basic.json", "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print("✓ Saved: openapi_basic.json")

    with open("api_docs_basic.md", "w", encoding="utf-8") as f:
        f.write(markdown)
    print("✓ Saved: api_docs_basic.md")

    print()
    print("Next steps:")
    print("  1. Open openapi_basic.json in Swagger Editor")
    print("     → https://editor.swagger.io/")
    print("  2. Import into Postman for API testing")
    print("  3. Generate client SDKs using openapi-generator")


if __name__ == "__main__":
    main()
