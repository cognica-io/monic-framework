#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

"""
FastAPI Integration Example

Demonstrates how to integrate Monic OpenAPI generation with FastAPI,
creating a REST API that exposes Monic-bound functions.

Prerequisites:
    pip install fastapi uvicorn

Run:
    python fastapi_integration.py

Then visit:
    - http://localhost:8000/docs (Swagger UI)
    - http://localhost:8000/redoc (ReDoc)
    - http://localhost:8000/monic-spec (Generated OpenAPI spec)
"""

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    print("Error: FastAPI not installed")
    print("Install with: pip install fastapi uvicorn")
    exit(1)

from monic.expressions.registry import monic_bind
from monic.extra.openapi import generate_openapi_spec
from monic.expressions.interpreter import ExpressionsInterpreter


# Define Monic functions
@monic_bind("calculator.add")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@monic_bind("calculator.subtract")
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@monic_bind("calculator.multiply")
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@monic_bind("calculator.divide")
def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator
        b: Denominator (must not be zero)

    Returns:
        Result of division

    Raises:
        ZeroDivisionError: If b is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


@monic_bind("text.uppercase")
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@monic_bind("text.count_words")
def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# Create FastAPI app
app = FastAPI(
    title="Monic Calculator API",
    description="REST API powered by Monic Framework",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Monic Calculator API",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "monic_spec": "/monic-spec",
            "execute": "/execute/{function_path}",
        },
    }


@app.get("/monic-spec")
async def get_monic_spec():
    """Get Monic-generated OpenAPI specification.

    This endpoint returns the OpenAPI spec automatically generated
    from @monic_bind decorated functions.
    """
    spec = generate_openapi_spec(
        title="Monic Bound Functions",
        version="1.0.0",
        description="OpenAPI spec generated from Monic registry",
    )
    return JSONResponse(content=spec)


@app.post("/execute/{function_path}")
async def execute_function(function_path: str, params: dict[str, Any]):
    """Execute a Monic-bound function dynamically.

    Args:
        function_path: Dot-separated function path (e.g., "calculator.add")
        params: Function parameters as JSON

    Returns:
        Function execution result

    Example:
        POST /execute/calculator.add
        Body: {"a": 10, "b": 20}
        Response: {"result": 30}
    """
    try:
        # Build function call expression
        args = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        expression = f"{function_path}({args})"

        # Execute using Monic interpreter
        interpreter = ExpressionsInterpreter()
        result = interpreter.execute(interpreter.parse(expression))

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Convenience endpoints for specific functions
@app.post("/calculator/add")
async def api_add(a: int, b: int):
    """Add two integers (FastAPI native endpoint)."""
    return {"result": add(a, b)}


@app.post("/calculator/multiply")
async def api_multiply(a: int, b: int):
    """Multiply two integers (FastAPI native endpoint)."""
    return {"result": multiply(a, b)}


@app.post("/text/uppercase")
async def api_uppercase(text: str):
    """Convert text to uppercase (FastAPI native endpoint)."""
    return {"result": uppercase(text)}


def main():
    """Start FastAPI server."""
    import uvicorn

    print("=" * 70)
    print("Monic Framework - FastAPI Integration")
    print("=" * 70)
    print()
    print("Starting server...")
    print()
    print("Available endpoints:")
    print("  • http://localhost:8000/docs - Swagger UI")
    print("  • http://localhost:8000/redoc - ReDoc documentation")
    print("  • http://localhost:8000/monic-spec - Monic OpenAPI spec")
    print()
    print("Example requests:")
    print("  curl -X POST http://localhost:8000/execute/calculator.add \\")
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"a": 10, "b": 20}\'')
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
