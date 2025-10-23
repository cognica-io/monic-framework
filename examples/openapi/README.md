# Monic Framework - OpenAPI Examples

This directory contains comprehensive examples demonstrating OpenAPI specification generation from Monic Framework bound functions.

## Overview

Monic Framework can automatically generate OpenAPI 3.0 specifications from functions decorated with `@monic_bind`. This enables:

- **Automatic API documentation** - No manual spec writing
- **Type safety** - Leverages Python type hints
- **Client SDK generation** - Use openapi-generator for any language
- **API gateway integration** - Import specs into AWS, Kong, etc.
- **Interactive testing** - Swagger UI, Postman, etc.

## Examples

### 1. `basic.py` - Basic Usage

**What it demonstrates:**
- Simple function binding with `@monic_bind`
- Automatic OpenAPI spec generation
- Saving JSON and Markdown output

**Key concepts:**
- Global registry (default behavior)
- Type hint extraction
- Docstring parsing

**Run:**
```bash
python basic.py
```

**Output:**
- `openapi_basic.json` - OpenAPI 3.0 specification
- `api_docs_basic.md` - Markdown documentation

---

### 2. `types.py` - Type Hints Showcase

**What it demonstrates:**
- Comprehensive type hint support
- Basic types: `int`, `float`, `str`, `bool`
- Optional types: `int | None`
- Collection types: `list[int]`, `dict[str, str]`, `tuple[int, str]`
- Union types: `int | str`
- Literal types: `Literal["read", "write"]`
- Nested generics: `list[dict[str, int]]`

**Type conversion examples:**

| Python Type | OpenAPI Schema |
|------------|----------------|
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `str` | `{"type": "string"}` |
| `bool` | `{"type": "boolean"}` |
| `list[int]` | `{"type": "array", "items": {"type": "integer"}}` |
| `dict[str, int]` | `{"type": "object", "additionalProperties": {"type": "integer"}}` |
| `int \| None` | `{"type": "integer", "nullable": true}` |
| `int \| str` | `{"oneOf": [{"type": "integer"}, {"type": "string"}]}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |

**Run:**
```bash
python types.py
```

---

### 3. `fastapi_integration.py` - FastAPI Integration

**What it demonstrates:**
- Integration with FastAPI web framework
- Serving Monic-generated OpenAPI spec
- Dynamic function execution via REST API
- Automatic Swagger UI

**Prerequisites:**
```bash
pip install fastapi uvicorn
```

**Run:**
```bash
python fastapi_integration.py
```

**Access:**
- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc
- http://localhost:8000/monic-spec - Monic OpenAPI spec

**Example request:**
```bash
curl -X POST http://localhost:8000/execute/calculator.add \
  -H "Content-Type: application/json" \
  -d '{"a": 10, "b": 20}'
```

---

### 4. `multi_registry.py` - Multiple Registries

**What it demonstrates:**
- Environment separation (prod/dev)
- Multi-tenant systems
- Plugin architectures
- Global vs custom registries

**Scenarios covered:**

**Scenario 1: Environment Separation**
```python
prod_registry = Registry()
dev_registry = Registry()

@prod_registry.bind("api.process")
def prod_process(): ...

@dev_registry.bind("api.process")
def dev_process(): ...  # Different implementation
```

**Scenario 2: Multi-tenant**
```python
tenant_a_registry = Registry()
tenant_b_registry = Registry()  # Premium features

spec_a = generate_openapi_spec(registry=tenant_a_registry)
spec_b = generate_openapi_spec(registry=tenant_b_registry)
```

**Scenario 3: Plugins**
```python
core_registry = Registry()
email_plugin_registry = Registry()
payment_plugin_registry = Registry()

# Generate separate specs for each plugin
```

**Run:**
```bash
python multi_registry.py
```

---

## Quick Start

### Basic Usage (Global Registry)

```python
from monic.expressions.registry import monic_bind
from monic.extra.openapi import generate_openapi_spec

@monic_bind("math.add")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Generate spec (no registry argument needed!)
spec = generate_openapi_spec()
```

### Custom Registry

```python
from monic.expressions.registry import Registry
from monic.extra.openapi import generate_openapi_spec

my_registry = Registry()

@my_registry.bind("custom.func")
def custom_func(x: int) -> int:
    return x * 2

# Explicit registry
spec = generate_openapi_spec(registry=my_registry)
```

---

## Generated Output

### OpenAPI JSON Structure

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "My API",
    "version": "1.0.0",
    "description": "API description"
  },
  "paths": {
    "/functions/math/add": {
      "post": {
        "summary": "Add two integers.",
        "operationId": "math_add",
        "tags": ["math"],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "a": {"type": "integer"},
                  "b": {"type": "integer"}
                },
                "required": ["a", "b"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response",
            "content": {
              "application/json": {
                "schema": {"type": "integer"}
              }
            }
          }
        }
      }
    }
  }
}
```

### Markdown Documentation

```markdown
## Math

### `add(a: integer, b: integer) -> integer`

Add two integers.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `a` | `integer` | ✓ | - | - |
| `b` | `integer` | ✓ | - | - |

**Returns:** `integer`
```

---

## Usage with External Tools

### Swagger Editor

1. Generate spec: `python basic.py`
2. Open https://editor.swagger.io/
3. Import `openapi_basic.json`
4. View interactive documentation

### Postman

1. Generate spec
2. Postman → Import → OpenAPI 3.0
3. Select generated JSON file
4. Test endpoints interactively

### OpenAPI Generator (Client SDKs)

Generate TypeScript client:
```bash
openapi-generator-cli generate \
  -i openapi_basic.json \
  -g typescript-axios \
  -o ./client
```

Generate Python client:
```bash
openapi-generator-cli generate \
  -i openapi_basic.json \
  -g python \
  -o ./client
```

Supported languages: Java, Go, Ruby, PHP, C#, Kotlin, Swift, Rust, and 50+ more.

### AWS API Gateway

1. AWS Console → API Gateway
2. Create API → Import from OpenAPI 3.0
3. Upload generated JSON
4. Deploy API

---

## API Reference

### `generate_openapi_spec()`

```python
def generate_openapi_spec(
    registry: Registry | None = None,
    title: str = "Monic Framework API",
    version: str = "1.0.0",
    description: str | None = None,
) -> dict[str, Any]:
    """Generate OpenAPI 3.0 specification.

    Args:
        registry: Registry instance (None = use global registry)
        title: API title
        version: API version
        description: API description

    Returns:
        OpenAPI 3.0 specification dictionary
    """
```

### `generate_markdown_docs()`

```python
def generate_markdown_docs(
    registry: Registry | None = None
) -> str:
    """Generate Markdown documentation.

    Args:
        registry: Registry instance (None = use global registry)

    Returns:
        Markdown formatted documentation
    """
```

---

## Best Practices

### 1. Use Type Hints

**Good:**
```python
@monic_bind("api.process")
def process(data: list[int], threshold: int = 0) -> list[int]:
    return [x for x in data if x > threshold]
```

**Bad:**
```python
@monic_bind("api.process")
def process(data, threshold=0):  # No types!
    return [x for x in data if x > threshold]
```

### 2. Write Docstrings

**Good:**
```python
@monic_bind("api.validate")
def validate(email: str) -> bool:
    """Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    return "@" in email
```

### 3. Use Descriptive Names

**Good:**
```python
@monic_bind("user.authentication.validate_credentials")
def validate_credentials(username: str, password: str) -> bool:
    ...
```

**Bad:**
```python
@monic_bind("validate")
def validate(u: str, p: str) -> bool:
    ...
```

### 4. Group Related Functions

```python
# User management
@monic_bind("user.create")
def create_user(): ...

@monic_bind("user.update")
def update_user(): ...

@monic_bind("user.delete")
def delete_user(): ...

# Authentication
@monic_bind("auth.login")
def login(): ...

@monic_bind("auth.logout")
def logout(): ...
```

---

## Common Patterns

### Pattern 1: Validation Functions

```python
@monic_bind("validation.email")
def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email

@monic_bind("validation.phone")
def validate_phone(phone: str) -> bool:
    """Validate phone number."""
    return phone.isdigit() and len(phone) >= 10
```

### Pattern 2: Data Transformation

```python
@monic_bind("transform.normalize")
def normalize(data: list[float]) -> list[float]:
    """Normalize data to 0-1 range."""
    min_val, max_val = min(data), max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]
```

### Pattern 3: Business Logic

```python
@monic_bind("business.calculate_discount")
def calculate_discount(
    price: float,
    customer_tier: Literal["bronze", "silver", "gold"]
) -> float:
    """Calculate discount based on customer tier."""
    discounts = {"bronze": 0.05, "silver": 0.10, "gold": 0.15}
    return price * (1 - discounts[customer_tier])
```

---

## Troubleshooting

### Issue: Functions not appearing in spec

**Cause:** Using wrong registry

**Solution:**
```python
# If using @monic_bind, don't pass registry
spec = generate_openapi_spec()  # Uses global registry

# If using custom registry, pass it explicitly
my_registry = Registry()
spec = generate_openapi_spec(registry=my_registry)
```

### Issue: Type shows as "object" instead of specific type

**Cause:** Missing or incorrect type hints

**Solution:**
```python
# Add proper type hints
def func(x: int) -> str:  # ✓ Good
    ...

def func(x):  # ✗ Bad - no type hint
    ...
```

### Issue: Optional parameter not marked as optional

**Cause:** Missing default value

**Solution:**
```python
def func(required: int, optional: int | None = None):  # ✓ Good
    ...

def func(required: int, optional: int | None):  # ✗ Bad - no default
    ...
```

---

## Advanced Topics

### Custom Type Converters

For custom types not automatically handled:

```python
from monic.extra.openapi import OpenAPIGenerator

# Extend generator
class CustomOpenAPIGenerator(OpenAPIGenerator):
    def _python_type_to_openapi(self, py_type):
        if py_type is MyCustomType:
            return {"type": "string", "format": "custom"}
        return super()._python_type_to_openapi(py_type)
```

### Versioned APIs

```python
v1_registry = Registry()
v2_registry = Registry()

@v1_registry.bind("api.process")
def process_v1(data: str) -> str:
    return data.upper()

@v2_registry.bind("api.process")
def process_v2(data: str, encoding: str = "utf-8") -> str:
    return data.upper()

spec_v1 = generate_openapi_spec(registry=v1_registry, version="1.0.0")
spec_v2 = generate_openapi_spec(registry=v2_registry, version="2.0.0")
```

---

## Further Reading

- [OpenAPI Specification](https://swagger.io/specification/)
- [Swagger Editor](https://editor.swagger.io/)
- [OpenAPI Generator](https://openapi-generator.tech/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## License

Copyright (c) 2024-2025 Cognica, Inc.
