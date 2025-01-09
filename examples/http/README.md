# Compute API Example

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn main:app --reload
```

## API Endpoints

### GET Endpoint

- URL: `/compute`
- Query Parameters:
  - `input` (required): The expression to compute
  - `timeout` (optional, default: 10.0): Timeout for computation in seconds

Examples:

```bash
# Basic usage (URL-encoded)
curl "http://localhost:8000/compute?input=21%2B21"

# With custom timeout (URL-encoded)
curl "http://localhost:8000/compute?input=21%2B21&timeout=5.0"
```

```bash
# With function call
curl "http://localhost:8000/compute?input=say_hello(\"Monic\")&timeout=5.0"
```

### POST Endpoint

- URL: `/compute`
- Request Body (JSON):

  ```jsonc
  {
    "input": "string",  // Required: Expression to compute
    "timeout": float    // Optional: Timeout for computation (default: 10.0)
  }
  ```

Examples:

```bash
curl -X POST http://localhost:8000/compute \
     -H "Content-Type: application/json" \
     -d '{"input": "21 + 21", "timeout": 5.0}'
```

```bash
curl -X POST http://localhost:8000/compute \
     -H "Content-Type: application/json" \
     -d '{"input": "say_hello(\"Monic\")", "timeout": 5.0}'
```

### Response

The response is a JSON object with `success`, `error`, and `pods` fields.

```json
{
  "success": true,
  "error": null,
  "pods": [
    {
      "type": "int",
      "content": 42
    }
  ]
}
```

```json
{
  "success": true,
  "error": null,
  "pods": [
    {
      "type": "str",
      "content": "Hello, Monic!"
    }
  ]
}
```

## Notes

- The `input` field is required for both GET and POST requests
- You can optionally specify a `timeout` to limit computation time
- The API uses the Monic Framework's expression parser and interpreter
- For GET requests, use URL encoding for special characters like `+`
