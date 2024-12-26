# Monic Framework

Monic Framework is a Python-based expression evaluation and code execution framework that provides a safe and flexible way to parse and interpret Python-like code. It offers a powerful expression parser and interpreter that can be used for dynamic code evaluation, scripting, and embedded programming scenarios.

## Features

- Python-like syntax support
- Safe code execution environment
- Expression parsing and interpretation
- Support for function definitions and calls
- Built-in type checking and validation
- Easy integration with existing Python projects

## Installation

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-tests.txt
```

## Quick Start

```python
import monic


# Initialize parser and interpreter
parser = monic.expressions.ExpressionsParser()
interpreter = monic.expressions.ExpressionsInterpreter()

# Parse and execute simple expressions
code = """
a = 1
b = 2
a + b
"""
tree = parser.parse(code)
result = interpreter.execute(tree)
print(result)  # Output: 3
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
